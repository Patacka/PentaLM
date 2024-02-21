import math
from tqdm import tqdm
import torch
from transformers import set_seed as set_transformers_seed
import numpy as np
import scipy
import hydra
from lamorel import BaseUpdater, BaseModuleFunction, Caller, lamorel_init
from lamorel.server.llms.module_functions import LogScoringModuleFn
from omegaconf import DictConfig
from pentestenv import PentestEnvLLM
import logging
import matplotlib.pyplot as plt
import os
lamorel_init()

# Adopted from Lamorel/examples/PPO_finetuning
class ValueHeadModuleFn(BaseModuleFunction):
    def __init__(self, pre_encoded_input: bool):
        super().__init__()
        self._pre_encoded_input = pre_encoded_input

    def _find_llm_hidden_size(self) -> int:
      if 'hidden_size' in self.llm_config.attribute_map:
          _hidden_size_key = self.llm_config.attribute_map['hidden_size']
      else:
          if "word_embed_proj_dim" in self.llm_config.to_dict():
              _hidden_size_key = "word_embed_proj_dim"
          elif "hidden_size" in self.llm_config.to_dict():
              _hidden_size_key = "hidden_size"
          else:
              raise NotImplementedError("Unknown hidden size key")
      return self.llm_config.to_dict()[_hidden_size_key]

    def initialize(self):
        self._llm_hidden_size = self._find_llm_hidden_size()
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(self._llm_hidden_size, 1024),
            torch.nn.Sigmoid(), # ReLU kann in manchen Fällen bessere Ergebnisse liefern als Sigmoid
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(), # Erneut ReLU für Konsistenz
            torch.nn.Linear(1024, 1),
        ).to(self.device, dtype=torch.float16)

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs) -> torch.Tensor:
        if self._pre_encoded_input:
            end_of_context_position = 0
        else:
            end_of_context_position = len(
                tokenized_contexts[0]["input_ids"])  # inputs are padded so all of same size

        model_head = forward_outputs['hidden_states'][-1][:, end_of_context_position, :]

        return self.value_head_op(model_head.to(self.device))
    

class PPOUpdater(BaseUpdater):
    def __init__(self, config: DictConfig):
        super(PPOUpdater, self).__init__()
        self.minibatch_size = config.minibatch_size
        self.lr = config.lr
        self.clip_eps = config.clip_eps
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, 'optimizer'):
            self._iterator_named_trainable_params = self._llm_module.named_parameters
            self._iterator_trainable_params = (p for n, p in self._iterator_named_trainable_params())
            self.optimizer = torch.optim.SGD(self._iterator_trainable_params, lr=self.lr)

        current_process_buffer = {}
        print(_current_batch_ids)
        print("loop beginnt")
        for x in _current_batch_ids:
            for k in ['actions', 'advantages', 'returns', 'logprobs', 'values']:
                current_process_buffer[k] = kwargs[k][x]


        epoch_losses = {
            "value": [],
            "policy": [],
            "loss": []
        }

        n_minibatches = math.ceil(len(contexts) / self.minibatch_size)
        for step in range(n_minibatches):
            _start_idx = step * self.minibatch_size
            _stop_idx = min((step + 1) * self.minibatch_size, len(contexts))

            _contexts = contexts[_start_idx:_stop_idx]
            _candidates = candidates[_start_idx:_stop_idx]

            output = self._llm_module(['score', 'value'], contexts=_contexts, candidates=_candidates,
                                      require_grad=True, minibatch_size=self.minibatch_size)
            scores = torch.stack([_o['score'] for _o in output]).squeeze()
            probas = torch.distributions.Categorical(logits=scores)
            values = torch.stack([_o["value"][0] for _o in output]).squeeze()
            scores = scores.to(self.device)
            # Compute policy loss
            entropy = probas.entropy().mean()
            log_prob = probas.log_prob(current_process_buffer['actions'][_start_idx:_stop_idx])
            #print("Scores")
            #print(scores)
            #print("logprobs: ")
            #print(current_process_buffer['logprobs'][_start_idx:_stop_idx])
            ratio = torch.exp(log_prob - current_process_buffer['logprobs'][_start_idx:_stop_idx])
            #print("ratio")
            #print(ratio)
            assert not (step == 0 and (torch.any(ratio < 0.95) or torch.any(ratio > 1.15))), "PPO ratio != 1 !!"

            clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * current_process_buffer['advantages'][_start_idx:_stop_idx]
            policy_loss = -(torch.min(ratio * current_process_buffer['advantages'][_start_idx:_stop_idx], clip_adv)).mean()
            epoch_losses["policy"].append(policy_loss.detach().cpu().item())

            # Compute value loss
            unclipped_value_error = ((values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
            clipped_values = current_process_buffer['values'][_start_idx:_stop_idx] + \
                              torch.clamp(values - current_process_buffer['values'][_start_idx:_stop_idx],
                                              -self.clip_eps, self.clip_eps)
            clipped_value_error = ((clipped_values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
            value_loss = torch.max(unclipped_value_error, clipped_value_error).mean()
            epoch_losses["value"].append(value_loss.detach().cpu().item())

            # Compute final loss
            loss = policy_loss - entropy * self.entropy_coef  + self.value_loss_coef * value_loss
            epoch_losses["loss"].append(loss.detach().cpu().item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._iterator_trainable_params, self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()

        return {'loss': np.mean(epoch_losses["loss"]), 'value_loss': np.mean(epoch_losses["value"]),
                'policy_loss': np.mean(epoch_losses["policy"])}


# Adopted from lamorel/examples/PPO_finetuning
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# Adopted from lamorel/examples/PPO_finetuning
class PPOBuffer:
    def __init__(self, size, gamma=0.99, lambda_gae=0.95):
        self.obs_buf = [None for _ in range(size)]
        self.act_buf = [None for _ in range(size)]
        self.possible_act_buf = [None for _ in range(size)]
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lambda_gae = gamma, lambda_gae
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, possible_act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.possible_act_buf[self.ptr] = possible_act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lambda_gae)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(obs=self.obs_buf, act=self.act_buf, possible_act=self.possible_act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf)
        return {
            k: torch.as_tensor(v, dtype=torch.float32)
            if not isinstance(v, list) else v
            for k, v in data.items()
        }
    

def reset_history():
    return {
        "ep_len": [],
        "ep_ret": [],
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "actions": [],
        "observations": [],
    }

def plot_rewards_steps(loss, dateiname):
    plt.figure(figsize=(10, 6))
    plt.plot(loss, marker='o', linestyle='-', color='b')
    plt.title('Loss pro Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Sicherstellen, dass das Verzeichnis für den Dateinamen existiert
    if not os.path.exists(os.path.dirname(dateiname)):
        os.makedirs(os.path.dirname(dateiname), exist_ok=True)

    plt.savefig(dateiname)
    plt.close()


@hydra.main(config_path='config', config_name='config')
def main(config: DictConfig):
    seed = config.rl_script_args.seed   
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_transformers_seed(seed)

    agent = Caller(config.lamorel_args,
                custom_updater=PPOUpdater(config.PPO_updater),
                custom_module_functions={
                    'score': LogScoringModuleFn(
                        0, # pad_token
                        config.lamorel_args.llm_args.model_type,
                        config.lamorel_args.llm_args.pre_encode_inputs
                    ),
                    'value': ValueHeadModuleFn(
                        config.lamorel_args.llm_args.pre_encode_inputs
                    )
                }
        )
    
    rl_script_logger = logging.getLogger('rl_script_logger')
    env = PentestEnvLLM(config)

    buffer = PPOBuffer(config.rl_script_args.steps_per_epoch, config.PPO_updater.gamma, config.PPO_updater.lambda_gae)

    task_prompt_template = (
        "Based on this observation: '{0}' "
        "generate concise the next action for pentesting IP the {1}? Provide only the action itself including "
        "the IP, without any additional characters, explanation."
        "Your Goal is to find and open the flag.txt file by exploit a vulnerability on the target system."
    )

    IP_under_test = "10.10.166.243"

    choose_prompt_template= (
        "Based on this observation: '{0}' and on these possible actions '{1}' choose one that help you to expoloit a vulnerability on "
        "the target system and  find and open the flag.txt"
    )

    episode_return = 0
    episode_length = 0
    observation = env.reset()
    history = reset_history()
    ZERO_AS_BS_IS_1 = 0
    for epoch in range(config.rl_script_args.epochs):
        for t in tqdm(range(config.rl_script_args.steps_per_epoch), ascii=" " * 9 + ">", ncols=100):
            task_prompt = task_prompt_template.format(observation, IP_under_test)

            rl_script_logger.info("Generating action...")
            action_space = []
            for x in range(3):
                action_space.append(agent.generate([task_prompt], **config.lamorel_args.llm_args.generation_args)[ZERO_AS_BS_IS_1][0]['text'])
            choose_prompt = choose_prompt_template.format(action_space, observation)
            action = agent.generate([choose_prompt], **config.lamorel_args.llm_args.generation_args)[ZERO_AS_BS_IS_1][0]['text']
            rl_script_logger.info(f"Generated action: {action}")
            rl_script_logger.info(f"Epoch Step: {t}")
            action = action.replace(choose_prompt, '')
            lines = action.split('\n')
            action = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            action = action.replace("<|im_end|>", '')

            score_and_value = agent.custom_module_fns(['score', 'value'], contexts=[choose_prompt], candidates=[action])[ZERO_AS_BS_IS_1]
            observation, reward, done, _ = env.step(action)

            epoch_ended = t+1 == config.rl_script_args.steps_per_epoch
            buffer.store(task_prompt, [action], [action_space], reward, score_and_value['value'][0], score_and_value['score'][0])
            episode_return += reward
            episode_length += 1
            timeout = episode_length == config.rl_script_args.max_episode_length
            terminal = done or timeout

            if terminal or epoch_ended:
                if not terminal:
                    next_task_prompt = task_prompt_template.format(observation, IP_under_test)
                    value = agent.custom_module_fns(module_function_keys=['value'], contexts=[next_task_prompt], candidates=[[action]])[ZERO_AS_BS_IS_1]['value'][0]
                    buffer.finish_path(value[0].cpu())
                else:
                    buffer.finish_path(0)
                    history["ep_len"].append(episode_length)
                    history["ep_ret"].append(episode_return)
                    episode_length, episode_return = 0, 0

        rl_script_logger.info(f"PPO update number {epoch + 1}")
        trajectories = buffer.get()
        update_results = agent.update(trajectories['obs'],
                                        trajectories['possible_act'],
                                        actions=trajectories['act'],
                                        returns=trajectories['ret'],
                                        advantages=trajectories['adv'],
                                        logprobs=trajectories['logp'],
                                        values=trajectories['val'],
                                        )[ZERO_AS_BS_IS_1]
        history["loss"].append(update_results['loss'])
        history["policy_loss"].append(update_results['policy_loss'])
        history["value_loss"].append(update_results['value_loss'])
        history["actions"].extend(trajectories['act'])
        history["observations"].extend(trajectories['obs'])
        rl_script_logger.info(f"Update loss: {update_results['loss']}")
    plot_rewards_steps(history["loss"], "graphs/loss_plot.png")
    
    agent.close()


if __name__ == '__main__':
    main()