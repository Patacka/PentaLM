#!/bin/bash
python -m lamorel_launcher.launch \
       --config-dir . \
       --config-path $(realpath .) \
       --config-name config \
       rl_script_args.path=$(realpath main.py)