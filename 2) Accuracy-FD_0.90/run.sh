#!/bin/bash

python main.py --model_dir=R_0818 \
        --max_timesteps=15 \
        --max_episodes_num=30000 \
        --start_episodes=100 \
        --eval_freq_episode=20 \
        --batch_size=256 \
        --discount=0.95 \
        --victim_n_episodes=80 \
        --ae_n_epochs=10 \