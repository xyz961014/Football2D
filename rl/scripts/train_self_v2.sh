# Run as ./scripts/***.sh
python train.py \
    --algorithm ppo \
    --env_name SelfTraining-v2 \
    --n_envs 64 \
    --n_updates 3000 \
    --n_steps_per_update 1024 \
    --randomize_domain \
    --model_name world \
    --hidden_size 512 \
    --gamma 0.999 \
    --lam 0.95 \
    --ent_coef 0.001 \
    --world_lr 1e-5 \
    --actor_lr 1e-5 \
    --critic_lr 1e-5 \
    --entropy_lr 1e-3 \
    --batch_size 4096 \
    --n_ppo_epochs 4 \
    --clip_param 0.1 \
    --lr_scheduler linear \
    --use_auxiliary_reward \
    --auxiliary_reward_type no_kick_reward \
    --use_cuda \
    --learn_to_kick \
    --relative_obs \
    ${@:1}
