# Run as ./scripts/***.sh
python train.py \
    --algorithm ppo \
    --lunarlander \
    --n_envs 16 \
    --n_updates 500 \
    --n_steps_per_update 128 \
    --hidden_size 128 \
    --gamma 0.999 \
    --lam 0.95 \
    --ent_coef 0.001 \
    --actor_lr 1e-3 \
    --critic_lr 5e-3 \
    --entropy_lr 1e-3 \
    --batch_size 64 \
    --n_ppo_epochs 4 \
    --clip_param 0.1 \
    --lr_scheduler linear \
    ${@:1}
