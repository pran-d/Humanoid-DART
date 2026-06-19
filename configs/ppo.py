"""PPO algorithm hyperparameters for the Humanoid-DART tracking controller.

Referenced in the Appendix of the paper in place of the
"PPO algorithm hyperparameters" table.
"""

ppo = dict(
    num_envs=4096,
    steps_per_env_per_update=24,
    learning_epochs_per_update=5,
    mini_batches_per_epoch=4,
    learning_rate=1e-3,
    lr_schedule="adaptive_kl",               # KL-target adaptive schedule
    gamma=0.99,                              # discount factor
    gae_lambda=0.95,
    clip_param=0.2,                          # PPO clip epsilon
    value_loss_coef=1.0,
    entropy_coef=0.005,
    desired_kl=0.01,
    max_grad_norm=1.0,
    clipped_value_loss=True,
)
