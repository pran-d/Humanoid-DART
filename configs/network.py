"""Actor-critic network architecture for the Humanoid-DART tracking controller.

Referenced in the Appendix of the paper in place of the
"Actor-critic network architecture" table.
"""

network = dict(
    architecture="MLP actor + MLP critic",
    actor_hidden_layers=[512, 256, 128],
    critic_hidden_layers=[512, 256, 128],
    activation="elu",
    initial_action_noise_std=1.0,            # sigma_0
    actor_observation_normalization=True,
    critic_observation_normalization=True,
    action_clipping=None,
)
