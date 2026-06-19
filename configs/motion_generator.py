"""Motion generator (dual-branch DiT1D) backbone and training hyperparameters
for Humanoid-DART.

Referenced in the Appendix of the paper in place of the
"Motion Generator Backbone" and "Motion Generator Training" sections.
"""

backbone = dict(
    hidden_size=256,
    num_transformer_blocks=8,                # total, 4 per branch
    blocks_per_branch=4,
    num_attention_heads=4,
    mlp_ratio=4.0,
    positional_embedding="learned_1d",       # with group-wise feature embeddings
    history_aggregation="attention",
    conditioning="adaln_zero",               # goal + state conditioning
    history_frames=4,                        # t_h
    segment_horizon=50,                       # p, denoised frames per segment
    reference_dt=0.02,                        # seconds
)

training = dict(
    objective="v_prediction",
    beta_schedule="sigmoid",
    num_train_noise_steps=200,
    batch_size=1024,
    learning_rate=5e-4,
    ema_decay=0.9995,
    mirror_symmetry_prob=0.5,
    # classifier-free guidance: random condition dropout during training
    state_condition_dropout=0.1,
    goal_condition_dropout=0.2,
    # auxiliary loss terms: L = L_diff + lambda_s * L_smooth + lambda_g * L_grasp
    temporal_smoothness_weight=0.1,          # lambda_s
    grasp_consistency_weight=0.25,           # lambda_g
)
