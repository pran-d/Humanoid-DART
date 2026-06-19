"""Domain randomization (DR) and reference state initialization (RSI) for
Humanoid-DART tracking training.

Referenced in the Appendix of the paper in place of the
"Domain randomisation and reference state initialisation" table.
Entries marked `box_only=True` apply only to box-manipulation task variants.
"""

domain_randomization = {
    "push_robot": dict(
        description="Random velocity impulse applied to the robot body.",
        linear="+/-0.5 m/s", angular="+/-0.52 rad/s"),
    "base_com_offset": dict(
        description="Additive noise to torso link inertial position (startup).",
        x="+/-0.025 m", yz="+/-0.05 m"),
    "joint_default_pos": dict(
        description="Additive noise to all joint default positions (startup).",
        range="+/-0.01 rad"),
    "foot_friction": dict(
        description="Absolute foot geom friction coefficient (startup).",
        range=[0.8, 1.1]),
    "push_object": dict(
        description="Random velocity impulse applied to the box.",
        linear="+/-0.5 m/s", angular="+/-1.5 rad/s", box_only=True),
    "object_mass": dict(
        description="Scale factor on box body mass (startup).",
        range=[0.7, 1.3], box_only=True),
    "hand_friction": dict(
        description="Absolute wrist geom friction coefficient (startup).",
        range=[0.9, 1.1], box_only=True),
}

reference_state_initialization = {
    "root_position": dict(
        description="Per-episode random offset to initial root position.",
        xy="+/-0.05 m", z="+/-0.01 m"),
    "root_orientation": dict(
        description="Per-episode random offset to initial root orientation.",
        range="+/-0.1"),
    "root_linear_velocity": dict(
        description="Per-episode random offset to initial linear velocity.",
        xyz="+/-0.5 m/s"),
    "root_angular_velocity": dict(
        description="Per-episode random offset to initial angular velocity.",
        range="+/-0.52 rad/s"),
    "joint_positions": dict(
        description="Per-episode random offset to initial joint positions.",
        range="+/-0.1 rad"),
}
