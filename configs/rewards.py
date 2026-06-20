"""Motion-tracker reward terms for Humanoid-DART.

Referenced in the paper in place of the "Motion tracker reward terms" table.
Each term is exp(-||.||^2 / sigma^2) unless `sigma` is None. Negative
weights denote penalties.
"""

rewards = {
    "motion_tracking": {
        "global_root_position":      dict(weight=0.5,  sigma=0.30),
        "global_root_orientation":   dict(weight=0.5,  sigma=0.40),
        "relative_body_position":    dict(weight=1.0,  sigma=0.30),
        "relative_body_orientation": dict(weight=1.0,  sigma=0.40),
        "body_linear_velocity":      dict(weight=1.0,  sigma=1.00),
        "body_angular_velocity":     dict(weight=1.0,  sigma=3.14),
    },
    "object_interaction": {
        "eef_contact_match":         dict(weight=2.0,  sigma=None),  # binary contact match
        "object_global_position":    dict(weight=1.0,  sigma=0.25),
        "object_global_orientation":  dict(weight=0.65, sigma=0.40),
        "relative_eef_position":      dict(weight=1.1,  sigma=0.10),
    },
    "regularization": {
        "action_rate_l2":            dict(weight=-0.1,   sigma=None),
        "joint_limit_violation":     dict(weight=-10.0,  sigma=None),
        "self_collision":            dict(weight=-10.0,  sigma=None),
        "bad_termination":           dict(weight=-100.0, sigma=None),
    },
}
