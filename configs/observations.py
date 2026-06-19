"""Motion-tracking policy observation terms for Humanoid-DART.

Referenced in the Appendix of the paper in place of the
"Motion-tracking policy observations" table.

Each entry: (description, dim, groups), where `groups` is the set of
networks that receive the term -- "actor" and/or "critic" (privileged).
"""

# Five future reference frames at these step offsets are appended to the
# current reference term.
reference_future_frame_offsets = [2, 3, 5, 17, 33]

observations = {
    "reference": dict(
        description="Current reference [q_ref, qdot_ref] plus five future "
                    "reference frames at the offsets above; each frame holds "
                    "the relative anchor position, 6D orientation, and "
                    "reference joint positions and velocities.",
        dim=460, groups=("actor", "critic"),
    ),
    "body_pos": dict(
        description="Tracked-body positions in the current robot anchor frame.",
        dim=42, groups=("critic",)),
    "body_ori": dict(
        description="Tracked-body orientations (6D) in the current robot anchor frame.",
        dim=84, groups=("critic",)),
    "base_lin_vel": dict(
        description="Robot IMU linear velocity.",
        dim=3, groups=("actor", "critic")),
    "base_ang_vel": dict(
        description="Robot IMU angular velocity.",
        dim=3, groups=("actor", "critic")),
    "joint_pos": dict(
        description="Joint positions relative to the default pose.",
        dim=29, groups=("actor", "critic")),
    "joint_vel": dict(
        description="Joint velocities relative to the default baseline.",
        dim=29, groups=("actor", "critic")),
    "actions": dict(
        description="Last action applied by the policy (one per actuated joint).",
        dim=29, groups=("actor", "critic")),
    "object_global_pos": dict(
        description="Reference object position in the current robot anchor frame.",
        dim=3, groups=("actor", "critic")),
    "object_global_ori": dict(
        description="Reference object orientation in the current robot anchor frame.",
        dim=3, groups=("actor", "critic")),
}
