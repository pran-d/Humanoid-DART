"""Pipeline and curriculum hyperparameters for Humanoid-DART.

Referenced in the Appendix of the paper in place of the
"Pipeline and curriculum hyperparameters" table.
"""

pipeline = dict(
    generations=4,
    iterations_per_generation=10,
    candidates_sampled_per_iteration=3000,
    elite_fitness_threshold=5e-4,            # tau_f
    goal_relabeling_neighbors=5,             # k
    goal_relabeling_tolerance_radius=0.02,
    max_elites_for_generator_retraining=200,
    generator_epochs_per_iteration=20,
    generator_epochs_initial_fit=40,
    generation_chunk_size=3000,
)

curriculum = dict(
    refine_threshold=0.01,                   # delta_ref
    explore_threshold=0.025,                 # delta_exp
    frontier_sampling_probability=0.01,
    bin_resolution=0.02,                     # meters
)
