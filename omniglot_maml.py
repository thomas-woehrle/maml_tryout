from maml import maml_learner  # TODO


theta = maml_learner(task_sampler, model, checkpoint_fct,
                     num_episodes, meta_batch_size, inner_gradient_steps, alpha, beta, device)


