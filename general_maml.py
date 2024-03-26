def sample_task():
    pass


def initialize_theta():
    pass


def compute_adapted_theta(theta, task, k, ):
    # theta might need to be cloned
    pass


def get_loss(theta_i, task, k):
    pass


def forward(x, params):
    pass


# Hyperparameters
num_episodes = ...
meta_batch_size = ...  # number of tasks sampled each episode
k = ...
# size of D_i. number of data points to train on per task. Will be batch processed (?)
# the paper does not mention whether the size of D_i' has to be the same, but anything else wouldnt make sense imo
inner_gradient_steps = ...  # gradient steps done in inner loop during training
# NOTE not entirely sure how multiple gradient steps are handled
alpha, beta = ...


# randomly_initialize paramaters
theta = initialize_theta()
for _ in range(num_episodes):
    acc_meta_update = 0
    for i in range(meta_batch_size):
        task = sample_task()
        theta_i = compute_adapted_theta(
            theta, task, k, alpha, inner_gradient_steps)
        # this function has to sample k datapoints, batch process them with the given theta,
        # calculate the loss, calculate a gradient of the params wrt this loss and
        # update and return this theta. In addition, it has to be possible to backpropagate through this theta
        test_loss = get_loss(theta_i, task, k)
        acc_meta_update -= grad(test_loss, theta)
    theta -= beta * acc_meta_update
