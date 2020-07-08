import torch
from neural_processes.data import gaussian_process as gp
from matplotlib import pyplot as plt


def plot_test(net, random_kernel_parameters, with_samples=True):
    # N.B. the auto_move_data decorator on the forward method of the models handles transfer of inputs to gpu - just need to transer outputs to cpu to plot (see below)
    test = gp.torch.CurveGenerator(
        batch_size=1,
        max_num_context=20,
        random_kernel_parameters=random_kernel_parameters,
        testing=True
    ).generate_batch()
    
    context_x, context_y, target_x, target_y = test

    num_samples = 100
    
    mean_mu = net.forward(context_x, context_y, target_x, use_mean_latent=True).cpu()
    sample_mus = [
        net.forward(context_x, context_y, target_x, use_mean_latent=False).cpu()
        for _ in range(num_samples)
    ]

    # If we're outputting sigma and mu, just take mu
    if len(mean_mu) == 2:
        mean_mu, _ = mean_mu
        sample_mus = [sample[0] for sample in sample_mus]


    mean_mu = mean_mu.squeeze().detach()
    context_x = context_x.squeeze()
    context_y = context_y.squeeze()
    target_x = target_x.squeeze()
    target_y = target_y.squeeze()

    print('scattering targets', target_x.type(), target_y.type(), context_x.type(), context_y.type(), mean_mu.type(), sample_mus[0].type())
    plt.scatter(target_x, target_y, s=10, color="grey", label="target", alpha=0.5)
    plt.scatter(context_x, context_y, s=20, color="tab:red", label="context")

    plt.plot(target_x, mean_mu, linestyle="--", color="tab:blue", label="output mean")
    if with_samples:
        for mu in sample_mus:
            plt.plot(target_x, mu.squeeze().detach(), linestyle="--", color="tab:blue", alpha=0.1)
    plt.plot(target_x, mean_mu, linestyle="--", color="tab:blue", label="output mean")
    plt.legend()
    plt.show()