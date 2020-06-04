"""A PyTorch translation of Deepmind's gaussian process curve generator
see https://github.com/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb
"""
import collections
import torch

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription", ("query", "target_y", "num_total_points", "num_context_points")
)


class CurveGenerator:
    def __init__(
        self,
        batch_size,
        max_num_context,
        x_size=1,
        y_size=1,
        l1_scale=0.6,
        sigma_scale=1.0,
        random_kernel_parameters=True,
        testing=False,
    ):
        """Creates a regression dataset of functions sampled from a GP.

    Args:
      batch_size: An integer.
      max_num_context: The max number of observations in the context.
      x_size: Integer >= 1 for length of "x values" vector.
      y_size: Integer >= 1 for length of "y values" vector.
      l1_scale: Float; typical scale for kernel distance function.
      sigma_scale: Float; typical scale for variance.
      random_kernel_parameters: If `True`, the kernel parameters (l1 and sigma)
          will be sampled uniformly within [0.1, l1_scale] and [0.1, sigma_scale].
      testing: Boolean that indicates whether we are testing. If so there are
          more targets for visualization.
    """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing

    def gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data.

    Args:
      xdata: Tensor of shape [B, num_total_points, x_size] with
          the values of the x-axis data.
      l1: Tensor of shape [B, y_size, x_size], the scale
          parameter of the Gaussian kernel.
      sigma_f: Tensor of shape [B, y_size], the magnitude
          of the std.
      sigma_noise: Float, std of the noise that we add for stability.

    Returns:
      The kernel, a float tensor of shape
      [B, y_size, num_total_points, num_total_points].
    """
        num_total_points = xdata.shape[1]

        # xdata is [B, num_total_points, x_size]
        # Force broadcasting to get the diff of every pair of points in num_total_points
        # aka diff is [B, num_total_points, num_total_points, x_size]
        diff = xdata.unsqueeze(1) - xdata.unsqueeze(2)

        # [B, y_size, num_total_points, num_total_points, x_size]
        # [:, None, ..] notation is another way to unsqueeze
        norm = torch.square(
            diff[:, None, :, :, :] / l1[:, :, None, None, :]
        )  # could use x ** 2 notation
        norm = norm.sum(dim=-1)  # [B, data_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = torch.square(sigma_f)[:, :, None, None] * torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise ** 2) * torch.eye(num_total_points)

        return kernel

    def get_test_x_values(self):
        """Seelct an even distribution of x-positions for testing (and plotting)."""
        num_target = 400

        step_size = 4 / num_target
        x_values = torch.arange(-2, 2, step_size).repeat([self._batch_size, 1])
        return x_values.unsqueeze(-1)

    def get_train_x_values(self, num_context):
        """Select a random number of x-positions at random for training."""
        num_target = torch.randint(size=(), low=0, high=self._max_num_context + 1 - num_context)
        num_total_points = num_context + num_target
        low = -2
        high = 2
        scale = high - low
        x_values = torch.rand(size=(self._batch_size, num_total_points, self._x_size))
        return x_values * scale + low

    def get_kernel_hyperparams(self):
        # Either choose a set of random parameters for the mini-batch
        if self._random_kernel_parameters:
            l1 = torch.rand(size=(self._batch_size, self._y_size, self._x_size))
            l1 = l1 * self._l1_scale + 0.1

            sigma_f = torch.rand(size=(self._batch_size, self._y_size))
            sigma_f * self._sigma_scale + 0.1
        # Or use the same fixed parameters for all mini-batches
        else:
            l1 = torch.ones(size=[self._batch_size, self._y_size, self._x_size]) * self._l1_scale
            sigma_f = torch.ones(size=[self._batch_size, self._y_size]) * self._sigma_scale
        return l1, sigma_f

    def _sample_cholesky(self, kernel):
        """Sample from the GP prior via the Cholesky decomposition."""
        num_total_points = kernel.shape[-1]
        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.cholesky(kernel.double()).float()

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = torch.matmul(
            cholesky, torch.randn([self._batch_size, self._y_size, num_total_points, 1])
        )
        # [batch_size, num_total_points, y_size]
        return y_values.squeeze(dim=3).permute((0, 2, 1))

    def _sample_direct(self, kernel):
        """Sample from the GP prior via torch multivariate Normal distribution."""
        num_total_points = kernel.shape[-1]

        y_values = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros([self._batch_size, 1, num_total_points]), covariance_matrix=kernel,
        ).sample()

        # [batch_size, num_total_points, y_size]
        return y_values.permute((0, 2, 1))

    def generate_curves(self):
        """Builds the op delivering the data.

    Generated functions are `float32` with x values between -2 and 2.

    Returns:
      A `NPRegressionDescription` namedtuple.
    """
        num_context = torch.randint(size=(), low=3, high=self._max_num_context + 1)

        if self._testing:
            x_values = self.get_test_x_values()
        else:
            x_values = self.get_train_x_values(num_context)
        num_total_points = x_values.shape[1]

        l1, sigma_f = self.get_kernel_hyperparams()

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self.gaussian_kernel(x_values, l1, sigma_f)

        y_values = self._sample_cholesky(kernel)

        if self._testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = torch.randperm(num_total_points)
            # Ignore type here (false positive that a slice index must be an int)
            context_x = x_values[:, idx[:num_context], :]  # type: ignore
            context_y = y_values[:, idx[:num_context], :]  # type: ignore

        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x_values
            target_y = y_values

            # Select the observations
            context_x = x_values[:, :num_context, :]  # type: ignore
            context_y = y_values[:, :num_context, :]  # type: ignore

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context,
        )
