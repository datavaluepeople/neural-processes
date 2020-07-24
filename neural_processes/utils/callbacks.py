from collections import defaultdict
from pytorch_lightning.callbacks import Callback

from neural_processes.utils.visualisation import plot_test


class BaseMetricAccumulator(Callback):
    """Callback that accumulates epoch averages of metrics.
       (like keras base logger)

       c.f. running loss monitoring in torch tutorial:
       https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

      N.B. batch_idx is index of batch within epoch (and what loggers use to trigger updates);
           total_batch_idx number of batches overall
           global_step is total number of gradient updates -
    """

    def on_epoch_start(self, *args, **kwargs):
        # reset running totals
        self.seen = 0
        self.totals = defaultdict(int)

    def on_batch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        batch_size = metrics.get("batch_size", 1)
        self.seen += batch_size

        for k, v in metrics.items():
            self.totals[k] += v * batch_size


class RunningMetricPrinter(BaseMetricAccumulator):
    """
    Print loss to stdout every log_freq steps
    """

    def __init__(self, log_freq, *args, attr_monitor="batch_idx", **kwargs):
        self.log_freq = log_freq
        self.attr_monitor = attr_monitor
        super().__init__(*args, **kwargs)

    def on_batch_end(self, trainer, pl_module):
        super().on_batch_end(trainer, pl_module)  # updates self.seen and self.totals
        # +1 bc (total_)batch_idx updated after call to on_batch_end (in run_training_step)
        counter = getattr(trainer, self.attr_monitor) + 1
        should_log = counter % self.log_freq == 0

        if should_log:
            msg = f"Running averages ({counter} steps):  "
            msg += "  -  ".join(
                [f"{k}: {v/self.seen:.3f}" for k, v in self.totals.items()]
            )
            print(msg, flush=True)


class CurvePlotter(Callback):
    def __init__(self, plot_freq, *args,
                 random_kernel_parameters=False, with_samples=True,
                 **kwargs):
        self.plot_freq = plot_freq
        self.with_samples = with_samples
        self.random_kernel_parameters = random_kernel_parameters

    def on_batch_end(self, trainer, pl_module):
        make_plot = trainer.total_batch_idx % self.plot_freq == 0
        if make_plot and trainer.total_batch_idx > 0:
            plot_test(
                pl_module, self.random_kernel_parameters, with_samples=self.with_samples
            )
