from collections import defaultdict
from pytorch_lightning.callbacks import Callback

from neural_processes.utils.visualisation import plot_test
# c.f. running loss monitoring in torch tutorial:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#  'For running averages you have to implement the logic in training step'
# https://github.com/PyTorchLightning/pytorch-lightning/issues/488
# command show_progress_bar=False in Trainer

class BaseMetricAccumulator(Callback):
    """Callback that accumulates epoch averages of metrics.
      (like keras base logger)

      N.B. batch_idx is index of batch within epoch; total_batch_idx number of batches overall
           global_step is total number of gradient updates -
           if (self.batch_idx + 1) % self.accumulate_grad_batches == 0:
                self.global_step += 1

          loggers use trainer.batch_idx when deciding whether to log or not
          we could use trainer.global_step instead

      Problem:
      self.callback_metrics.update seems to happen after on_batch_ned?
    """

    def on_epoch_start(self, *args, **kwargs):
        # reset running totals
        self.seen = 0
        self.totals = defaultdict(int)

    def on_batch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        batch_size = metrics.get('batch_size', 1)
        self.seen += batch_size

        for k, v in metrics.items():
          self.totals[k] += v * batch_size

    # This (below) could be used to update logger with epoch level training set running average metrics 
    # def on_epoch_end(self, trainer, pl_module):
    #   problem is that this is going to conflict with batch level logs
    #   metrics = {k: v / self.seen for k, v in self.totals.items()}
    #   trainer.log_metrics(metrics)

class RunningMetricPrinter(BaseMetricAccumulator):
    """
    Print loss to stdout every log_freq steps
    """

    def __init__(self, log_freq, *args, attr_monitor='batch_idx', **kwargs):
        self.log_freq = log_freq
        self.attr_monitor = attr_monitor
        super().__init__(*args, **kwargs)

    def on_batch_end(self, trainer, pl_module):
        super().on_batch_end(trainer, pl_module) # this updates self.seen and self.totals
        counter = getattr(trainer, self.attr_monitor) + 1
        should_log = counter % self.log_freq == 0 # +1 bc total_batch_idx, global_step are updated after run_training_batch (which includes on_batch_end)

        if should_log:
            msg = f'Running average metrics after {counter} steps ({self.attr_monitor})'
            msg += '\t'.join([f'{k}: {v/self.seen:.3f}' for k, v in self.totals.items()])
            print(msg, flush=True)

class RunningMetricCSVLogger(BaseMetricAccumulator):
    """
    Save running average training metrics to CSV [not really much point if this isnt integrated with val loss?]
    """
    pass

class CurvePlotter(Callback):

    def __init__(self, plot_freq, *args, random_kernel_parameters=False, with_samples=True, **kwargs):
        self.plot_freq = plot_freq
        self.with_samples = with_samples
        self.random_kernel_parameters = random_kernel_parameters

    def on_batch_end(self, trainer, pl_module):
        make_plot = (trainer.total_batch_idx % self.plot_freq == 0) and trainer.total_batch_idx > 0
        if make_plot:
            plot_test(pl_module, self.random_kernel_parameters, with_samples=self.with_samples)
