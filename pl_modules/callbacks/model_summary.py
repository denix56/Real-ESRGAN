import contextlib
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary as PL_ModelSummary


class _ModelSummary(PL_ModelSummary):
    def _forward_example_input(self) -> None:
        """Run the example input through each layer to get input- and output sizes."""
        model = self._model
        trainer = self._model.trainer

        input_ = model.example_input_array
        input_ = model._apply_batch_transfer_handler(input_)

        mode = model.training
        model.eval()

        forward_context = contextlib.nullcontext() if trainer is None else trainer.precision_plugin.forward_context()
        with torch.no_grad(), forward_context:
            # let the model hooks collect the input- and output shapes
            model(input_)
        model.train(mode)  # restore mode of module


def summarize(lightning_module: "pl.LightningModule", max_depth: int = 1) -> _ModelSummary:
    """Summarize the LightningModule specified by `lightning_module`.

    Args:
        lightning_module: `LightningModule` to summarize.

        max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the
            layer summary off. Default: 1.

    Return:
        The model summary object
    """
    return _ModelSummary(lightning_module, max_depth=max_depth)


class ModelSummary(pl.callbacks.ModelSummary):
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._max_depth:
            return None

        model_summary = summarize(pl_module, max_depth=self._max_depth)
        summary_data = model_summary._get_summary_data()
        total_parameters = model_summary.total_parameters
        trainable_parameters = model_summary.trainable_parameters
        model_size = model_summary.model_size

        if trainer.is_global_zero:
            self.summarize(summary_data, total_parameters, trainable_parameters, model_size)
