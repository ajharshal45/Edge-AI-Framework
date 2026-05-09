"""
EdgeFlow Framework
==================
Train on your laptop. Deploy to any IoT device. Run forever offline.

EdgeFlow is an open-source Edge AI framework that enables any IoT developer
to add real Machine Learning inference to ESP32, Raspberry Pi, or any Linux
device — with zero cloud dependency, zero internet requirement, and zero
complex configuration.

Quick Start:
    from edgeflow import EdgeFlow

    ef = EdgeFlow(domain="safenest", target="esp32", model="decision_tree")
    report = ef.train("safenest_data.csv")
    result = ef.validate_target()
    ef.export(output="safenest_model", format="both")
    bench  = ef.benchmark(data="safenest_data.csv", n_runs=100)
"""

from typing import Optional

from edgeflow.core.trainer import EdgeFlowTrainer, TrainingReport
from edgeflow.core.validator import DeviceValidator, ValidationResult
from edgeflow.core.exporter import EdgeFlowExporter
from edgeflow.core.benchmarker import EdgeFlowBenchmarker, BenchmarkReport
from edgeflow.datasets.formatter import EdgeFlowFormatter

__version__ = "1.0.0"
__author__  = "EdgeFlow — VIT Pune EDI Group E-16"
__license__ = "MIT"


class EdgeFlow:
    """
    Main entry point for the EdgeFlow framework.

    Wraps the full pipeline — training, validation, export, and benchmarking —
    behind a single, clean object API.

    Parameters
    ----------
    domain : str
        User-defined name for this use case, e.g. "safenest", "environment".
        Used in export file naming and report headers.
    target : str
        Target deployment device. One of: "esp32", "raspberry_pi", "linux".
        Default: "esp32"
    model : str
        ML model type. One of: "logistic", "decision_tree", "naive_bayes",
        "random_forest". Default: "decision_tree"
    verbose : bool
        If True, prints progress messages during training/export/benchmark.
        Default: True

    Example
    -------
    >>> from edgeflow import EdgeFlow
    >>> ef = EdgeFlow(domain="safenest", target="esp32", model="decision_tree")
    >>> report = ef.train("safenest_data.csv")
    >>> ef.export(output="safenest_model", format="both")
    """

    def __init__(
        self,
        domain: str = "custom",
        target: str = "esp32",
        model: str = "decision_tree",
        verbose: bool = True,
    ):
        self.domain  = domain
        self.target  = target
        self.model   = model
        self.verbose = verbose

        # Set after train() is called
        self._trainer:      Optional[EdgeFlowTrainer]     = None
        self._report:       Optional[TrainingReport]      = None
        self._fitted_model                                = None
        self._formatter:    Optional[EdgeFlowFormatter]   = None
        self._is_trained:   bool                          = False

    # ------------------------------------------------------------------
    # train()
    # ------------------------------------------------------------------

    def train(
        self,
        data: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> TrainingReport:
        """
        Load data, train the selected model, and store results internally.

        Parameters
        ----------
        data : str
            Path to training CSV file (last column = label).
        test_size : float
            Fraction of data reserved for testing. Default 0.2.
        random_state : int
            Seed for reproducibility. Default 42.

        Returns
        -------
        TrainingReport with accuracy, model size, label map, etc.
        """
        self._trainer = EdgeFlowTrainer(
            model=self.model,
            target=self.target,
            domain=self.domain,
            verbose=self.verbose,
        )

        self._report = self._trainer.train(
            data=data,
            test_size=test_size,
            random_state=random_state,
        )

        self._fitted_model = self._trainer.get_fitted_model()
        self._formatter    = self._trainer.get_formatter()
        self._is_trained   = True

        return self._report

    # ------------------------------------------------------------------
    # validate_target()
    # ------------------------------------------------------------------

    def validate_target(self) -> ValidationResult:
        """
        Check whether the trained model is compatible with the target device.

        Prints the formatted validator report to stdout.

        Returns
        -------
        ValidationResult with compatible flag, RAM/Flash estimates, warnings.

        Notes
        -----
        Can be called before or after train() — it uses self.model and
        self.target directly from the constructor, not the fitted model.
        """
        validator = DeviceValidator(verbose=self.verbose)
        return validator.validate(
            model_type=self.model,
            target_device=self.target,
        )

    # ------------------------------------------------------------------
    # export()
    # ------------------------------------------------------------------

    def export(
        self,
        output: str = "edgeflow_model",
        format: str = "header",
    ) -> list:
        """
        Export the trained model to C++ header and/or binary format.

        Parameters
        ----------
        output : str
            Output filename without extension, e.g. "safenest_model".
            Produces "safenest_model.h" and/or "safenest_model.edgeai".
        format : str
            One of "header", "binary", or "both". Default "header".

        Returns
        -------
        List of file paths written.

        Raises
        ------
        RuntimeError if called before train().
        """
        self._require_trained("export()")

        exporter = EdgeFlowExporter(verbose=self.verbose)
        return exporter.export(
            fitted_model=self._fitted_model,
            formatter=self._formatter,
            report=self._report,
            output=output,
            format=format,
        )

    # ------------------------------------------------------------------
    # benchmark()
    # ------------------------------------------------------------------

    def benchmark(
        self,
        data: str,
        n_runs: int = 100,
    ) -> BenchmarkReport:
        """
        Benchmark prediction latency and accuracy over n_runs passes.

        Parameters
        ----------
        data : str
            Path to CSV file to use as test data (same format as training CSV).
        n_runs : int
            Number of full passes over the dataset. Default 100.

        Returns
        -------
        BenchmarkReport with avg/min/max latency, accuracy, model size,
        and data_transferred_bytes (always 0).

        Raises
        ------
        RuntimeError if called before train().
        """
        self._require_trained("benchmark()")

        benchmarker = EdgeFlowBenchmarker(verbose=self.verbose)
        return benchmarker.benchmark(
            fitted_model=self._fitted_model,
            formatter=self._formatter,
            data=data,
            n_runs=n_runs,
            target_device=self.target,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_report(self) -> TrainingReport:
        """Return the TrainingReport from the last train() call."""
        self._require_trained("get_report()")
        return self._report

    def get_fitted_model(self):
        """Return the raw fitted sklearn model."""
        self._require_trained("get_fitted_model()")
        return self._fitted_model

    def get_formatter(self) -> EdgeFlowFormatter:
        """Return the fitted EdgeFlowFormatter."""
        self._require_trained("get_formatter()")
        return self._formatter

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_trained(self, caller: str):
        """Raise a clear RuntimeError if train() has not been called yet."""
        if not self._is_trained:
            raise RuntimeError(
                f"EdgeFlow.{caller} requires a trained model. "
                f"Call ef.train('your_data.csv') first."
            )

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "not trained"
        return (
            f"EdgeFlow(domain='{self.domain}', target='{self.target}', "
            f"model='{self.model}', status={status})"
        )
