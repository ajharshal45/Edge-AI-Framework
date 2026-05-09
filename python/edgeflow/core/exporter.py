"""
EdgeFlow Exporter
=================
Generates C++ header (.h) and binary (.edgeai) model files from a trained
sklearn model, ready to be included in an Arduino/ESP32 project.

Follows Section 6 of the EdgeFlow Blueprint exactly.

Usage:
    from edgeflow.core.exporter import EdgeFlowExporter

    exporter = EdgeFlowExporter()
    exporter.export(
        fitted_model=trainer.get_fitted_model(),
        formatter=trainer.get_formatter(),
        report=report,
        output="safenest_model",
        format="both",
    )
    # Produces: safenest_model.h  and  safenest_model.edgeai
"""

import json
import struct
import os
from datetime import datetime, timezone
from typing import List

import numpy as np

from edgeflow.datasets.formatter import EdgeFlowFormatter
from edgeflow.core.trainer import TrainingReport


# Model type codes (must match C++ runtime constants)
MODEL_TYPE_CODES = {
    "logistic":      0,
    "decision_tree": 1,
    "naive_bayes":   2,
    "random_forest": 3,
}

VALID_FORMATS = ("header", "binary", "both")


class EdgeFlowExporter:
    """
    Exports a trained EdgeFlow model to C++ header and/or binary format.

    The .h  file is included directly in an Arduino sketch.
    The .edgeai file is loaded at runtime from SPIFFS/SD card on ESP32.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        fitted_model,
        formatter: EdgeFlowFormatter,
        report: TrainingReport,
        output: str = "edgeflow_model",
        format: str = "header",
    ) -> List[str]:
        """
        Export a trained model to C++ header and/or binary format.

        Parameters
        ----------
        fitted_model : sklearn estimator (already fitted)
        formatter    : fitted EdgeFlowFormatter
        report       : TrainingReport from trainer.train()
        output       : output filename without extension
        format       : "header", "binary", or "both"

        Returns
        -------
        List of file paths written.
        """
        if format not in VALID_FORMATS:
            raise ValueError(
                f"Invalid format '{format}'. Choose from: {VALID_FORMATS}"
            )

        self._log("Starting export...")
        self._log(f"Model  : {report.model_type}")
        self._log(f"Target : {report.target_device}")
        self._log(f"Format : {format}")

        written = []

        if format in ("header", "both"):
            path = self._export_header(fitted_model, formatter, report, output)
            written.append(path)

        if format in ("binary", "both"):
            path = self._export_binary(fitted_model, formatter, report, output)
            written.append(path)

        self._log("=" * 39)
        for p in written:
            self._log(f"[OK] {p}")
        self._log("=" * 39)

        return written

    # ------------------------------------------------------------------
    # Header (.h) export
    # ------------------------------------------------------------------

    def _export_header(
        self,
        fitted_model,
        formatter: EdgeFlowFormatter,
        report: TrainingReport,
        output: str,
    ) -> str:
        filepath = output + ".h"
        content = self._build_header(fitted_model, formatter, report)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        if self.verbose:
            print(f"[EdgeFlow Exporter] Exported header -> {filepath}")

        return filepath

    def _build_header(
        self,
        fitted_model,
        formatter: EdgeFlowFormatter,
        report: TrainingReport,
    ) -> str:
        """Build the full .h file content as a string."""

        timestamp   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        model_code  = MODEL_TYPE_CODES.get(report.model_type, 0)
        scaler      = formatter.get_scaler_params()
        label_map   = formatter.get_label_map()
        feature_names = report.feature_names
        n_features  = report.n_features
        n_classes   = report.n_classes
        accuracy_pct = report.accuracy * 100

        guard_name = f"EDGEFLOW_{report.domain.upper()}_{report.model_type.upper()}_H"

        lines = []

        # ---- Top comment block ----
        lines += [
            "// ============================================",
            "// EdgeFlow Framework v1.0",
            "// Auto-generated model header",
            "// DO NOT EDIT MANUALLY",
            "// ============================================",
            f"// Domain      : {report.domain}",
            f"// Model       : {report.model_type}",
            f"// Target      : {report.target_device}",
            f"// Features    : {n_features} ({', '.join(feature_names)})",
            f"// Classes     : {n_classes} ({', '.join(f'{v}={k}' for k, v in sorted(label_map.items(), key=lambda x: x[1]))})",
            f"// Accuracy    : {accuracy_pct:.2f}%",
            f"// Generated   : {timestamp}",
            "// ============================================",
            "",
            f"#ifndef {guard_name}",
            f"#define {guard_name}",
            "",
            f"#define EDGEFLOW_N_FEATURES {n_features}",
            f"#define EDGEFLOW_N_CLASSES  {n_classes}",
            f"#define EDGEFLOW_MODEL_TYPE {model_code}   "
            f"// 0=logistic, 1=decision_tree, 2=naive_bayes",
            "",
        ]

        # ---- Feature names (comments) ----
        lines.append("// Feature names (for reference)")
        for i, name in enumerate(feature_names):
            lines.append(f"// Feature {i}: {name}")
        lines.append("")

        # ---- Label map (comments) ----
        lines.append("// Label map")
        for label, idx in sorted(label_map.items(), key=lambda x: x[1]):
            lines.append(f"// {idx} = {label}")
        lines.append("")

        # ---- Scaler parameters ----
        mean_str  = ", ".join(f"{v:.6f}" for v in scaler["mean"])
        scale_str = ", ".join(f"{v:.6f}" for v in scaler["scale"])
        lines += [
            "// StandardScaler parameters (stored in Flash)",
            f"const float EDGEFLOW_SCALER_MEAN[]  PROGMEM = {{{mean_str}}};",
            f"const float EDGEFLOW_SCALER_SCALE[] PROGMEM = {{{scale_str}}};",
            "",
        ]

        # ---- Model-specific weights/nodes ----
        if report.model_type == "decision_tree":
            lines += self._header_decision_tree(fitted_model, n_features)
        elif report.model_type == "logistic":
            lines += self._header_logistic(fitted_model, n_classes)
        elif report.model_type == "naive_bayes":
            lines += self._header_naive_bayes(fitted_model, n_classes, n_features)
        elif report.model_type == "random_forest":
            lines += self._header_random_forest(fitted_model, n_features)
        else:
            lines.append(f"// Model type '{report.model_type}' export not implemented")

        lines += [
            "",
            f"#endif // {guard_name}",
            "",
        ]

        return "\n".join(lines)

    # ---- Decision Tree header block ----
    def _header_decision_tree(self, model, n_features: int) -> List[str]:
        """
        Export DT as flat node array.
        Format: {feature_index, threshold, left_child, right_child, leaf_label}
        Leaf nodes: feature_index=-1, threshold=-1, left=-1, right=-1, leaf_label=class_idx
        Internal nodes: leaf_label=-1
        """
        tree = model.tree_

        lines = [
            "// Decision Tree nodes (stored in Flash)",
            "// Format: {feature_index, threshold, left_child, right_child, leaf_label}",
            "// leaf_label = -1 means internal node -- use children to traverse",
            "// For leaf nodes: feature_index = threshold = left = right = -1",
        ]

        n_nodes = tree.node_count
        lines.append(f"const int EDGEFLOW_TREE_NODES = {n_nodes};")
        lines.append("const float EDGEFLOW_TREE[][5] PROGMEM = {")

        for i in range(n_nodes):
            feature   = tree.feature[i]
            threshold = tree.threshold[i]
            left      = tree.children_left[i]
            right     = tree.children_right[i]

            is_leaf = (left == -1)
            if is_leaf:
                leaf_label = int(np.argmax(tree.value[i]))
                row = f"    {{-1, -1.000000, -1, -1, {leaf_label}}}"
                comment = f"// node {i}: leaf -> class {leaf_label}"
            else:
                row = f"    {{{feature}, {threshold:.6f}, {left}, {right}, -1}}"
                feat_name = f"feature[{feature}]"
                comment = (
                    f"// node {i}: if {feat_name} <= {threshold:.4f} "
                    f"-> left({left}) else right({right})"
                )

            sep = "," if i < n_nodes - 1 else " "
            lines.append(f"{row}{sep}   {comment}")

        lines.append("};")
        return lines

    # ---- Logistic Regression header block ----
    def _header_logistic(self, model, n_classes: int) -> List[str]:
        """
        Export LR weights matrix and bias vector.
        Binary: single weight vector + scalar bias.
        Multiclass: one weight row per class + bias per class.
        """
        coef = model.coef_     # shape (n_classes_in_model, n_features)
        bias = model.intercept_  # shape (n_classes_in_model,)

        lines = [
            "// Logistic Regression weights (stored in Flash)",
            "// Inference: dot(scaled_input, weights[class]) + bias[class]",
            "// Apply sigmoid (binary) or softmax (multiclass), return argmax",
        ]

        n_rows = coef.shape[0]

        # Weights matrix
        lines.append(f"const int EDGEFLOW_LR_N_COEF_ROWS = {n_rows};")
        lines.append("const float EDGEFLOW_LR_WEIGHTS[][EDGEFLOW_N_FEATURES] PROGMEM = {")
        for r in range(n_rows):
            row_vals = ", ".join(f"{v:.8f}" for v in coef[r])
            sep = "," if r < n_rows - 1 else " "
            lines.append(f"    {{{row_vals}}}{sep}")
        lines.append("};")
        lines.append("")

        # Bias vector
        bias_vals = ", ".join(f"{v:.8f}" for v in bias)
        lines.append(f"const float EDGEFLOW_LR_BIAS[] PROGMEM = {{{bias_vals}}};")

        return lines

    # ---- Naive Bayes header block ----
    def _header_naive_bayes(self, model, n_classes: int, n_features: int) -> List[str]:
        """
        Export GaussianNB class priors + per-feature mean and variance per class.
        Inference: log P(class) + sum(log P(feature|class)) -> argmax
        """
        priors    = model.class_prior_     # shape (n_classes,)
        means     = model.theta_           # shape (n_classes, n_features)
        variances = model.var_             # shape (n_classes, n_features)

        lines = [
            "// Naive Bayes parameters (stored in Flash)",
            "// Inference: log_prior[c] + sum over features of gaussian_log_likelihood",
        ]

        # Class priors
        prior_vals = ", ".join(f"{v:.8f}" for v in priors)
        lines.append(f"const float EDGEFLOW_NB_PRIORS[] PROGMEM = {{{prior_vals}}};")
        lines.append("")

        # Feature means per class
        lines.append("// Feature means[class][feature]")
        lines.append("const float EDGEFLOW_NB_MEANS[][EDGEFLOW_N_FEATURES] PROGMEM = {")
        for c in range(n_classes):
            row_vals = ", ".join(f"{v:.8f}" for v in means[c])
            sep = "," if c < n_classes - 1 else " "
            lines.append(f"    {{{row_vals}}}{sep}")
        lines.append("};")
        lines.append("")

        # Feature variances per class
        lines.append("// Feature variances[class][feature]")
        lines.append("const float EDGEFLOW_NB_VARS[][EDGEFLOW_N_FEATURES] PROGMEM = {")
        for c in range(n_classes):
            row_vals = ", ".join(f"{v:.8f}" for v in variances[c])
            sep = "," if c < n_classes - 1 else " "
            lines.append(f"    {{{row_vals}}}{sep}")
        lines.append("};")

        return lines

    # ---- Random Forest header block ----
    def _header_random_forest(self, model, n_features: int) -> List[str]:
        """
        Export each tree in the forest as a separate flat node array.
        Uses majority vote across all trees for inference.
        """
        lines = [
            "// Random Forest -- individual tree node arrays (stored in Flash)",
            "// Inference: run each tree, majority vote across all predictions",
        ]

        n_estimators = len(model.estimators_)
        lines.append(f"const int EDGEFLOW_RF_N_TREES = {n_estimators};")
        lines.append("")

        for t_idx, estimator in enumerate(model.estimators_):
            tree = estimator.tree_
            n_nodes = tree.node_count
            lines.append(f"// Tree {t_idx} - {n_nodes} nodes")
            lines.append(f"const int EDGEFLOW_RF_TREE{t_idx}_NODES = {n_nodes};")
            lines.append(
                f"const float EDGEFLOW_RF_TREE{t_idx}[][5] PROGMEM = {{"
            )
            for i in range(n_nodes):
                feature   = tree.feature[i]
                threshold = tree.threshold[i]
                left      = tree.children_left[i]
                right     = tree.children_right[i]
                is_leaf   = (left == -1)
                if is_leaf:
                    leaf_label = int(np.argmax(tree.value[i]))
                    row = f"    {{-1, -1.000000, -1, -1, {leaf_label}}}"
                else:
                    row = f"    {{{feature}, {threshold:.6f}, {left}, {right}, -1}}"
                sep = "," if i < n_nodes - 1 else " "
                lines.append(f"{row}{sep}")
            lines.append("};")
            lines.append("")

        return lines

    # ------------------------------------------------------------------
    # Binary (.edgeai) export
    # ------------------------------------------------------------------

    def _export_binary(
        self,
        fitted_model,
        formatter: EdgeFlowFormatter,
        report: TrainingReport,
        output: str,
    ) -> str:
        filepath = output + ".edgeai"
        data = self._build_binary(fitted_model, formatter, report)

        with open(filepath, "wb") as f:
            f.write(data)

        if self.verbose:
            print(f"[EdgeFlow Exporter] Exported binary -> {filepath}")

        return filepath

    def _build_binary(
        self,
        fitted_model,
        formatter: EdgeFlowFormatter,
        report: TrainingReport,
    ) -> bytes:
        """
        Build the .edgeai binary following blueprint Section 6.2:

        [0:4]   Magic: "EDGF"
        [4:8]   Version: uint32 (100 = v1.0.0)
        [8:12]  Model type: uint32
        [12:16] N features: uint32
        [16:20] N classes: uint32
        [20:24] N scaler params: uint32
        [24:X]  Scaler means: float32 array
        [X:Y]   Scaler scales: float32 array
        [Y:Z]   Model weights/nodes: float32 array
        [Z:]    Metadata JSON: UTF-8 string
        """
        scaler      = formatter.get_scaler_params()
        label_map   = formatter.get_label_map()
        model_code  = MODEL_TYPE_CODES.get(report.model_type, 0)
        n_features  = report.n_features
        n_classes   = report.n_classes
        n_scaler    = n_features  # one mean + one scale per feature (packed separately)

        buf = bytearray()

        # Header fields
        buf += b"EDGF"                                  # [0:4]  magic
        buf += struct.pack("<I", 100)                   # [4:8]  version 1.0.0
        buf += struct.pack("<I", model_code)            # [8:12] model type
        buf += struct.pack("<I", n_features)            # [12:16]
        buf += struct.pack("<I", n_classes)             # [16:20]
        buf += struct.pack("<I", n_scaler)              # [20:24]

        # Scaler means  [24:X]
        for v in scaler["mean"]:
            buf += struct.pack("<f", float(v))

        # Scaler scales [X:Y]
        for v in scaler["scale"]:
            buf += struct.pack("<f", float(v))

        # Model weights/nodes [Y:Z]
        weights_floats = self._extract_weights(fitted_model, report.model_type)
        for v in weights_floats:
            buf += struct.pack("<f", float(v))

        # Metadata JSON [Z:]
        metadata = {
            "domain":        report.domain,
            "model_type":    report.model_type,
            "target_device": report.target_device,
            "feature_names": report.feature_names,
            "label_map":     label_map,
            "accuracy":      report.accuracy,
            "n_features":    n_features,
            "n_classes":     n_classes,
            "generated":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        buf += json.dumps(metadata, separators=(",", ":")).encode("utf-8")

        return bytes(buf)

    def _extract_weights(self, model, model_type: str) -> List[float]:
        """
        Flatten model parameters into a float32 list for binary export.
        The C++ runtime knows the layout from the model_type field.
        """
        if model_type == "logistic":
            # coef_ flattened row-major, then intercept_
            weights = model.coef_.flatten().tolist()
            weights += model.intercept_.tolist()
            return weights

        elif model_type == "decision_tree":
            # Each node: [feature, threshold, left, right, leaf_label]
            tree = model.tree_
            rows = []
            for i in range(tree.node_count):
                left  = tree.children_left[i]
                right = tree.children_right[i]
                if left == -1:  # leaf
                    leaf = float(np.argmax(tree.value[i]))
                    rows += [-1.0, -1.0, -1.0, -1.0, leaf]
                else:
                    rows += [
                        float(tree.feature[i]),
                        float(tree.threshold[i]),
                        float(left),
                        float(right),
                        -1.0,
                    ]
            return rows

        elif model_type == "naive_bayes":
            # priors, then means[class][feature], then vars[class][feature]
            weights = model.class_prior_.tolist()
            weights += model.theta_.flatten().tolist()
            weights += model.var_.flatten().tolist()
            return weights

        elif model_type == "random_forest":
            # All trees concatenated; each tree prefixed with node count
            weights = [float(len(model.estimators_))]
            for estimator in model.estimators_:
                tree = estimator.tree_
                weights.append(float(tree.node_count))
                for i in range(tree.node_count):
                    left  = tree.children_left[i]
                    right = tree.children_right[i]
                    if left == -1:
                        leaf = float(np.argmax(tree.value[i]))
                        weights += [-1.0, -1.0, -1.0, -1.0, leaf]
                    else:
                        weights += [
                            float(tree.feature[i]),
                            float(tree.threshold[i]),
                            float(left),
                            float(right),
                            -1.0,
                        ]
            return weights

        else:
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, message: str):
        if self.verbose:
            print(f"[EdgeFlow Exporter] {message}")
