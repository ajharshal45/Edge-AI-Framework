"""
EdgeFlow CLI — Command Line Interface
======================================
All EdgeFlow commands available after `pip install edgeflow`.

Commands:
    edgeflow init           Create project scaffold (config, data/, models/)
    edgeflow train          Train a model on a CSV dataset
    edgeflow validate       Check model-device compatibility
    edgeflow export         Export trained model to .h and/or .edgeai
    edgeflow benchmark      Benchmark prediction latency and accuracy
    edgeflow generate-data  Generate synthetic sensor data

Usage examples (Section 8 of blueprint):
    edgeflow init --domain safenest --target esp32
    edgeflow train --data data/safenest_data.csv --model decision_tree --target esp32 --domain safenest
    edgeflow validate --target esp32 --model decision_tree
    edgeflow export --output models/safenest_model --format both
    edgeflow benchmark --data data/safenest_data.csv --runs 100
    edgeflow generate-data --domain safenest --samples 500 --output data/safenest_data.csv
"""

import argparse
import os
import sys

SEP = "=" * 39


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_box(title: str, lines: list):
    """Print a titled ═══ box."""
    print()
    print(f"[{title}]")
    print(SEP)
    for line in lines:
        print(line)
    print(SEP)
    print()


def _check_trained_model_exists() -> bool:
    """Return True if a trained state can be inferred from local model files."""
    return os.path.isfile("edgeflow_config.yaml")


# ---------------------------------------------------------------------------
# Command: init
# ---------------------------------------------------------------------------

def cmd_init(args):
    """
    edgeflow init --domain safenest --target esp32

    Creates:
        edgeflow_config.yaml
        data/
        models/
    Prints quickstart instructions.
    """
    import yaml  # optional import — only needed for init

    domain = args.domain
    target = args.target
    model  = getattr(args, "model", "decision_tree")

    config = {
        "domain": domain,
        "target": target,
        "model":  model,
        "data_dir":   "data",
        "models_dir": "models",
    }

    # Create folders
    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Write config
    config_path = "edgeflow_config.yaml"
    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        config_written = True
    except ImportError:
        # pyyaml not installed — write a plain text fallback
        with open(config_path, "w") as f:
            for k, v in config.items():
                f.write(f"{k}: {v}\n")
        config_written = True

    _print_box("EdgeFlow Init", [
        f"Domain   : {domain}",
        f"Target   : {target}",
        f"Model    : {model}",
        "",
        f"[OK] Created: {config_path}",
        "[OK] Created: data/",
        "[OK] Created: models/",
    ])

    print("Next steps:")
    print(f"  1. Add your CSV data to data/")
    print(f"  2. edgeflow generate-data --domain {domain} --samples 500 --output data/{domain}_data.csv")
    print(f"  3. edgeflow train --data data/{domain}_data.csv --model {model} --target {target} --domain {domain}")
    print(f"  4. edgeflow validate --target {target}")
    print(f"  5. edgeflow export --output models/{domain}_model --format both")
    print()


# ---------------------------------------------------------------------------
# Command: generate-data
# ---------------------------------------------------------------------------

def cmd_generate_data(args):
    """
    edgeflow generate-data --domain safenest --samples 500 --output data/safenest_data.csv
    """
    domain  = args.domain
    samples = args.samples
    output  = args.output

    print()
    print("[EdgeFlow Generate Data]")
    print(SEP)
    print(f"Domain  : {domain}")
    print(f"Samples : {samples}")
    print(f"Output  : {output}")
    print(SEP)

    if domain == "safenest":
        from edgeflow.datasets.builtin.generate_safenest import generate_safenest_dataset
        os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
        df = generate_safenest_dataset(n_samples=samples, output_path=output)
        print(f"[OK] Generated {len(df)} samples -> {output}")
        print(f"   SAFE: {(df['label'] == 'SAFE').sum()}  |  DANGER: {(df['label'] == 'DANGER').sum()}")
        print(f"   Preview:")
        print(df.head(5).to_string(index=False))
    else:
        print(f"[WARNING] No built-in generator for domain '{domain}'.")
        print(f"   Supported domains: safenest")
        print(f"   Provide your own CSV with format: feature_1, ..., feature_n, label")

    print()


# ---------------------------------------------------------------------------
# Command: train
# ---------------------------------------------------------------------------

def cmd_train(args):
    """
    edgeflow train --data data/safenest_data.csv --model decision_tree --target esp32 --domain safenest
    """
    from edgeflow import EdgeFlow

    ef = EdgeFlow(
        domain=args.domain,
        target=args.target,
        model=args.model,
        verbose=False,          # we print our own formatted output
    )

    print()
    print("[EdgeFlow Trainer]")
    print(SEP)
    print(f"Loading data...     ", end="", flush=True)

    try:
        report = ef.train(
            data=args.data,
            test_size=getattr(args, "test_size", 0.2),
            random_state=getattr(args, "random_state", 42),
        )
    except Exception as e:
        print(f"[FAILED]")
        print(SEP)
        print(f"Error: {e}")
        print()
        sys.exit(1)

    print(f"[OK] {report.n_samples} samples loaded")
    print(f"Validating...       [OK] Format correct")
    print(f"Training...         [OK] {report.model_type.replace('_', ' ').title()} fitted")
    print(f"Accuracy...         [OK] {report.accuracy * 100:.1f}%")
    print(f"Model size...       [OK] {report.model_size_kb} KB")
    print(SEP)
    print(f"Training complete. Run 'edgeflow export' next.")
    print()

    # Persist trained model to disk for use by export/benchmark commands
    _save_session(ef, args)


# ---------------------------------------------------------------------------
# Command: validate
# ---------------------------------------------------------------------------

def cmd_validate(args):
    """
    edgeflow validate --target esp32 [--model decision_tree]
    """
    from edgeflow.core.validator import DeviceValidator

    model  = getattr(args, "model", "decision_tree")
    target = args.target

    validator = DeviceValidator(verbose=True)
    validator.validate(model_type=model, target_device=target)


# ---------------------------------------------------------------------------
# Command: export
# ---------------------------------------------------------------------------

def cmd_export(args):
    """
    edgeflow export --output models/safenest_model --format both

    Requires a previously trained model saved in the session.
    """
    ef = _load_session_model(args)
    if ef is None:
        return

    print()
    print("[EdgeFlow Exporter]")
    print(SEP)

    try:
        paths = ef.export(output=args.output, format=args.format)
    except Exception as e:
        print(f"[FAILED] Export failed: {e}")
        print(SEP)
        sys.exit(1)

    for p in paths:
        label = "header" if p.endswith(".h") else "binary"
        print(f"Exporting {label}...  [OK] {p}")

    print(SEP)

    h_file = next((p for p in paths if p.endswith(".h")), None)
    if h_file:
        model_name = os.path.basename(h_file)
        print(f"Copy {model_name} to your Arduino project.")
        print(f"Include EdgeFlow.h and {model_name} in your sketch.")

    print()


# ---------------------------------------------------------------------------
# Command: benchmark
# ---------------------------------------------------------------------------

def cmd_benchmark(args):
    """
    edgeflow benchmark --data data/safenest_data.csv --runs 100
    """
    ef = _load_session_model(args)
    if ef is None:
        return

    try:
        ef.benchmark(data=args.data, n_runs=args.runs)
    except Exception as e:
        print(f"[FAILED] Benchmark failed: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Session helpers (persist trained model within a shell session via pickle)
# ---------------------------------------------------------------------------

_SESSION_FILE = ".edgeflow_session.pkl"


def _save_session(ef, args):
    """Pickle the trained EdgeFlow instance so export/benchmark can reuse it."""
    import pickle
    try:
        with open(_SESSION_FILE, "wb") as f:
            pickle.dump(ef, f)
    except Exception:
        pass  # silently skip — export can always retrain


def _load_session_model(args):
    """
    Load session EdgeFlow instance if available, else prompt user to train first.
    Falls back to retraining if --data flag is present.
    """
    import pickle
    from edgeflow import EdgeFlow

    # Try loading pickled session
    if os.path.isfile(_SESSION_FILE):
        try:
            with open(_SESSION_FILE, "rb") as f:
                ef = pickle.load(f)
            return ef
        except Exception:
            pass

    # Fallback: retrain if --data provided
    data = getattr(args, "data", None)
    if data and os.path.isfile(data):
        domain = getattr(args, "domain", "custom")
        target = getattr(args, "target", "esp32")
        model  = getattr(args, "model",  "decision_tree")
        ef = EdgeFlow(domain=domain, target=target, model=model, verbose=False)
        ef.train(data=data)
        return ef

    print()
    print("[ERROR] No trained model found.")
    print("   Run 'edgeflow train --data your_data.csv' first.")
    print()
    return None


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edgeflow",
        description=(
            "EdgeFlow — Train on your laptop. Deploy to any IoT device. Run forever offline.\n"
            "Edge AI framework for ESP32, Raspberry Pi, and Linux."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version="EdgeFlow 1.0.0"
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # ---- init ----
    p_init = subparsers.add_parser(
        "init",
        help="Create project scaffold (config, data/, models/)",
    )
    p_init.add_argument("--domain", default="custom",
                        help="Project domain name (e.g. safenest)")
    p_init.add_argument("--target", default="esp32",
                        choices=["esp32", "raspberry_pi", "linux"],
                        help="Target deployment device")
    p_init.add_argument("--model",  default="decision_tree",
                        choices=["logistic", "decision_tree", "naive_bayes", "random_forest"],
                        help="Default model type")
    p_init.set_defaults(func=cmd_init)

    # ---- generate-data ----
    p_gen = subparsers.add_parser(
        "generate-data",
        help="Generate synthetic sensor data for a built-in domain",
    )
    p_gen.add_argument("--domain",  default="safenest",
                       help="Domain name (currently: safenest)")
    p_gen.add_argument("--samples", type=int, default=500,
                       help="Number of samples to generate (default: 500)")
    p_gen.add_argument("--output",  default="data/safenest_data.csv",
                       help="Output CSV path")
    p_gen.set_defaults(func=cmd_generate_data)

    # ---- train ----
    p_train = subparsers.add_parser(
        "train",
        help="Train a model on a CSV dataset",
    )
    p_train.add_argument("--data",    required=True,
                         help="Path to training CSV file")
    p_train.add_argument("--model",   default="decision_tree",
                         choices=["logistic", "decision_tree", "naive_bayes", "random_forest"],
                         help="Model type (default: decision_tree)")
    p_train.add_argument("--target",  default="esp32",
                         choices=["esp32", "raspberry_pi", "linux"],
                         help="Target device (default: esp32)")
    p_train.add_argument("--domain",  default="custom",
                         help="Domain name for this project")
    p_train.add_argument("--test-size", dest="test_size", type=float, default=0.2,
                         help="Test split fraction (default: 0.2)")
    p_train.add_argument("--random-state", dest="random_state", type=int, default=42,
                         help="Random seed (default: 42)")
    p_train.set_defaults(func=cmd_train)

    # ---- validate ----
    p_val = subparsers.add_parser(
        "validate",
        help="Check model-device compatibility",
    )
    p_val.add_argument("--target", default="esp32",
                       choices=["esp32", "raspberry_pi", "linux"],
                       help="Target device")
    p_val.add_argument("--model",  default="decision_tree",
                       choices=["logistic", "decision_tree", "naive_bayes", "random_forest"],
                       help="Model type to validate")
    p_val.set_defaults(func=cmd_validate)

    # ---- export ----
    p_exp = subparsers.add_parser(
        "export",
        help="Export trained model to C++ header and/or binary",
    )
    p_exp.add_argument("--output", default="edgeflow_model",
                       help="Output filename without extension")
    p_exp.add_argument("--format", default="header",
                       choices=["header", "binary", "both"],
                       help="Export format (default: header)")
    p_exp.add_argument("--data",   default=None,
                       help="(optional) Training CSV path — re-trains if no session found")
    p_exp.add_argument("--domain", default="custom")
    p_exp.add_argument("--target", default="esp32",
                       choices=["esp32", "raspberry_pi", "linux"])
    p_exp.add_argument("--model",  default="decision_tree",
                       choices=["logistic", "decision_tree", "naive_bayes", "random_forest"])
    p_exp.set_defaults(func=cmd_export)

    # ---- benchmark ----
    p_bench = subparsers.add_parser(
        "benchmark",
        help="Benchmark prediction latency and accuracy",
    )
    p_bench.add_argument("--data",   required=True,
                         help="Path to test CSV file")
    p_bench.add_argument("--runs",   type=int, default=100,
                         help="Number of benchmark runs (default: 100)")
    p_bench.add_argument("--domain", default="custom")
    p_bench.add_argument("--target", default="esp32",
                         choices=["esp32", "raspberry_pi", "linux"])
    p_bench.add_argument("--model",  default="decision_tree",
                         choices=["logistic", "decision_tree", "naive_bayes", "random_forest"])
    p_bench.set_defaults(func=cmd_benchmark)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point registered in setup.py as 'edgeflow'."""
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
