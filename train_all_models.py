import argparse
import subprocess


BASELINE_CONFIGS = [
    "configs/baseline_ae.yaml",
    "configs/baseline_devae_fixed.yaml",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    configs = BASELINE_CONFIGS if args.all else BASELINE_CONFIGS[:1]

    for cfg in configs:
        subprocess.run(
            ["python", "run_experiment.py", "--config", cfg],
            check=True,
        )


if __name__ == "__main__":
    main()