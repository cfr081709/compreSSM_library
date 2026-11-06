import subprocess
from pathlib import Path
from compression import compress_module

def sMNIST_experiment():
    results_dir = Path("results/original")
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = {
        "baseline": [
            "python", "scripts/reproduce.py",
            "configs/smnist/default.yaml",
            "--seeds", "42"
        ],
        "compressed": [
            "python", "scripts/reproduce.py",
            "configs/smnist/default.yaml",
            "--seeds", "42",
            "--", "--mode", "tolerance", "--tol", "0.99"
        ]
    }

    for name, command in experiments.items():
        log_path = results_dir / f"smnist_{name}_seed42.log"

        print(f"\nRunning {name} experiment...")

        with open(log_path, "w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )

        print(f"Finished. Results saved to {log_path}")


if __name__ == "__main__":
    sMNIST_experiment()