import argparse
from pathlib import Path
import optuna
import evaluate


class Objective:
    def __init__(self, solver: Path, n_processes: int):
        self.solver = solver
        self.n_processes = n_processes

    def __call__(self, trial: optuna.Trial):
        params = {
            "alpha": trial.suggest_loguniform("alpha", 0.01, 100),
            "window_size": trial.suggest_int("window_size", 11, 19, 2),
        }
        return evaluate.main(solver=self.solver, n_processes=self.n_processes, **params)


def main(solver: Path, n_processes: int, study_name: str, storage: str, n_trials: int):
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize"
    )
    objective = Objective(solver=solver, n_processes=n_processes)
    study.optimize(objective, n_trials=n_trials)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run cross validation.")
    parser.add_argument("solver", type=Path, help="Solver program.")
    parser.add_argument(
        "-n",
        "--n_processes",
        type=int,
        default=6,
        help="Number of processes execute in parallel.",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="ahc003",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///ahc003.db",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=40,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(**vars(args))
