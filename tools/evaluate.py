import sys
from multiprocessing import Pool
import subprocess
import argparse
from pathlib import Path
import re
import logging
import numpy as np
import scipy.stats
import pandas as pd
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(command: str):
    p = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
    )
    stdout, stderr = p.communicate()
    return stderr


def worker(args):
    infile, command = args
    s = time.time()
    out = run_command(command)
    t = time.time()
    g = re.search("Score = (\d+)", out)
    score = int(g.group(1))
    # print("{}: {}".format(infile, score))
    return score, t - s


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def main(solver: Path, n_processes: int, confidence: float=0.95, **kwargs):
    if not solver.exists():
        logger.info(f"file not exists: {str(solver)}", file=sys.stderr)
        return -1

    infiles = list(Path("./in").iterdir())
    
    args = []
    for infile in infiles:
        command = f"cargo run --release --bin tester {infile} python {solver}"
        for k, v in kwargs.items():
            command += f" --{k} {v}"
        args.append((str(infile), command))
    with Pool(processes=n_processes) as pool:
        result = pool.map(worker, args)
    df = pd.DataFrame(result, columns=["score", "time"])
    # logger.info(df.describe())
    mid, low, high = mean_confidence_interval(df["score"].values, confidence)
    logger.info("{:.1f}% confidence interval: {:.0f} [{:.0f}, {:.0f}]".format(confidence*100, mid, low, high))
    # return average score
    return mid


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run evaluation.")
    parser.add_argument("solver", type=Path, help="Solver program.")
    parser.add_argument(
        "-n",
        "--n_processes",
        type=int,
        default=6,
        help="Number of processes execute in parallel.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(**vars(args))
