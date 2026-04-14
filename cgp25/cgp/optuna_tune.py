#!/usr/bin/env python3
import argparse
import ast
import json
import shutil
import pathlib
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from dataclasses import dataclass

import optuna
from optuna.distributions import IntDistribution
from optuna.trial import TrialState, create_trial

# Example run
# python3 optuna_tune.py --trials 8 --workers 8 --run-timeout 1800

ROOT = pathlib.Path(__file__).resolve().parent
HEADER_PATH = ROOT / "cgp.h"
TRIAL_COPY_FILES = ["cgp.cpp", "cgp.h", "makefile"]


DEFINE_KEYS = [
    "POPULACE_MAX",
    "MUTACE_MAX",
    "N_SHUFFLE",
    "PARAM_GENERATIONS",
]

SEARCH_PARAM_KEYS = [
    "POPULACE_MAX",
    "MUTACE_MAX",
    "N_SHUFFLE",
]


DEFINE_PATTERNS = {
    key: re.compile(rf"^(\s*#define\s+{key}\s+)(\d+)(.*)$", re.MULTILINE)
    for key in DEFINE_KEYS
}

BLK_RE = re.compile(r"Best chromosome blk:\s*(\d+)\s*/\s*(\d+)")
FIT_RE = re.compile(r"Best chromosome fitness:\s*(\d+)\s*/\s*(\d+)")
TRIAL_LINE_RE = re.compile(
    r"Trial\s+\d+\s+finished\s+with\s+value:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"\s+and\s+parameters:\s*(\{.*\})"
)
SCRIPT_LINE_RE = re.compile(
    r"trial=\d+\s+params:\s*(\{.*\})\s+score_blk=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
SCRIPT_LINE_RE_ALT = re.compile(
    r"trial=\d+\s+params=(\{.*\})\s+score_blk=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


@dataclass
class RunResult:
    blk: int
    blk_max: int
    fitness: int
    fitness_max: int
    stdout: str
    stderr: str


def read_header() -> str:
    return HEADER_PATH.read_text(encoding="utf-8")


def write_header(text: str) -> None:
    HEADER_PATH.write_text(text, encoding="utf-8")


def prepare_trial_root() -> pathlib.Path:
    trial_root = pathlib.Path(tempfile.mkdtemp(prefix="cgp_optuna_"))
    for filename in TRIAL_COPY_FILES:
        shutil.copy2(ROOT / filename, trial_root / filename)
    shutil.copytree(ROOT / "data", trial_root / "data")
    return trial_root


def set_defines(header_text: str, values: dict[str, int]) -> str:
    updated = header_text
    for key, value in values.items():
        pattern = DEFINE_PATTERNS[key]
        match = pattern.search(updated)
        if not match:
            raise RuntimeError(f"Could not find #define for {key} in {HEADER_PATH}")
        updated = pattern.sub(rf"\g<1>{value}\g<3>", updated, count=1)
    return updated


def extract_defines(header_text: str, keys: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in keys:
        pattern = DEFINE_PATTERNS[key]
        match = pattern.search(header_text)
        if not match:
            raise RuntimeError(f"Could not find #define for {key} in header text")
        out[key] = int(match.group(2))
    return out


def run_cmd(cmd: list[str], timeout: int, cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def parse_run_output(stdout: str, stderr: str) -> RunResult:
    blk_match = BLK_RE.search(stdout)
    fit_match = FIT_RE.search(stdout)
    if not blk_match:
        raise RuntimeError(f"Could not parse block count from output.\nOutput:\n{stdout}\nStderr:\n{stderr}")
    if not fit_match:
        raise RuntimeError(f"Could not parse fitness from output.\nOutput:\n{stdout}\nStderr:\n{stderr}")

    return RunResult(
        blk=int(blk_match.group(1)),
        blk_max=int(blk_match.group(2)),
        fitness=int(fit_match.group(1)),
        fitness_max=int(fit_match.group(2)),
        stdout=stdout,
        stderr=stderr,
    )


def build_and_run(run_name: str, build_timeout: int, run_timeout: int, trial_root: pathlib.Path) -> RunResult:
    build = run_cmd(["make"], timeout=build_timeout, cwd=trial_root)
    if build.returncode != 0:
        raise RuntimeError(
            "Build failed\n"
            f"stdout:\n{build.stdout}\n"
            f"stderr:\n{build.stderr}\n"
        )

    try:
        run = run_cmd(["./cgp", run_name], timeout=run_timeout, cwd=trial_root)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        # If the program produced final stats before timeout, use them.
        # Otherwise rethrow and let objective mark trial as failed.
        if BLK_RE.search(stdout) and FIT_RE.search(stdout):
            return parse_run_output(stdout, stderr)
        raise RuntimeError(
            "Execution timed out before final stats were printed\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}\n"
        ) from exc

    if run.returncode != 0:
        raise RuntimeError(
            "Execution failed\n"
            f"stdout:\n{run.stdout}\n"
            f"stderr:\n{run.stderr}\n"
        )

    return parse_run_output(run.stdout, run.stderr)


def run_post_trial_plots(trial_root: pathlib.Path, run_name: str) -> None:
    col_input = trial_root / f"{run_name}_col.txt"
    xls_input = trial_root / f"{run_name}.xls"
    if not col_input.is_file() or not xls_input.is_file():
        return

    plots_dir = trial_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        [sys.executable, str(ROOT / "./scripts/col_visualizer.py"), str(col_input), "-o", str(trial_root / f"{run_name}_col_visualization.svg")],
        [sys.executable, str(ROOT / "./scripts/plot_logs.py"), "--logs-dir", str(trial_root), "--pattern", f"{run_name}.xls", "--single", f"{run_name}.xls", "--outdir", str(plots_dir)],
    ]

    for cmd in commands:
        try:
            subprocess.run(
                cmd,
                cwd=trial_root,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except Exception as exc:
            print(f"plot skip ({run_name}): {exc}", file=sys.stderr, flush=True)


def save_trial_logs(
    trial_root: pathlib.Path,
    logs_root: pathlib.Path,
    trial_number: int,
    run_name: str,
    params: dict[str, int],
    result: RunResult | None,
    error: str | None,
) -> None:
    trial_dir = logs_root / f"trial_{trial_number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Copy only the CGP output files for this trial.
    file_patterns = [
        f"{run_name}*.xls",
        f"{run_name}*_col.txt",
        f"{run_name}*.chr",
        f"{run_name}*.svg",
    ]
    for pattern in file_patterns:
        for src in trial_root.glob(pattern):
            if src.is_file():
                shutil.copy2(src, trial_dir / src.name)

    trial_plots_dir = trial_root / "plots"
    if trial_plots_dir.is_dir():
        for src in trial_plots_dir.glob("*.png"):
            if src.is_file():
                shutil.copy2(src, trial_dir / src.name)

    summary = {
        "trial": trial_number,
        "run_name": run_name,
        "params": params,
        "status": "ok" if error is None else "failed",
        "error": error,
    }
    if result is not None:
        summary["blk"] = result.blk
        summary["blk_max"] = result.blk_max
        summary["fitness"] = result.fitness
        summary["fitness_max"] = result.fitness_max

    (trial_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def search_distributions() -> dict[str, IntDistribution]:
    return {
        "POPULACE_MAX": IntDistribution(4, 64),
        "MUTACE_MAX": IntDistribution(2, 24),
        "N_SHUFFLE": IntDistribution(100, 50000, log=True),
    }


def add_seed_trials(study: optuna.Study, seed_summary_paths: list[str]) -> int:
    if not seed_summary_paths:
        return 0

    dists = search_distributions()
    added = 0
    for path_str in seed_summary_paths:
        p = pathlib.Path(path_str)
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            print(f"seed skip (missing file): {p}", file=sys.stderr, flush=True)
            continue

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if "params" not in data or "blk" not in data:
                raise ValueError("summary must contain params and blk")

            raw_params = data["params"]
            params = {k: int(raw_params[k]) for k in SEARCH_PARAM_KEYS}
            value = float(data["blk"])

            full_params = {}
            for k in DEFINE_KEYS:
                if k in raw_params:
                    full_params[k] = int(raw_params[k])
            if "PARAM_GENERATIONS" not in full_params:
                full_params["PARAM_GENERATIONS"] = 50000

            user_attrs = {
                "seed_from": str(p),
                "full_params": full_params,
            }
            if "fitness" in data:
                user_attrs["fitness"] = data["fitness"]
            if "fitness_max" in data and "fitness" in data:
                user_attrs["fitness"] = f"{data['fitness']}/{data['fitness_max']}"
            if "blk_max" in data:
                user_attrs["blk"] = f"{int(data['blk'])}/{int(data['blk_max'])}"
            else:
                user_attrs["blk"] = str(int(data["blk"]))

            trial = create_trial(
                params=params,
                distributions=dists,
                value=value,
                user_attrs=user_attrs,
                state=TrialState.COMPLETE,
            )
            study.add_trial(trial)
            added += 1
        except Exception as exc:
            print(f"seed skip ({p}): {exc}", file=sys.stderr, flush=True)

    return added


def trial_signature(value: float, params: dict[str, int]) -> tuple[int, int, int, float]:
    return (
        int(params["POPULACE_MAX"]),
        int(params["MUTACE_MAX"]),
        int(params["N_SHUFFLE"]),
        float(value),
    )


def parse_seed_trial_line(line: str) -> tuple[float, dict[str, int]] | None:
    m = TRIAL_LINE_RE.search(line)
    if m:
        value = float(m.group(1))
        raw_params = ast.literal_eval(m.group(2))
    else:
        m2 = SCRIPT_LINE_RE.search(line) or SCRIPT_LINE_RE_ALT.search(line)
        if not m2:
            return None
        raw_params = ast.literal_eval(m2.group(1))
        value = float(m2.group(2))

    if not isinstance(raw_params, dict):
        raise ValueError("parameters payload is not a dict")

    params = {k: int(raw_params[k]) for k in SEARCH_PARAM_KEYS}
    full_params = dict(params)
    full_params["PARAM_GENERATIONS"] = int(raw_params.get("PARAM_GENERATIONS", 50000))
    return value, full_params


def add_seed_trials_from_lines(study: optuna.Study, lines: list[str]) -> int:
    if not lines:
        return 0

    dists = search_distributions()
    added = 0
    seen: set[tuple[int, int, int, float]] = set()
    for t in study.trials:
        if t.value is None or not t.params:
            continue
        try:
            sig = trial_signature(float(t.value), {k: int(t.params[k]) for k in SEARCH_PARAM_KEYS})
            seen.add(sig)
        except Exception:
            continue

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = parse_seed_trial_line(line)
            if parsed is None:
                print(f"seed line skip (unrecognized format): {line}", file=sys.stderr, flush=True)
                continue

            value, full_params = parsed
            params = {k: int(full_params[k]) for k in SEARCH_PARAM_KEYS}
            sig = trial_signature(value, params)
            if sig in seen:
                continue
            trial = create_trial(
                params=params,
                distributions=dists,
                value=float(value),
                user_attrs={
                    "seed_from": "line",
                    "full_params": full_params,
                },
                state=TrialState.COMPLETE,
            )
            study.add_trial(trial)
            seen.add(sig)
            added += 1
        except Exception as exc:
            print(f"seed line skip: {exc}", file=sys.stderr, flush=True)

    return added


def read_seed_lines_from_files(paths: list[str]) -> list[str]:
    lines: list[str] = []
    for path_str in paths:
        p = pathlib.Path(path_str)
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            print(f"seed log skip (missing file): {p}", file=sys.stderr, flush=True)
            continue
        try:
            lines.extend(p.read_text(encoding="utf-8", errors="replace").splitlines())
        except Exception as exc:
            print(f"seed log skip ({p}): {exc}", file=sys.stderr, flush=True)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune CGP hyperparameters with Optuna.")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--build-timeout", type=int, default=120, help="Build timeout in seconds.")
    parser.add_argument("--run-timeout", type=int, default=900, help="Run timeout in seconds.")
    parser.add_argument("--sampler-seed", type=int, default=40, help="Seed for Optuna sampler.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel trial workers.")
    parser.add_argument(
        "--logs-dir",
        default="optuna_logs",
        help="Directory for persistent per-trial logs (relative to project root unless absolute).",
    )
    parser.add_argument(
        "--seed-summary",
        action="append",
        default=[],
        help="Path to summary.json from an old trial to preload into the fresh study. Can be passed multiple times.",
    )
    parser.add_argument(
        "--seed-line",
        action="append",
        default=[],
        help="Raw Optuna log line to preload as a completed trial. Can be passed multiple times.",
    )
    parser.add_argument(
        "--seed-log-file",
        action="append",
        default=[],
        help="Path to a text log file; matching Optuna 'Trial ... finished ... parameters ...' lines are seeded.",
    )
    args = parser.parse_args()

    original_header = read_header()
    logs_base = pathlib.Path(args.logs_dir)
    if not logs_base.is_absolute():
        logs_base = ROOT / logs_base
    logs_base_parent = logs_base.parent
    logs_base_name = logs_base.name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_root = logs_base_parent / f"{logs_base_name}_{stamp}"
    idx = 1
    while logs_root.exists():
        logs_root = logs_base_parent / f"{logs_base_name}_{stamp}_{idx}"
        idx += 1
    logs_root.mkdir(parents=True, exist_ok=False)
    print(f"Run logs directory: {logs_root}", flush=True)

    def objective(trial: optuna.trial.Trial) -> float:
        # param_generations = trial.suggest_int("PARAM_GENERATIONS", 40000, 400000, log=True)
        # n_shuffle_max = max(100, param_generations // 2)
        params = {
            "POPULACE_MAX": trial.suggest_int("POPULACE_MAX", 4, 64),
            "MUTACE_MAX": trial.suggest_int("MUTACE_MAX", 2, 24),
            "N_SHUFFLE": trial.suggest_int("N_SHUFFLE", 100, 100000, log=True),
            "PARAM_GENERATIONS": 1000000,
        }
        trial.set_user_attr("full_params", params)

        trial_root = prepare_trial_root()
        run_name = f"optuna_trial_{trial.number}"
        result: RunResult | None = None
        error: str | None = None
        try:
            trial_header = set_defines(original_header, params)
            (trial_root / "cgp.h").write_text(trial_header, encoding="utf-8")
            result = build_and_run(
                run_name=run_name,
                build_timeout=args.build_timeout,
                run_timeout=args.run_timeout,
                trial_root=trial_root,
            )
            trial.set_user_attr("fitness", f"{result.fitness}/{result.fitness_max}")
            trial.set_user_attr("blk", f"{result.blk}/{result.blk_max}")
            print(
                f"trial={trial.number} params={params} "
                f"score_blk={result.blk} fitness={result.fitness}/{result.fitness_max}",
                flush=True,
            )
            # User requested: score only by block count.
            return float(result.blk)
        except Exception as exc:
            error = str(exc)
            trial.set_user_attr("error", error)
            print(f"trial={trial.number} failed: {error}", file=sys.stderr, flush=True)
            return 1e6
        finally:
            run_post_trial_plots(trial_root, run_name)
            save_trial_logs(
                trial_root=trial_root,
                logs_root=logs_root,
                trial_number=trial.number,
                run_name=run_name,
                params=params,
                result=result,
                error=error,
            )
            shutil.rmtree(trial_root, ignore_errors=True)

    study = optuna.create_study(
        study_name="cgp_block_only",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )
    seeded = 0
    seeded += add_seed_trials(study, args.seed_summary)
    file_lines = read_seed_lines_from_files(args.seed_log_file)
    seeded += add_seed_trials_from_lines(study, file_lines)
    seeded += add_seed_trials_from_lines(study, args.seed_line)
    if seeded:
        print(f"Seeded {seeded} completed trial(s) into fresh study.", flush=True)

    try:
        study.optimize(objective, n_trials=args.trials, n_jobs=args.workers)
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not complete_trials:
            print("No completed trials; keeping original header defines.", flush=True)
            return 0

        best = study.best_params
        best_full = study.best_trial.user_attrs.get("full_params")
        if not isinstance(best_full, dict):
            best_full = {}
        baseline = extract_defines(original_header, DEFINE_KEYS)
        merged = dict(baseline)
        for k, v in best.items():
            merged[k] = int(v)
        for k in DEFINE_KEYS:
            if k in best_full:
                merged[k] = int(best_full[k])
        best_full = merged
        best_header = set_defines(original_header, {k: int(best_full[k]) for k in DEFINE_KEYS})
        write_header(best_header)

        print("\nBest trial:")
        print(f"  value (blk) = {study.best_value}")
        for key in DEFINE_KEYS:
            print(f"  {key} = {best_full[key]}")
    finally:
        # Keep best settings if optimization reached a best trial,
        # otherwise restore original header.
        if len(study.trials) == 0 or study.best_trial is None:
            write_header(original_header)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
