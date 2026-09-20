#!/usr/bin/env python3
"""Collect self-reported demo verdicts, with fail-closed local evidence receipts.

Use --lane NUMBER for a bounded run. The default requires all 999 source files.
PASS is a lane's own claim, never independent proof of an Erdos problem.
The existing lanes use unseeded simulator shots; this runner does not seed them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent
SEQUENCES_DIR = BASE_DIR / "sequences"
RESULTS_JSON = BASE_DIR / "RESULTS.json"
RESULTS_MD = BASE_DIR / "RESULTS.md"
RECEIPTS_DIR = BASE_DIR / ".runs"
REQUIREMENTS = BASE_DIR / "requirements.txt"
RUNNER = Path(__file__).resolve()
TIMEOUT_SECONDS = 120
EXPECTED_LANES = set(range(1, 1000))

# Whole-line grammar, audited against all 999 sources. Explanations may follow
# ':' or '('; a mere occurrence in narrative text is never a verdict.
VERDICT_RE = re.compile(
    r"^(?:(?:RESULT|PASS/FAIL):\s*|Overall verdict for Erdos problem #\d+:\s*)?"
    r"(PASS|FAIL)(?:\s*[:(].*)?$"
)
QUANTUM_RESULT_RE = re.compile(
    r"^Quantum result \[[\d, ]*\] (?:matches|does NOT match) "
    r"classical answer \[[\d, ]*\]: (PASS|FAIL)$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def file_evidence(path: Path) -> dict:
    try:
        data = path.read_bytes()
        return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
    except OSError as exc:
        return {"path": str(path.resolve()), "sha256": None, "error": str(exc)}


def create_run_directory() -> tuple[Path, dict]:
    started = utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex
    directory = RECEIPTS_DIR / run_id
    directory.mkdir(parents=True)  # Never reuse an existing run, even on collision.
    dependencies = {}
    for name in ("qiskit", "qiskit-aer", "numpy"):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "started_utc": started,
        "completed_utc": None,
        "status": "running",
        "runner": file_evidence(RUNNER),
        "requirements": file_evidence(REQUIREMENTS),
        "python": {"executable": sys.executable, "version": sys.version,
                   "implementation": platform.python_implementation()},
        "dependency_versions": dependencies,
        "seed_status": "not_controlled_by_runner; existing_lanes_unseeded",
        "evidence_scope": "self_reported_demo_results",
        "reviewer_status": "not_independently_reviewed",
        "proves_named_problem": False,
        "quantum_advantage_established": False,
        "hardware_execution_verified": False,
    }
    write_json(directory / "manifest.json", manifest)
    return directory, manifest


def lane_number(path: Path) -> int:
    match = re.fullmatch(r"problem_(\d+)\.py", path.name)
    return int(match.group(1)) if match else -1


def parse_verdict(stdout: str) -> tuple[str | None, str]:
    verdicts = set()
    for line in stdout.splitlines():
        line = line.strip()
        match = VERDICT_RE.fullmatch(line) or QUANTUM_RESULT_RE.fullmatch(line)
        if match:
            verdicts.add(match.group(1))
            # Reject conflicting verdict words even in a recognized explanation.
            explanation = line[match.end(1):]
            verdicts.update(re.findall(r"\b(?:PASS|FAIL)\b", explanation))
    if len(verdicts) > 1:
        return None, "contradictory_verdicts"
    if not verdicts:
        return None, "missing_verdict"
    return verdicts.pop(), "valid"


def raw_bytes(value: bytes | str | None) -> bytes:
    # TimeoutExpired can carry bytes even when a caller requested text mode.
    return value.encode("utf-8") if isinstance(value, str) else value or b""


def run_lane(path: Path, run_dir: Path | None = None) -> dict:
    """Run one lane; preserve raw logs and a receipt even on timeout/launch failure."""
    own_manifest = None
    if run_dir is None:
        run_dir, own_manifest = create_run_directory()
    path = path.resolve()
    lane_dir = run_dir / path.stem
    lane_dir.mkdir()  # A duplicate attempt cannot overwrite an earlier receipt.
    source = file_evidence(path)
    started = utc_now()
    start = time.monotonic()
    stdout, stderr = b"", b""
    returncode = None
    error = None
    execution_status = "clean"
    command = [sys.executable, str(path)]
    if source["sha256"] is None:
        execution_status, error = "source_error", source["error"]
    else:
        try:
            proc = subprocess.run(command, cwd=str(path.parent), capture_output=True,
                                  timeout=TIMEOUT_SECONDS)
            stdout, stderr = raw_bytes(proc.stdout), raw_bytes(proc.stderr)
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = raw_bytes(exc.stdout), raw_bytes(exc.stderr)
            execution_status, error = "timeout", str(exc)
        except OSError as exc:
            execution_status, error = "launch_error", str(exc)

    verdict, protocol_status = parse_verdict(stdout.decode("utf-8", errors="replace"))
    if execution_status == "clean":
        traceback = b"Traceback (most recent call last)" in stdout + stderr
        if traceback or returncode not in (0, 1) or (returncode == 1 and verdict != "FAIL"):
            execution_status = "failed"
            error = "traceback or exit code inconsistent with a clean reported verdict"
        elif file_evidence(path)["sha256"] != source["sha256"]:
            execution_status, error = "source_changed", "source changed during execution"
    ran_ok = execution_status == "clean" and protocol_status == "valid"
    reported_pass = ran_ok and verdict == "PASS"
    fallback = {"timeout": "TIMEOUT", "launch_error": "LAUNCH_ERROR",
                "source_error": "SOURCE_ERROR", "source_changed": "SOURCE_CHANGED",
                "failed": "CRASH"}
    result = {
        "number": lane_number(path),
        "file": path.name,
        "ran_ok": ran_ok,
        # Compatibility alias only: this has never been independent verification.
        "verified_against_classical": reported_pass,
        "verdict": fallback.get(execution_status) or verdict or (
                    "CONTRADICTORY_VERDICTS" if protocol_status == "contradictory_verdicts"
                    else "NO_VERDICT_PRINTED"),
        "reported_verdict": verdict,
        "returncode": returncode,
        "duration_s": round(time.monotonic() - start, 4),
        "started_utc": started,
        "completed_utc": utc_now(),
        "execution_status": execution_status,
        "protocol_status": protocol_status,
        "hypothesis_status": ("self_reported_pass" if reported_pass else
                              "self_reported_fail" if ran_ok else "not_established"),
        "reviewer_status": "not_independently_reviewed",
        "self_reported_demo_pass": reported_pass,
        "proves_named_problem": False,
        "source": source,
        "command": command,
        "cwd": str(path.parent),
        "timeout_seconds": TIMEOUT_SECONDS,
        "receipt_directory": str(lane_dir),
        "error": error,
    }
    (lane_dir / "stdout.log").write_bytes(stdout)
    (lane_dir / "stderr.log").write_bytes(stderr)
    result["logs"] = {name: file_evidence(lane_dir / name)
                      for name in ("stdout.log", "stderr.log")}
    write_json(lane_dir / "receipt.json", result)
    if own_manifest is not None:
        own_manifest.update(status="completed", completed_utc=utc_now(),
                            success=reported_pass, lane_count=1)
        write_json(run_dir / "manifest.json", own_manifest)
    return result


def write_results(results: list[dict], run_dir: Path) -> None:
    total = len(results)
    ran_ok = sum(r["ran_ok"] for r in results)
    passed = sum(r["self_reported_demo_pass"] for r in results)
    lines = [
        "# Self-reported demo results (generated by run_all.py)", "",
        f"- Lanes attempted: {total}",
        f"- ran_ok (clean execution and unambiguous verdict): {ran_ok}",
        f"- Self-reported demo PASS: {passed}",
        "- verified_against_classical is a legacy alias for self_reported_demo_pass.",
        "- Reviewer: not independently reviewed; no named problem proof, hardware or advantage claim.",
        "- Existing lanes are unseeded; this runner does not control simulator/transpiler seeds.",
        f"- Durable receipt directory: {run_dir}", "",
        "| # | file | verdict | execution | protocol | hypothesis |",
        "|---|------|---------|-----------|----------|------------|",
    ]
    for result in results:
        lines.append(f"| {result['number']} | {result['file']} | {result['verdict']} | "
                     f"{result['execution_status']} | {result['protocol_status']} | "
                     f"{result['hypothesis_status']} |")
    # Durable history first; legacy files are convenience copies of the latest run.
    write_json(run_dir / "RESULTS.json", results)
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(RESULTS_JSON, results)
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=int, help="run one lane by problem number")
    args = parser.parse_args(argv)
    run_dir, manifest = create_run_directory()
    results = []
    try:
        lanes = sorted(SEQUENCES_DIR.glob("problem_*.py"), key=lane_number)
        if args.lane is not None:
            lanes = [p for p in lanes if lane_number(p) == args.lane]
            if len(lanes) != 1:
                raise ValueError(f"Expected exactly one lane for problem {args.lane}; found {len(lanes)}")
        elif len(lanes) != 999 or {lane_number(p) for p in lanes} != EXPECTED_LANES:
            raise ValueError("Default run requires exactly problems 1..999; corpus is empty or incomplete")
        if manifest["runner"]["sha256"] is None or manifest["requirements"]["sha256"] is None:
            raise ValueError("Runner or requirements could not be hashed")
        manifest["selected_lanes"] = [lane_number(p) for p in lanes]
        write_json(run_dir / "manifest.json", manifest)
        for path in lanes:
            result = run_lane(path, run_dir)
            results.append(result)
            print(f"[{result['number']:>4}] {result['file']:<20} "
                  f"ran_ok={result['ran_ok']} verdict={result['verdict']}")
        write_results(results, run_dir)
        success = bool(results) and all(r["self_reported_demo_pass"] for r in results)
        manifest.update(status="completed", success=success, lane_count=len(results))
        passed = sum(r["self_reported_demo_pass"] for r in results)
        print(f"\n{passed}/{len(results)} self-reported demo PASS. Receipts: {run_dir}")
        return 0 if success else 1
    except (OSError, ValueError) as exc:
        manifest.update(status="error", success=False, error=str(exc), lane_count=len(results))
        print(f"Runner error: {exc}. Receipts: {run_dir}", file=sys.stderr)
        return 2
    finally:
        manifest["completed_utc"] = utc_now()
        write_json(run_dir / "manifest.json", manifest)


if __name__ == "__main__":
    sys.exit(main())
