"""Frozen, local-only Grover 1-of-16 demonstrator. No experiment runs on import."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import pathlib
import platform
import random
import sys
import time
import uuid
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "quantum-erdos-sequences" / "sequences" / "problem_123.py"
SOURCE_FILES = ("study.py", "README.md", "tests/test_study.py")
LABELS = tuple(format(i, "04b") for i in range(16))
SPEC = {
    "qubits": 4, "states": 16, "target": 13, "wrong_oracle_target": 12,
    "iterations": 3, "shots_per_arm": 4096,
    "seed_simulator": 1729, "seed_transpiler": 1729, "seed_classical": 1729,
    "max_parallel_threads": 1, "optimization_level": 1,
    "basis_gates": ["u3", "cx"], "noise": "none",
    "statevector_absolute_tolerance": 1e-12,
    "sampling_familywise_alpha": 0.01, "sampling_bins": 48,
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with pathlib.Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path, value):
    with pathlib.Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_text(path, value):
    with pathlib.Path(path).open("x", encoding="utf-8") as stream:
        stream.write(value)


def analytic_distribution(marked):
    # Independent closed-form two-dimensional amplitude amplification reference.
    theta = math.asin(1 / math.sqrt(16))
    marked_probability = math.sin((2 * 3 + 1) * theta) ** 2
    background = (1 - marked_probability) / 15
    return {label: marked_probability if int(label, 2) == marked else background for label in LABELS}


def analytic_reference():
    epsilon = math.sqrt(math.log(2 * SPEC["sampling_bins"] / SPEC["sampling_familywise_alpha"])
                        / (2 * SPEC["shots_per_arm"]))
    return {
        "formula": "p_marked = sin^2(7 * asin(1/4)); p_other = (1-p_marked)/15",
        "positive": analytic_distribution(13),
        "wrong_oracle": analytic_distribution(12),
        "uniform_classical": {label: 1 / 16 for label in LABELS},
        "sampling_absolute_tolerance": epsilon,
        "sampling_rule": "All 48 empirical bins must lie within epsilon of their predeclared probability; Hoeffding union bound, familywise alpha=0.01.",
        "amplification_rule": "Positive target-13 frequency > 0.5 and exceeds each control's target-13 frequency by > 2*epsilon; positive mode=13 and wrong-oracle mode=12.",
    }


def source_snapshot():
    return {name: sha(HERE / name) for name in SOURCE_FILES}


def dependency_snapshot():
    """Hash installed direct scientific distributions, not just version labels."""
    snapshot = {"python": platform.python_version(), "implementation": platform.python_implementation()}
    for name in ("qiskit", "qiskit-aer", "numpy"):
        distribution = importlib.metadata.distribution(name)
        files = distribution.files
        if files is None:
            raise RuntimeError(f"{name} has no installed file inventory")
        actual = []
        total_bytes = 0
        for relative in sorted(files, key=str):
            if str(relative).endswith(".pyc"):
                continue
            path = pathlib.Path(distribution.locate_file(relative))
            if not path.is_file():
                raise RuntimeError(f"{name} is missing an inventoried file: {relative}")
            actual.append((str(relative).replace("\\", "/"), sha(path)))
            total_bytes += path.stat().st_size
        metadata = {entry: distribution.read_text(entry) for entry in ("METADATA", "RECORD")}
        snapshot[name] = {
            "version": distribution.version,
            "installed_files_sha256": hashlib.sha256(canonical(actual)).hexdigest(),
            "files_hashed": len(actual), "bytes_hashed": total_bytes,
            "metadata_sha256": {entry: hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
                                for entry, text in metadata.items()},
        }
    # These are metadata fingerprints of the remaining installed environment,
    # not claims that all transitive package bytes were independently validated.
    snapshot["environment_metadata"] = {
        dist.metadata["Name"]: {"version": dist.version,
            "record_sha256": hashlib.sha256((dist.read_text("RECORD") or "").encode("utf-8")).hexdigest()}
        for dist in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower())
    }
    return snapshot


def prepare(path, original=ORIGINAL):
    manifest = {
        "schema_version": 1,
        "study_name": "Grover 1-of-16 known-target demonstrator",
        "created_utc": utc(), "specification": SPEC, "analytic_reference": analytic_reference(),
        "hypothesis": "The specified ideal Grover circuit amplifies known target 13 according to the closed-form distribution, unlike uniform sampling and a circuit whose oracle marks 12.",
        "original_source": {"path": str(pathlib.Path(original).resolve()), "sha256": sha(original),
                            "role": "historical provenance only; preserved and never imported"},
        "source_sha256": source_snapshot(), "dependencies": dependency_snapshot(),
        "circuit_semantics": {
            "initial_state": "|0000>, followed by H on all four qubits",
            "oracle": "phase -1 exactly on the chosen bitstring; target-13 positive arm and target-12 negative arm",
            "diffuser": "H^4 X^4 MCZ X^4 H^4, equal to the conventional diffuser up to a global minus sign",
            "bit_order": "Integer n uses qubit i for bit i; printed/count labels are q3 q2 q1 q0. 13=1101; 12=1100.",
            "iterations": "three oracle-plus-diffuser pairs; no primality computation is performed in-circuit",
            "classical_reference": "4096 independent uniform draws over integers 0..15 using Python random.Random(1729), not an optimized search algorithm",
            "export": "OpenQASM 2.0, strict Qiskit local parser round-trip plus statevector probability comparison; no OpenQASM 3.1 or formal validation claim",
        },
        "decision_rules": {
            "execution": "complete only if all three arms and durable artifacts finish; otherwise failed or not_run",
            "protocol": "passes only if frozen manifest/source/dependencies, fixed sampling totals, circuit dimensions, seeds and QASM round-trip requirements hold",
            "hypothesis": "passes only if protocol passes and every exact distribution, sampling tolerance, mode, and amplification comparison passes; otherwise fails or not_evaluated",
            "independent_review": "pending; never inferred from automated results",
            "T9_acceptance": "not_assessed",
        },
        "limitations": [
            "Not an Erdos-123 proof, problem-specific verification, or recovery of the historical unseeded run.",
            "Known target is hardcoded; this is a four-qubit algorithm demonstrator, not discovery of an unknown answer.",
            "No quantum advantage or speedup evidence: both quantum arms are classically simulated and the uniform reference is not a competitive classical search.",
            "Noiseless local simulation only; no providers, credentials, hardware, network, or installation operations.",
            "Statistical tolerances are predeclared; a seeded sample can fail and its record is retained.",
            "The statevector and Aer tools share Qiskit; only the closed-form numeric reference is independently derived.",
        ],
    }
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    digest = sha(path)
    write_text(path.with_suffix(path.suffix + ".sha256"), digest + "\n")
    return digest


def check_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("specification") != SPEC:
        raise ValueError("manifest does not match the fixed study specification")
    if manifest.get("analytic_reference") != analytic_reference():
        raise ValueError("manifest analytic reference differs from independent closed-form values")
    if manifest.get("source_sha256") != source_snapshot():
        raise ValueError("study source changed after protocol freeze")
    original = manifest["original_source"]
    if sha(original["path"]) != original["sha256"]:
        raise ValueError("original historical source changed after protocol freeze")
    if manifest.get("dependencies") != dependency_snapshot():
        raise ValueError("installed dependencies changed after protocol freeze")


def build_circuit(marked):
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit(4, name=f"grover_1of16_marks_{marked}")
    circuit.h(range(4))
    zero_bits = [i for i in range(4) if not (marked >> i) & 1]
    for _ in range(3):
        for i in zero_bits:
            circuit.x(i)
        circuit.h(3)
        circuit.mcx([0, 1, 2], 3)
        circuit.h(3)
        for i in zero_bits:
            circuit.x(i)
        circuit.h(range(4))
        circuit.x(range(4))
        circuit.h(3)
        circuit.mcx([0, 1, 2], 3)
        circuit.h(3)
        circuit.x(range(4))
        circuit.h(range(4))
    return circuit


def probabilities(circuit):
    from qiskit.quantum_info import Statevector
    values = Statevector.from_instruction(circuit).probabilities()
    return {label: float(values[i]) for i, label in enumerate(LABELS)}


def perform_local_study(directory):
    # All scientific imports are local libraries, deferred until explicit execute.
    from qiskit import qasm2, transpile
    from qiskit_aer import AerSimulator
    simulator = AerSimulator(method="statevector", max_parallel_threads=1)
    output = {}
    for arm, marked in (("positive", 13), ("wrong_oracle", 12)):
        started = utc()
        circuit = build_circuit(marked)
        exact = probabilities(circuit)
        write_text(directory / f"{arm}.circuit.txt", str(circuit.draw(output="text")) + "\n")
        measured = circuit.copy()
        measured.measure_all()
        transpiled = transpile(measured, simulator, optimization_level=1,
                               seed_transpiler=1729, basis_gates=["u3", "cx"])
        transpiled_exact = probabilities(transpiled.remove_final_measurements(inplace=False))
        qasm_text = qasm2.dumps(transpiled)
        write_text(directory / f"{arm}.qasm", qasm_text + "\n")
        restored = qasm2.loads(qasm_text, strict=True)
        restored_exact = probabilities(restored.remove_final_measurements(inplace=False))
        result = simulator.run(transpiled, shots=4096, seed_simulator=1729).result()
        if not result.success:
            raise RuntimeError(f"{arm} simulator result reported failure")
        raw_counts = dict(result.get_counts())
        counts = {label: int(raw_counts.get(label, 0)) for label in LABELS}
        if set(raw_counts) - set(LABELS):
            raise ValueError(f"{arm} returned unexpected count labels")
        evidence = {
            "oracle_target": marked, "shots": 4096, "counts": counts,
            "seed_simulator": 1729, "seed_transpiler": 1729,
            "exact_source": exact, "exact_transpiled": transpiled_exact,
            "exact_qasm_roundtrip": restored_exact,
            "qasm_format": "OpenQASM 2.0", "qasm_strict_parse": True,
            "qubits": transpiled.num_qubits, "classical_bits": transpiled.num_clbits,
            "transpiled_depth": transpiled.depth(),
            "transpiled_operations": dict(transpiled.count_ops()),
            "started_utc": started, "completed_utc": utc(),
        }
        write_json(directory / f"{arm}.json", evidence)
        output[arm] = evidence
    started = utc()
    generator = random.Random(1729)
    counts = {label: 0 for label in LABELS}
    for _ in range(4096):
        counts[format(generator.randrange(16), "04b")] += 1
    output["uniform_classical"] = {"counts": counts, "shots": 4096, "seed_classical": 1729,
                                   "started_utc": started, "completed_utc": utc()}
    write_json(directory / "uniform_classical.json", output["uniform_classical"])
    return output


def evaluate(evidence, reference):
    protocol, hypothesis = {}, {}
    expected_arms = {"positive", "wrong_oracle", "uniform_classical"}
    protocol["exact_arm_inventory"] = set(evidence) == expected_arms
    epsilon = reference["sampling_absolute_tolerance"]
    for arm in sorted(expected_arms):
        row = evidence.get(arm, {})
        counts = row.get("counts", {})
        valid_counts = (set(counts) == set(LABELS) and
                        all(type(v) is int and v >= 0 for v in counts.values()) and
                        sum(counts.values()) == 4096 and row.get("shots") == 4096)
        protocol[arm + "_counts"] = valid_counts
        if not valid_counts:
            continue
        observed = {label: counts[label] / 4096 for label in LABELS}
        hypothesis[arm + "_sampling"] = all(abs(observed[k] - reference[arm][k]) <= epsilon for k in LABELS)
        if arm == "uniform_classical":
            protocol[arm + "_seed"] = row.get("seed_classical") == 1729
            continue
        target = 13 if arm == "positive" else 12
        protocol[arm + "_circuit"] = (row.get("oracle_target") == target and row.get("qubits") == 4 and row.get("classical_bits") == 4)
        protocol[arm + "_seeds"] = row.get("seed_simulator") == row.get("seed_transpiler") == 1729
        protocol[arm + "_qasm"] = row.get("qasm_format") == "OpenQASM 2.0" and row.get("qasm_strict_parse") is True
        for phase in ("exact_source", "exact_transpiled", "exact_qasm_roundtrip"):
            distribution = row.get(phase, {})
            hypothesis[arm + "_" + phase] = (set(distribution) == set(LABELS) and
                all(type(distribution[k]) in (int, float) and math.isfinite(distribution[k]) and
                    abs(distribution[k] - reference[arm][k]) <= 1e-12 for k in LABELS))
        hypothesis[arm + "_mode"] = counts[format(target, "04b")] > max(v for k, v in counts.items() if k != format(target, "04b"))
    if all(protocol.values()):
        primary = evidence["positive"]["counts"]["1101"] / 4096
        hypothesis["target_amplified"] = primary > 0.5 and all(
            primary - evidence[arm]["counts"]["1101"] / 4096 > 2 * epsilon
            for arm in ("wrong_oracle", "uniform_classical"))
    protocol_status = "passes" if protocol and all(protocol.values()) else "fails"
    hypothesis_status = ("passes" if hypothesis and all(hypothesis.values()) else "fails") if protocol_status == "passes" else "not_evaluated"
    return protocol_status, hypothesis_status, {"protocol_checks": protocol, "hypothesis_checks": hypothesis}


def check_artifacts(directory, evidence):
    for arm in ("positive", "wrong_oracle", "uniform_classical"):
        stored = json.loads((directory / f"{arm}.json").read_text(encoding="utf-8"))
        if stored != evidence[arm]:
            raise ValueError(f"{arm} persisted evidence differs from the evaluated data")
        if arm != "uniform_classical":
            for suffix in ("circuit.txt", "qasm"):
                if not (directory / f"{arm}.{suffix}").read_text(encoding="utf-8").strip():
                    raise ValueError(f"{arm} has an empty {suffix} artifact")


def execute(manifest_path, expected_digest, output_root, experiment=None):
    """Only the root's explicit execute command may call the default experiment."""
    started_clock = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    directory = pathlib.Path(output_root).resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {"study_name": "Grover 1-of-16 known-target demonstrator", "run_id": run_id,
               "started_utc": utc(), "execution_status": "not_run", "protocol_status": "not_assessed",
               "hypothesis_status": "not_evaluated", "independent_review_status": "pending",
               "T9_acceptance": "not_assessed", "expected_manifest_sha256": expected_digest}
    write_json(directory / "start.json", receipt)
    experiment = experiment or perform_local_study
    try:
        actual_digest = sha(manifest_path)
        receipt["actual_manifest_sha256"] = actual_digest
        if actual_digest != expected_digest:
            raise ValueError("manifest digest does not match the reviewed frozen protocol")
        manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
        with (directory / "frozen_manifest.json").open("xb") as stream:
            stream.write(pathlib.Path(manifest_path).read_bytes())
        check_manifest(manifest)
        receipt["protocol_status"] = "incomplete"
        receipt["execution_status"] = "failed"
        evidence = experiment(directory)
        check_artifacts(directory, evidence)
        receipt["execution_status"] = "complete"
        protocol_status, hypothesis_status, checks = evaluate(evidence, manifest["analytic_reference"])
        # Repeat integrity checks after execution; altered sources cannot pass.
        check_manifest(manifest)
        receipt.update(protocol_status=protocol_status, hypothesis_status=hypothesis_status, **checks)
    except (Exception, KeyboardInterrupt) as exc:
        receipt["protocol_status"] = "fails"
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
    receipt["completed_utc"] = utc()
    receipt["elapsed_seconds"] = time.perf_counter() - started_clock
    receipt["artifacts_sha256"] = {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()}
    write_json(directory / "receipt.json", receipt)
    passed = all(receipt[key] == value for key, value in (
        ("execution_status", "complete"), ("protocol_status", "passes"), ("hypothesis_status", "passes")))
    print(json.dumps({"run_directory": str(directory), **{k: receipt[k] for k in (
        "execution_status", "protocol_status", "hypothesis_status", "independent_review_status", "T9_acceptance")}}, indent=2))
    return (0 if passed else 1), directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("prepare", help="freeze protocol and hashes; does not execute the experiment")
    freeze.add_argument("--output", type=pathlib.Path, required=True)
    freeze.add_argument("--original-source", type=pathlib.Path, default=ORIGINAL)
    run = sub.add_parser("execute", help="explicit local study execution under a reviewed manifest")
    run.add_argument("--manifest", type=pathlib.Path, required=True)
    run.add_argument("--expected-manifest-sha256", required=True)
    run.add_argument("--output-root", type=pathlib.Path, default=HERE / "runs")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(prepare(args.output, args.original_source))
        return 0
    return execute(args.manifest, args.expected_manifest_sha256, args.output_root)[0]


if __name__ == "__main__":
    sys.exit(main())
