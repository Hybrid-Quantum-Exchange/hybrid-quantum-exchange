"""
Erdos problem #256 — quantum-testable sequence lane.

LIMITATION (read first): problem #256's entry in erdosproblems/data/problems.yaml
(number: "256") has oeis: ["N/A"] and tags: ["analysis"], with no further
numeric detail in the metadata (informal_status: open, unformalized). There is
therefore no OEIS sequence id to derive a finite, computable property from for
this problem specifically. Per instructions, this script is the best honest
fallback: it does NOT fabricate a fake connection to problem #256's actual
mathematical content. Instead it builds and runs a REAL, genuine quantum
circuit (Grover's search) on a small, well-defined, independently-verifiable
finite search problem, and reports the result honestly.

Chosen finite instance (generic, not derived from problem #256's own content,
because no such content is numerically available):
    Search space: integers 0..15 (4 qubits).
    Property being searched for: x is divisible by 3 (x % 3 == 0).
    Classical answer (computed first, by brute force in this script):
        the exact set of marked values in {0, ..., 15}.

Grover's algorithm is used to amplify the marked states and the measurement
outcome is compared against the classical brute-force answer.

Because this is a generic fallback (no real OEIS/problem-256-specific content
to test), verified_against_classical here means "the quantum search result
statistically matches the classically-computed marked set" — not "problem
#256 has been verified." This limitation should be treated as an honest
ran_ok=True / verified_against_classical=True result for a GENERIC oracle,
not a problem-256-specific mathematical result.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
import math


# ---------------------------------------------------------------------------
# Step 1: classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N_QUBITS = 4
N = 2 ** N_QUBITS  # 16

def is_marked(x: int) -> bool:
    return x % 3 == 0

classical_marked = sorted(x for x in range(N) if is_marked(x))
num_marked = len(classical_marked)

assert classical_marked == [0, 3, 6, 9, 12, 15]
assert num_marked == 6


# ---------------------------------------------------------------------------
# Step 2: Grover oracle for "x % 3 == 0" over 4 qubits, built as an explicit
# multi-controlled phase flip over the precomputed marked basis states (this
# is a legitimate way to build an oracle for an arbitrary finite predicate;
# each marked computational basis state gets a controlled-Z-style phase flip).
# ---------------------------------------------------------------------------

def build_oracle(n_qubits: int, marked_values: list[int]) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for val in marked_values:
        bits = format(val, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def optimal_grover_iterations(n_items: int, n_marked: int) -> int:
    theta = math.asin(math.sqrt(n_marked / n_items))
    r = round((math.pi / (4 * theta)) - 0.5)
    return max(1, r)


def build_grover_circuit(n_qubits: int, marked_values: list[int]) -> QuantumCircuit:
    n_items = 2 ** n_qubits
    iterations = optimal_grover_iterations(n_items, len(marked_values))

    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc.decompose(reps=3)


# ---------------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

def run_grover(n_qubits: int, marked_values: list[int], shots: int = 4096):
    qc = build_grover_circuit(n_qubits, marked_values)
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    shots = 4096
    counts = run_grover(N_QUBITS, classical_marked, shots=shots)

    # Qiskit bitstrings are big-endian in the printed key (qubit n-1 ... qubit 0),
    # which matches standard integer interpretation of the classical bits since
    # we measured qubit i into classical bit i directly (little-endian classical
    # register order is reversed by Qiskit's string convention) -- normalize by
    # reversing before int() to be explicit and safe.
    def key_to_int(bitstring: str) -> int:
        return int(bitstring[::-1], 2)

    observed_counts = {}
    for bitstring, c in counts.items():
        val = key_to_int(bitstring)
        observed_counts[val] = observed_counts.get(val, 0) + c

    marked_hits = sum(c for v, c in observed_counts.items() if v in classical_marked)
    total = sum(observed_counts.values())
    marked_fraction = marked_hits / total

    # Grover's algorithm with a near-optimal iteration count should amplify
    # the marked subspace well above its uniform-random probability
    # (num_marked / N = 6/16 = 0.375); a strong pass threshold demonstrates
    # genuine amplification, not noise.
    uniform_fraction = num_marked / N
    # With M=6, N=16, the near-optimal 1-iteration Grover circuit has an
    # ideal (noiseless) success probability of sin(3*theta)^2 ~= 0.844, so a
    # 0.85 threshold could never pass even under perfect simulation; 0.75
    # keeps comfortable margin below the ideal value (room for shot noise)
    # while still requiring amplification far above the 0.375 uniform baseline.
    threshold = 0.75

    print("Erdos problem #256 quantum-testable sequence lane")
    print(f"OEIS id(s) for problem #256: N/A (none listed in problems.yaml)")
    print(f"Classical marked set (x in 0..{N-1} with x % 3 == 0): {classical_marked}")
    print(f"Uniform-random baseline probability of hitting a marked state: {uniform_fraction:.4f}")
    print(f"Observed fraction of shots landing on a marked state: {marked_fraction:.4f}")
    print(f"Top measured value counts: {sorted(observed_counts.items(), key=lambda kv: -kv[1])[:8]}")

    passed = marked_fraction >= threshold and all(
        v in classical_marked for v, c in observed_counts.items() if c >= total * 0.05
    )

    if passed:
        print("PASS")
    else:
        print("FAIL")

    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
