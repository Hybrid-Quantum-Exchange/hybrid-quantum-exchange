"""
Erdos problem #660 (from https://github.com/manman4/erdosproblems,
data/problems.yaml, entry `number: "660"`).

Source metadata for #660 (verbatim from the data file):
    prize: "no"
    status: open (last_update 2025-08-31)
    oeis: ["possible"]
    tags: ["geometry", "distances", "convex"]
    comments: "ambiguous statement"

IMPORTANT / HONEST LIMITATION: problem #660 does NOT carry a real OEIS
sequence id. Its `oeis` field is the literal string "possible" -- a
placeholder the source repository uses when no sequence has actually been
identified/assigned, not an id of the form A-NNNNNN. There is therefore no
concrete, citable integer sequence for this problem to build a quantum
"is n a member of the sequence" oracle around, and the problem's own
statement is flagged by the source data as "ambiguous statement". Per the
task instructions, we do not fabricate a fake OEIS-backed property for
#660.

What this script actually does instead (best honest attempt):
Since no genuine OEIS-derived property is available for #660, this script
demonstrates a REAL, self-contained, verifiable quantum computation on a
small finite/computable number-theoretic property -- "is n a perfect
square?" for n in {0, ..., 15} -- using an authentic Grover search circuit
on qiskit_aer's AerSimulator. This is included so the file is a genuine,
runnable, verified quantum circuit rather than an empty placeholder, but
it should NOT be read as representing Erdos problem #660's own sequence:
it is a stand-in demonstration circuit, clearly documented as such.

Classical property tested: for n in {0, 1, ..., 15} (4 bits), is n a
perfect square? The classical answer set (computed here directly, by
brute-force integer-square-root checking, not copied from anywhere) is
{0, 1, 4, 9}.

Quantum approach: Grover's algorithm over the 4-qubit computational basis
{0,...,15}. The oracle phase-flips exactly the marked states {0,1,4,9}
(implemented as an explicit multi-controlled-Z per marked bitstring, so
the oracle is built directly from the classically-computed answer set,
not hard-coded as "the answer"). One Grover iteration (optimal for
4 marked states out of 16) is applied, and the resulting distribution is
measured on the ideal AerSimulator. PASS means the simulator's most-likely
outcomes are exactly the classically verified perfect squares.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed here from first principles.
# ---------------------------------------------------------------------------

N_QUBITS = 4
N = 2 ** N_QUBITS  # search space {0, ..., 15}


def is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


classical_marked = sorted(n for n in range(N) if is_perfect_square(n))
assert classical_marked == [0, 1, 4, 9], classical_marked


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classically-computed marked set.
# ---------------------------------------------------------------------------

def apply_multi_controlled_z(qc: QuantumCircuit, qubits) -> None:
    """Phase-flip the |11...1> state on the given qubits (>=1 qubits)."""
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def mark_state(qc: QuantumCircuit, value: int, n_qubits: int) -> None:
    """Phase-flip the computational basis state |value> (n_qubits wide)."""
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    flip = [i for i, b in enumerate(bits) if b == 0]
    for i in flip:
        qc.x(i)
    apply_multi_controlled_z(qc, list(range(n_qubits)))
    for i in flip:
        qc.x(i)


def build_oracle(marked_values, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for v in marked_values:
        mark_state(qc, v, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    apply_multi_controlled_z(qc, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked_values, n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

def main() -> bool:
    M = len(classical_marked)  # number of marked items = 4
    # Optimal number of Grover iterations for M marked out of N=16.
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))

    qc = build_grover_circuit(classical_marked, N_QUBITS, iterations)

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    shots = 4096
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's count-key convention: the rightmost character is classical
    # bit 0 (qubit 0), which is already the standard binary-string
    # convention, so parsing the key directly as binary gives the integer
    # value (verified empirically against a single-qubit X test).
    def key_to_int(bitstring: str) -> int:
        return int(bitstring, 2)

    outcome_counts = {}
    for bitstring, c in counts.items():
        outcome_counts[key_to_int(bitstring)] = outcome_counts.get(key_to_int(bitstring), 0) + c

    # The M most frequently measured outcomes are the quantum-search result.
    top_outcomes = sorted(outcome_counts.items(), key=lambda kv: -kv[1])[:M]
    quantum_marked = sorted(v for v, _ in top_outcomes)

    total_marked_prob = sum(c for v, c in outcome_counts.items() if v in classical_marked) / shots

    print(f"Erdos problem #660 -- OEIS: none (source field is literally 'possible');")
    print("no real sequence id exists for this problem, so this script runs a")
    print("clearly-labeled stand-in demonstration circuit instead (see docstring).")
    print(f"Search space: n in [0, {N - 1}] ({N_QUBITS} qubits)")
    print(f"Property tested: n is a perfect square")
    print(f"Classical answer (brute force): {classical_marked}")
    print(f"Grover iterations used: {iterations}")
    print(f"Quantum measured top-{M} outcomes: {quantum_marked}")
    print(f"Total measured probability mass on classically-correct states: {total_marked_prob:.4f}")

    passed = quantum_marked == classical_marked and total_marked_prob > 0.8
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
