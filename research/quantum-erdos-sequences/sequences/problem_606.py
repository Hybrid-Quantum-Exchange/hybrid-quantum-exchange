"""
Erdos problem #606 -- quantum-testable sequence attempt.

Source metadata (from erdosproblems.com data, data/problems.yaml, block
"number: \"606\""):
    prize: no
    informal_status: solved (2025-08-31)
    tags: ["geometry"]
    oeis: ["N/A"]

LIMITATION (reported honestly, per task instructions): problem 606 has NO
OEIS sequence attached -- its `oeis` field is the literal string "N/A", not
an A-number. There is therefore no OEIS-derived integer sequence whose
membership/term/counting property could legitimately be encoded here. It
would be fabrication to invent an OEIS id or to pretend a term of "the
606 sequence" exists when none is defined in the source data. Per the task's
fallback instructions ("write the script anyway with your best honest
attempt ... report ran_ok/verified_against_classical accurately"), this
script does NOT attempt to test any property of problem 606 itself.

Instead, as the best-effort fallback, it runs a genuine, self-contained
quantum computation on a small, real, classically-checkable arithmetic
property -- Grover's search for perfect squares modulo N -- so that the
lane still produces a working, verified quantum circuit rather than a
fabricated or faked result. This property is NOT claimed to be related to
Erdos problem 606's content in any way; it exists only because problem 606
itself supplied no computable sequence to build a circuit around.

Classical property tested (computed from first principles below, not
copied from any table):
    For N = 16 (4 qubits, search space {0, ..., 15}), find all x such that
    x is a quadratic residue mod 16 AND x != 0, i.e. x = k^2 mod 16 for some
    k in {1, ..., 15}. The classical answer is computed by brute force in
    this script.

Grover's algorithm is used to amplify the marked (quadratic-residue) basis
states, and the result is checked against the classical brute-force answer.
"""

import sys
import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def classical_quadratic_residues(n_mod):
    """Brute-force: all nonzero x in [0, n_mod) such that x = k^2 mod n_mod
    for some k in [1, n_mod)."""
    residues = set()
    for k in range(1, n_mod):
        residues.add((k * k) % n_mod)
    residues.discard(0)
    return sorted(residues)


def build_oracle(marked_states, n_qubits):
    """Phase oracle flipping the sign of each marked computational basis
    state (given as little-endian bit tuples is NOT used here -- we use
    plain integers and standard big-endian-to-qubit mapping via binary
    string, consistent with how we read results back)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")
        # Flip qubits that are 0 in this state so that the all-ones pattern
        # corresponds exactly to `state`.
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_states, n_qubits, shots=2048):
    n_marked = len(marked_states)
    n_total = 2 ** n_qubits
    # Optimal number of Grover iterations.
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked_states, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    classical_answer = classical_quadratic_residues(N)
    print(f"Classical (first-principles) quadratic residues mod {N}: {classical_answer}")

    counts, iterations = run_grover(classical_answer, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    # Sort measured outcomes by frequency; take the top len(classical_answer)
    # as the quantum-found candidate set.
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = sorted_counts[: len(classical_answer)]
    quantum_states = sorted(int(bitstring, 2) for bitstring, _ in top_k)

    total_shots = sum(counts.values())
    marked_shots = sum(c for bitstring, c in counts.items()
                        if int(bitstring, 2) in classical_answer)
    marked_fraction = marked_shots / total_shots

    print(f"Quantum top-{len(classical_answer)} measured states: {quantum_states}")
    print(f"Fraction of shots landing on a true marked state: {marked_fraction:.3f}")

    passed = (
        quantum_states == classical_answer
        and marked_fraction > 0.8
    )

    if passed:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
