"""
Erdos problem #521 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 521"):
    prize: no
    status: open
    oeis: ["N/A"]
    tags: ["analysis", "polynomials", "probability"]

IMPORTANT LIMITATION, stated honestly up front: problem #521 has NO OEIS
sequence id attached in the source data (oeis: ["N/A"]). There is therefore
no actual integer sequence from this problem to test membership/terms of on
a quantum circuit. This script does NOT fabricate an OEIS value or claim a
verified connection to problem #521's actual open mathematical content
(which concerns analysis/probability of polynomials and is not a finite,
computably-checkable statement at circuit scale).

What this script does instead, as the best honest substitute the tags
support: it builds a genuine small finite computable arithmetic problem in
the spirit of the "polynomials" tag -- root-finding of a quadratic modulo a
power of two -- and solves it two ways:
  1. Classically, from first principles, by brute-force enumeration.
  2. Quantumly, with a real Grover search circuit on the ideal AerSimulator,
     whose oracle marks exactly the classically-verified root set.

The finite computable property tested:
    Find all x in Z_16 = {0, 1, ..., 15} such that
        (x^2 - 5x + 6) mod 16 == 0
    i.e. the roots of the polynomial p(x) = x^2 - 5x + 6 modulo 16.

This is computed classically in this script (see `classical_solve`), giving
the solution set {2, 3} out of 16 candidates. A 4-qubit Grover search is
then run with an oracle that marks exactly those two basis states (built
from the classically-derived solution set, as is standard for a Grover
"known marked items" oracle), and the circuit is verified to amplify
measurement probability onto {2, 3} with the correct number of Grover
iterations for a 2-out-of-16 search space.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16 candidates: x = 0..15


def classical_solve():
    """Brute-force, from first principles: roots of x^2 - 5x + 6 mod 16."""
    solutions = []
    for x in range(N):
        if (x * x - 5 * x + 6) % 16 == 0:
            solutions.append(x)
    return solutions


def build_oracle(marked, n_qubits):
    """Phase oracle: flips the sign of exactly the basis states in `marked`."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for target in marked:
        bits = format(target, f"0{n_qubits}b")[::-1]  # little-endian
        # Flip qubits that should be 0 in the target, so a multi-controlled Z
        # fires exactly when the register equals `target`.
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(i)
    return qc


def build_diffuser(n_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    solutions = classical_solve()
    print(f"Classical solve: roots of x^2 - 5x + 6 mod 16 in [0,15] = {solutions}")
    assert solutions == [2, 3], "unexpected classical result -- refusing to proceed"

    m = len(solutions)
    # Optimal Grover iteration count for m marked items out of N.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / m)))
    print(f"Grover iterations used: {iterations} (N={N}, m={m})")

    qc = build_grover_circuit(solutions, N_QUBITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit prints classical bits MSB-first (leftmost char = highest index),
    # matching our qubit-index-as-binary-digit convention directly.
    def bits_to_int(bs):
        return int(bs, 2)

    hits = sum(c for bs, c in counts.items() if bits_to_int(bs) in solutions)
    prob_marked = hits / shots
    print(f"Measured probability mass on marked states {solutions}: {prob_marked:.4f}")

    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    print("Top measured outcomes (bitstring, count, decoded x):")
    for bs, c in top:
        print(f"  {bs}  count={c}  x={bits_to_int(bs)}")

    # For N=16, m=2, one Grover iteration should put the vast majority of
    # amplitude on the marked subspace (theoretical success probability is
    # close to 1 for this size). Require it be clearly amplified above the
    # uniform baseline (m/N = 0.125) with good margin.
    verified = prob_marked > 0.85

    if verified:
        print("PASS: Grover search recovered the classical root set with high probability.")
    else:
        print("FAIL: Grover search did not sufficiently amplify the classical root set.")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
