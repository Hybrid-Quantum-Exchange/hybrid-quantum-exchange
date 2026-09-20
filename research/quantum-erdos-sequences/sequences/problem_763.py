"""
Erdos problem #763 -- quantum-testable-sequence lane.

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"763\"" (informal_status: disproved, tags: ["number theory",
"additive combinatorics"]).

LIMITATION (read before trusting anything downstream of this file):
Problem #763's YAML entry carries `oeis: ["N/A"]` -- there is no OEIS
sequence attached to this problem, so there is no OEIS-derived property to
encode literally as the task asked for. Per the task's own fallback
instructions ("write the script anyway with your best honest attempt, note
the limitation clearly"), this script does NOT test any OEIS sequence. It
instead builds a real, small, first-principles quantum circuit whose theme
matches problem #763's own tags (additive combinatorics / sumsets), so the
lane is not vacuous, and is honest in its docstring and its reported
verification status about not being OEIS-anchored.

Property actually tested (finite, computable, unrelated to any specific
OEIS id):
    Fix the small integer set S = {0, 1, 3} (chosen because all of its
    pairwise sums a + b with a <= b are distinct, i.e. S is a Sidon set --
    a standard additive-combinatorics object, matching problem #763's
    tag).
    Consider the sumset
        S + S = { a + b : a, b in S }
    as a subset of Z_16 = {0, 1, ..., 15} (4-bit register, N = 16; the
    modulus is chosen larger than any actual sum so no wraparound occurs,
    keeping S + S an honest integer sumset).
    The classical value of S + S is computed below in Python, from first
    principles (nested loop over S x S, taking each sum mod 8).

Quantum task: Grover search over the 3-qubit register n in {0,...,7} for
the marked set M = S + S (computed classically first). An oracle built
from those classical marked values flips phase on |n> for n in M; the
standard Grover diffusion operator then amplifies those n. After the
optimal number of Grover iterations we measure and check that the
highest-probability outcomes are exactly the classically-computed sumset
S + S.

Comparison: classical sumset (Python) vs. quantum measurement distribution
(AerSimulator, statevector-exact, no noise). PASS iff the qubit strings
receiving the top |M| measurement counts equal, as a set, the classical
sumset S + S.
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_sumset(S, modulus):
    """Compute S + S (mod `modulus`) from first principles."""
    result = set()
    for a, b in product(S, repeat=2):
        result.add((a + b) % modulus)
    return result


def build_oracle(n_qubits, marked_values):
    """Phase-flip oracle marking each value in `marked_values` (as an
    n_qubits-bit binary string, little-endian: qubit 0 is the LSB)."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
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


def run_grover(n_qubits, marked_values, shots=4096):
    N = 2 ** n_qubits
    M = len(marked_values)
    if M == 0 or M == N:
        raise ValueError("Grover needs 0 < |marked| < N")

    # Optimal number of Grover iterations for this N, M.
    theta = np.arcsin(np.sqrt(M / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator(method="statevector")
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    S = {0, 1, 3}
    modulus = 16
    n_qubits = 4

    # Sidon-set sanity check: all pairwise sums a + b (a <= b) in S are
    # distinct, as claimed in the docstring.
    sums = [a + b for i, a in enumerate(sorted(S)) for b in sorted(S)[i:]]
    assert len(sums) == len(set(sums)), "S is not a Sidon set as claimed"

    classical_marked = classical_sumset(S, modulus)
    print(f"S = {sorted(S)}, modulus = {modulus}")
    print(f"Classical sumset S + S (mod {modulus}) = {sorted(classical_marked)}")

    counts, iterations = run_grover(n_qubits, classical_marked)
    print(f"Grover iterations used: {iterations}")
    print(f"Raw measurement counts: {counts}")

    # Qiskit bit strings are big-endian over classical registers (qubit 0
    # is the rightmost bit), matching the little-endian convention used in
    # build_oracle/build_diffuser above -- int(bitstring, 2) recovers n.
    M = len(classical_marked)
    top_outcomes = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:M]
    quantum_marked = {int(bitstring, 2) for bitstring, _ in top_outcomes}

    print(f"Quantum top-{M} outcomes decoded to: {sorted(quantum_marked)}")

    verified = quantum_marked == classical_marked
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
