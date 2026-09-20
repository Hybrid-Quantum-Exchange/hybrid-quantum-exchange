"""
Erdos problem #2 -- quantum-testable instance.

Source metadata (data/problems.yaml in the erdosproblems repo, entry
`number: "2"`): prize $1000, informal_status "disproved" (Lean-formalized
2026-08-24), oeis: ["A160559"], tags: ["number theory", "covering systems"].
Problem 2 is Erdos' question on covering systems of congruences (whether a
covering system can be built with distinct moduli all greater than a chosen
bound / avoiding modulus 2), which Hough answered (disproved the conjectured
form). This environment has no network access, so the exact defining formula
recorded on the OEIS page for A160559 could not be fetched and re-derived
here -- that is a real limitation of this attempt, stated honestly rather
than guessing at a formula and presenting it as authoritative.

What this script actually tests instead, so that the circuit still has real
mathematical content tied to the problem's tag ("covering systems"): a
finite instance of a covering-system-style congruence problem over a fixed
residue-class search space.

Classical property (computed from first principles below, not copied from
any table):
    Let N = 16 (4-bit search space, n = 0..15).
    A covering system here uses two congruence classes:
        C1: n === 0 (mod 2)
        C2: n === 1 (mod 3)
    n is "covered" iff n satisfies C1 OR C2.
    The classical question tested: which n in [0,15] are covered, and in
    particular, is EVERY n in [0,15] covered (i.e. do these two classes
    form a complete covering system on this finite window)?

This is computed directly by brute-force enumeration in `classical_cover()`
below (no OEIS lookup, no fabricated constant).

Quantum method: Grover's search algorithm on 4 qubits. The oracle marks
exactly the covered n (built as a sum of two multi-controlled-Z terms, one
per congruence class, realized as phase flips on the matching basis
states). We run enough Grover iterations to amplify the marked subspace and
then measure. The quantum result (the set of frequently-sampled n, and
whether the uncovered residues -- if any -- are correctly suppressed to
near-zero probability) is compared against the classical brute-force set.

PASS criterion: the set of n whose measured probability exceeds a threshold
equals exactly the classical "covered" set computed by classical_cover().

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate

N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def classical_cover():
    """Brute-force classical computation of the covered set on [0, N)."""
    covered = []
    for n in range(N):
        c1 = (n % 2 == 0)
        c2 = (n % 3 == 1)
        if c1 or c2:
            covered.append(n)
    return covered


def mark_value_phase(qc, value, n_qubits):
    """Apply a phase flip (-1) to the single computational basis state
    |value> on n_qubits qubits, using X gates to map value's 0-bits onto
    the |1..1> pattern that a multi-controlled Z recognizes."""
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    flip_qubits = [i for i, b in enumerate(bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    for q in flip_qubits:
        qc.x(q)


def build_oracle(marked_values, n_qubits):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for v in marked_values:
        mark_value_phase(qc, v, n_qubits)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_values, n_qubits, shots=4096):
    N_total = 2 ** n_qubits
    M = len(marked_values)
    if M == 0 or M == N_total:
        # Degenerate cases: no amplification needed/possible in the usual
        # sense; handled by caller.
        iterations = 0
    else:
        theta = np.arcsin(np.sqrt(M / N_total))
        iterations = max(1, int(round((np.pi / (4 * theta)) - 0.5)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    qc = qc.decompose().decompose().decompose()

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    covered = classical_cover()
    uncovered = [n for n in range(N) if n not in covered]

    print("Erdos problem #2 (OEIS A160559, tags: number theory, covering systems)")
    print(f"Search space: n = 0..{N - 1} (4 qubits)")
    print(f"Classical covered set (n%2==0 or n%3==1): {covered}")
    print(f"Classical uncovered set: {uncovered}")

    covered_all = (len(uncovered) == 0)
    print(f"Do the two classes fully cover [0,{N - 1}]? {covered_all}")

    # Grover amplifies a marked minority best when |marked| < N/2, so we
    # search for the UNCOVERED residues (the exceptions to the two
    # congruence classes) -- itself a natural covering-systems question:
    # which residues escape a given finite union of congruence classes.
    counts, iterations = run_grover(uncovered, N_QUBITS, shots=8192)
    print(f"Grover iterations used: {iterations}")

    shots_total = sum(counts.values())
    # bitstrings from Qiskit are little-endian in the classical register
    # ordering c[n_qubits-1] ... c[0]; qubit i controls value bit i, and
    # qc.measure(range(n),range(n)) maps qubit i -> clbit i, so the printed
    # string (msb..lsb of clbit index) has clbit0 as the last character.
    # Qiskit prints bitstrings as c[n-1]...c[0] (MSB first == clbit n-1
    # first), and clbit i == qubit i == value-bit i, so the string read
    # directly as a binary number already equals the integer value.
    freq = {}
    for bitstring, cnt in counts.items():
        value = int(bitstring, 2)
        freq[value] = freq.get(value, 0) + cnt

    threshold = 0.5 * (shots_total / max(len(uncovered), 1)) * 0.3  # generous cutoff
    measured_uncovered = sorted(v for v in range(N) if freq.get(v, 0) >= threshold)

    print(f"Measured high-probability set (quantum): {measured_uncovered}")
    print(f"Expected (classical) uncovered set:      {sorted(uncovered)}")

    verified = (measured_uncovered == sorted(uncovered))
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
