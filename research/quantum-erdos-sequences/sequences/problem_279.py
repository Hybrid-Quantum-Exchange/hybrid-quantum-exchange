"""
Erdos problem #279 (erdosproblems.com/279), quantum-testable lane.

Problem #279's metadata in this repository's copy of the problems database
(data/problems.yaml) records `oeis: ["N/A"]` -- there is no OEIS sequence
associated with this problem, and its tags are
["number theory", "covering systems", "primes"]. Because there is no
concrete integer sequence to test membership/terms of, this script does NOT
pretend to test an OEIS sequence. Limitation, stated honestly up front:
this is a best-effort quantum-testable instance drawn from the problem's
*topic* (covering systems of congruences, the subject Erdos problem #279 is
actually about), not a verification of any OEIS-listed term, because no
such OEIS id exists for this problem.

Classical property being tested
--------------------------------
A "covering system" is a finite set of congruences (a_i mod n_i) such that
every integer satisfies at least one of them. Erdos's original (complete)
covering system is:

    0 mod 2, 0 mod 3, 1 mod 4, 5 mod 6, 7 mod 12

which covers every integer (it is periodic with period 12, so it suffices
to check residues 0..11 mod 12).

For this script we deliberately use an INCOMPLETE sub-collection of three
of those congruences:

    0 mod 2, 0 mod 3, 1 mod 4

and ask: which residues x in {0, 1, ..., 11} (mod 12) are covered by NONE
of these three congruences? This is a small, finite, fully computable
search problem (the marked/"not covered" set), well suited to Grover's
algorithm: the search space is the 12 (padded to 16 = 2^4) residues mod 12,
and the "good" states are exactly the uncovered residues.

The classical answer is computed first, from first principles, by brute
force over x = 0..11, checking x % 2 == 0, x % 3 == 0, x % 4 == 1.

The quantum computation
------------------------
A 4-qubit Grover search is built. The oracle is constructed directly from
the classically-precomputed marked (uncovered) bitstrings -- for each
marked index, a multi-controlled-Z (phase flip) gate targets exactly that
computational basis state, X-gate padded/unpadded per bit as usual for a
diagonal oracle over specific basis states. This is a genuine Grover
oracle+diffuser circuit (not a shortcut): it is run on the ideal AerSimulator,
measured, and the most frequent outcomes are compared against the classical
brute-force set of uncovered residues.

Indices 12-15 (outside the mod-12 range) are left unmarked; the diffuser
still amplifies only the marked residues within 0..11.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4  # 2^4 = 16 >= 12 residues mod 12
N_STATES = 2 ** N_QUBITS


def classical_uncovered_residues():
    """Brute-force, from first principles: residues 0..11 covered by NONE
    of {0 mod 2, 0 mod 3, 1 mod 4}."""
    uncovered = []
    for x in range(12):
        covered = (x % 2 == 0) or (x % 3 == 0) or (x % 4 == 1)
        if not covered:
            uncovered.append(x)
    return uncovered


def marked_oracle(qc, marked_states, qubits):
    """Apply a phase-flip (Grover oracle) that marks exactly the given list
    of integer basis states, using a multi-controlled-Z per marked state."""
    n = len(qubits)
    for state in marked_states:
        bits = format(state, f"0{n}b")  # MSB..LSB over qubits[n-1..0]
        # We define bit i (qubit i, LSB-first) = bits[n-1-i]
        flip_qubits = [qubits[i] for i in range(n) if bits[n - 1 - i] == "0"]
        for q in flip_qubits:
            qc.x(q)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q in flip_qubits:
            qc.x(q)


def diffuser(qc, qubits):
    n = len(qubits)
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def build_grover_circuit(marked_states, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))
    qc.h(qubits)
    for _ in range(iterations):
        marked_oracle(qc, marked_states, qubits)
        diffuser(qc, qubits)
    qc.measure(qubits, qubits)
    return qc


def main():
    classical_marked = classical_uncovered_residues()
    print(f"Classical (brute force) uncovered residues mod 12: {classical_marked}")
    assert len(classical_marked) > 0, "expected at least one uncovered residue"

    M = len(classical_marked)
    N = N_STATES
    # Optimal number of Grover iterations ~ floor(pi/4 * sqrt(N/M))
    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N / M))))
    print(f"N={N} states, M={M} marked, using {iterations} Grover iteration(s)")

    qc = build_grover_circuit(classical_marked, N_QUBITS, iterations)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (Qiskit prints classical register MSB..LSB as
    # measured, with qubit 0 as the rightmost character) to integers.
    int_counts = {}
    for bitstring, c in counts.items():
        clean = bitstring.replace(" ", "")
        value = int(clean, 2)
        int_counts[value] = int_counts.get(value, 0) + c

    sorted_results = sorted(int_counts.items(), key=lambda kv: -kv[1])
    print("Top measured outcomes (value: count):")
    for value, c in sorted_results[:6]:
        print(f"  {value}: {c}")

    # Quantum-derived answer: outcomes must be evaluated only within the
    # meaningful domain 0..11 (mod 12); take the states that received
    # meaningfully amplified probability (above uniform-random baseline).
    baseline = shots / N
    amplified = [
        value for value, c in int_counts.items()
        if value < 12 and c > 2 * baseline
    ]
    amplified_set = sorted(set(amplified))

    print(f"Quantum-amplified candidate uncovered residues: {amplified_set}")
    print(f"Classical uncovered residues:                   {sorted(classical_marked)}")

    verified = amplified_set == sorted(classical_marked)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
