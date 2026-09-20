"""
Erdos problem #884 -- quantum-testable lane.

Source metadata (data/problems.yaml in the manman4/erdosproblems clone):
    number: "884"
    prize: no
    informal_status: disproved (Lean formalization, last_update 2026-07-03)
    oeis: ["N/A"]
    tags: ["number theory", "divisors"]

LIMITATION (read this first): problem 884 carries **no OEIS sequence id**
(the field is literally "N/A" in the source data), and the erdosproblems
clone available in this environment contains only the metadata table --
no problem statement text -- so there is no way to derive the problem's
actual mathematical content here. This script therefore cannot build a
circuit that tests "the OEIS sequence for problem 884", because no such
sequence exists to test.

Per the task's fallback instruction ("if no genuine quantum circuit can be
constructed for this problem's sequence ... write the script anyway with
your best honest attempt, note the limitation clearly"), what follows is
a best-honest-attempt substitute: a REAL Grover-search circuit that tests
a genuine, finite, computable divisor-theoretic property -- membership in
the set of ABUNDANT NUMBERS (n whose proper divisors sum to more than n)
-- chosen because it matches problem 884's own tags ("number theory",
"divisors") even though it is not derived from an OEIS id belonging to
884 specifically (abundant numbers are OEIS A005101, used here only as a
representative divisor property, not as "the" sequence of problem 884).

Classical instance (N = 16, 4 qubits, computed from first principles
below, no lookup table):
    n : proper divisor sum : abundant?
    the classical loop below enumerates n = 1..15 and computes
    sigma(n) - n by trial division, exactly, in this script.

Grover circuit:
    - 4 qubits encode n in [0, 15] (n = 0 is a dummy, never abundant).
    - An oracle, built directly from the classical truth table above
      (a real phase-flip oracle on marked basis states, not a shortcut
      that just returns the answer), flips the phase of every n that is
      classically abundant.
    - ~ (pi/4) * sqrt(2^4 / M) Grover iterations are run, M = number of
      abundant marked states in [0,15].
    - The circuit is executed on the ideal AerSimulator and the most
      probable measured n must be a classically-abundant number for the
      run to PASS.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def proper_divisor_sum(n: int) -> int:
    """sigma(n) - n via trial division, first principles, no lookups."""
    if n <= 1:
        return 0
    total = 0
    for d in range(1, n):
        if n % d == 0:
            total += d
    return total


def is_abundant(n: int) -> bool:
    return proper_divisor_sum(n) > n


def build_oracle(marked_states, num_qubits):
    """Phase-flip oracle: multi-controlled Z on each marked computational
    basis state, built directly from the classical truth table."""
    oracle = QuantumCircuit(num_qubits, name="Oracle")
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            oracle.x(i)
        oracle.h(num_qubits - 1)
        oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        oracle.h(num_qubits - 1)
        for i in zero_positions:
            oracle.x(i)
    return oracle


def build_diffuser(num_qubits):
    diff = QuantumCircuit(num_qubits, name="Diffuser")
    diff.h(range(num_qubits))
    diff.x(range(num_qubits))
    diff.h(num_qubits - 1)
    diff.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    diff.h(num_qubits - 1)
    diff.x(range(num_qubits))
    diff.h(range(num_qubits))
    return diff


def main():
    N = 16
    num_qubits = 4

    # --- classical ground truth, computed here, from first principles ---
    classical_table = {n: is_abundant(n) for n in range(N)}
    marked_states = [n for n, abundant in classical_table.items() if abundant]

    print("Classical divisor-sum table for n = 0..15:")
    for n in range(N):
        print(f"  n={n:2d}  sigma(n)-n={proper_divisor_sum(n):3d}  "
              f"abundant={classical_table[n]}")
    print(f"Classically abundant numbers in [0,15]: {marked_states}")

    if not marked_states:
        raise RuntimeError("No abundant numbers found in range -- cannot "
                            "build a non-trivial Grover search.")

    # --- build Grover circuit ---
    oracle = build_oracle(marked_states, num_qubits)
    diffuser = build_diffuser(num_qubits)

    M = len(marked_states)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))

    # --- run on ideal AerSimulator ---
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-register bitstrings already read as standard
    # big-endian binary for n (verified against the oracle's marked
    # amplitude via a direct statevector check during development).
    def bitstring_to_int(bs):
        return int(bs, 2)

    freq = {}
    for bitstring, c in counts.items():
        n_val = bitstring_to_int(bitstring)
        freq[n_val] = freq.get(n_val, 0) + c

    most_probable = max(freq, key=freq.get)
    marked_probability = sum(c for n, c in freq.items() if n in marked_states) / shots

    print(f"\nGrover iterations used: {iterations}")
    print(f"Most probable measured n: {most_probable} "
          f"(count {freq[most_probable]}/{shots})")
    print(f"Total probability mass on abundant states: {marked_probability:.3f}")

    quantum_says_abundant = most_probable in marked_states
    classical_says_abundant = classical_table[most_probable]
    verified = quantum_says_abundant and classical_says_abundant and marked_probability > 0.5

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
