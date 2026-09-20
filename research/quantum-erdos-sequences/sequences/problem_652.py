"""
Erdos problem #652 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, problems.yaml, entry "number: 652"):
    prize: no
    status: proved (as of 2026-01-17)
    oeis: ["N/A"]
    tags: ["geometry", "distances"]

LIMITATION, stated honestly: problem 652 has NO OEIS sequence attached
(oeis: ["N/A"] in the source data). There is therefore no OEIS-sequence
membership/term property to test here, and this script does not fabricate
one. Instead, per the task's fallback instructions, this is a best-honest-
effort quantum circuit built from the problem's own tags ("geometry",
"distances"): a small, finite, computable geometric-distance predicate on
integer points on a line, solved with Grover's algorithm and checked
against a brute-force classical computation performed in this same script.

Classical property being tested
--------------------------------
Let i, j range independently over {0, 1, 2, 3} (points on the integer
line, i.e. 1-D coordinates -- the simplest nontrivial "geometry/distances"
search space). Mark the pairs (i, j) with i != j whose Euclidean distance
|i - j| equals 2. This is a direct, finite, brute-force-checkable
geometric-distance predicate (not sourced from OEIS, since none exists for
this problem).

Classical brute force (done first, in Python, from first principles) over
all 16 ordered pairs (i, j) in {0,1,2,3}^2 finds the solution set. Grover's
algorithm is then run over the same 4-qubit (2 qubits for i, 2 for j)
search space with an oracle built directly from that classical solution
set, and the measured high-probability outcomes are compared against it.

Result: PASS if the two most frequent measurement outcomes from the Aer
simulator match exactly the classically computed solution set; FAIL
otherwise.
"""

from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


def classical_solutions():
    """Brute-force all (i, j) in {0,1,2,3}^2 with i != j and |i-j| == 2."""
    sols = []
    for i, j in product(range(4), repeat=2):
        if i != j and abs(i - j) == 2:
            sols.append((i, j))
    return sols


def bits_for_pair(i, j):
    """
    Encode (i, j) as a 4-bit string q3 q2 q1 q0 where
    q1 q0 = i (2 bits, LSB first) and q3 q2 = j (2 bits, LSB first).
    Returns the bitstring in Qiskit's little-endian qubit order (q0 first).
    """
    bi = format(i, "02b")[::-1]  # q0 q1
    bj = format(j, "02b")[::-1]  # q2 q3
    return bi + bj  # q0 q1 q2 q3, index order low->high


def build_grover_circuit(solutions, n_qubits=4):
    qc = QuantumCircuit(n_qubits, n_qubits)

    # 1. Uniform superposition
    qc.h(range(n_qubits))

    # Determine number of Grover iterations: pi/4 * sqrt(N/M)
    N = 2 ** n_qubits
    M = len(solutions)
    theta = np.arcsin(np.sqrt(M / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    def apply_oracle(circuit):
        for (i, j) in solutions:
            bitstr = bits_for_pair(i, j)  # length n_qubits, index 0 = q0
            # Flip qubits that should be 0 so the target state becomes |111...1>
            zero_positions = [k for k, b in enumerate(bitstr) if b == "0"]
            for k in zero_positions:
                circuit.x(k)
            # Multi-controlled Z on all qubits (phase flip of |11..1>)
            circuit.h(n_qubits - 1)
            circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            circuit.h(n_qubits - 1)
            for k in zero_positions:
                circuit.x(k)

    def apply_diffuser(circuit):
        circuit.h(range(n_qubits))
        circuit.x(range(n_qubits))
        circuit.h(n_qubits - 1)
        circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        circuit.h(n_qubits - 1)
        circuit.x(range(n_qubits))
        circuit.h(range(n_qubits))

    for _ in range(iterations):
        apply_oracle(qc)
        apply_diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


def decode_bits(bitstring, n_qubits=4):
    """bitstring as returned by Qiskit (q_{n-1}...q0), decode back to (i, j)."""
    # Qiskit's classical register string is c3 c2 c1 c0 (MSB first in the string)
    # so reverse to get q0 q1 q2 q3 order matching bits_for_pair.
    bits = bitstring[::-1]
    i = int(bits[1] + bits[0], 2)
    j = int(bits[3] + bits[2], 2)
    return (i, j)


def main():
    solutions = classical_solutions()
    solutions_set = set(solutions)
    print("Erdos problem #652 (tags: geometry, distances; oeis: N/A)")
    print("Classical brute-force solutions for |i-j| == 2, i,j in {0,1,2,3}:")
    print(sorted(solutions_set))

    n_qubits = 4
    qc, iterations = build_grover_circuit(solutions, n_qubits=n_qubits)
    print(f"Grover iterations used: {iterations}")

    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    shots = 4096
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Take the top-M measured bitstrings (M = number of classical solutions)
    M = len(solutions)
    top_outcomes = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:M]
    print("Top measured outcomes (bitstring: count):")
    for bitstring, count in top_outcomes:
        decoded = decode_bits(bitstring, n_qubits)
        print(f"  {bitstring} -> {decoded} : {count}")

    measured_solutions = set(decode_bits(b, n_qubits) for b, _ in top_outcomes)

    verified = measured_solutions == solutions_set
    if verified:
        print("PASS: Grover search's top outcomes match the classical solution set.")
    else:
        print("FAIL: Grover search's top outcomes do NOT match the classical solution set.")
        print(f"  classical: {solutions_set}")
        print(f"  quantum:   {measured_solutions}")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
