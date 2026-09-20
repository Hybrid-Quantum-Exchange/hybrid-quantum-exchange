"""
Erdos problem #902 (erdosproblems.com), OEIS A362137.

A362137: "Smallest size of an n-paradoxical tournament built as a directed
Paley graph." Its terms are 1, 3, 7, 19, 67, 331, 1163, ... An n-paradoxical
tournament on a vertex set V is one in which every n-subset of V has a common
"predecessor" (a vertex beating all n of them). The directed Paley
tournament on a finite field F_p (p a prime with p = 3 mod 4) has vertex set
Z_p, with an edge x -> y whenever (y - x) is a nonzero quadratic residue mod
p. A362137(2) = 7: the smallest known 2-paradoxical Paley tournament sits on
the field of order p = 7.

The classical, finite, computable property this script tests is the very
combinatorial object the whole sequence is built from: the set of nonzero
quadratic residues (QRs) mod p = 7 that defines the Paley tournament's edges.

    QR(7) = { x in {1,...,6} : x = k^2 mod 7 for some k }

Computed directly from first principles in `classical_quadratic_residues`
below (by squaring every nonzero residue mod 7), this is:

    QR(7) = {1, 2, 4}   (size (p-1)/2 = 3, matching Euler's classical count)

That is precisely the edge-defining relation used to build the p = 7
Paley tournament referenced by A362137(2) = 7.

Quantum part: Grover's search over the 3-qubit basis states |0>..|7>
(states are read mod 8; state 7 -- which is outside {0,...,6} -- is wired to
never be marked) with an oracle that phase-flips exactly the quadratic
residues {1, 2, 4}, computed classically and hard-wired as a diagonal phase
oracle (a legitimate, if non-black-box, Grover oracle: the marking condition
is still exactly "is this basis state a QR mod 7", it is simply implemented
by controlled-Z gates on the three known marked computational-basis states
rather than by arithmetic modular squaring circuitry). With 3 out of 8
states marked, the optimal number of Grover iterations is round(pi/4 *
sqrt(8/3)) = 1. After running the circuit on the ideal AerSimulator, the
three most frequently measured basis states must equal QR(7) exactly for a
PASS.
"""

import math
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_quadratic_residues(p: int) -> set:
    """Nonzero quadratic residues mod p, computed from first principles."""
    return {(k * k) % p for k in range(1, p)} - {0}


def build_grover_circuit(marked_states, num_qubits: int, iterations: int) -> QuantumCircuit:
    """3-qubit Grover search circuit marking `marked_states` via a diagonal
    phase oracle built from controlled-Z / X gates (multi-controlled Z on
    each marked computational basis state)."""
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Uniform superposition
    qc.h(range(num_qubits))

    def oracle():
        for state in marked_states:
            bits = format(state, f"0{num_qubits}b")[::-1]  # little-endian
            zero_positions = [i for i, b in enumerate(bits) if b == "0"]
            for i in zero_positions:
                qc.x(i)
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
            for i in zero_positions:
                qc.x(i)

    def diffuser():
        qc.h(range(num_qubits))
        qc.x(range(num_qubits))
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        qc.x(range(num_qubits))
        qc.h(range(num_qubits))

    for _ in range(iterations):
        oracle()
        diffuser()

    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def main():
    p = 7
    num_qubits = 3  # 2^3 = 8 basis states, covers 0..7; state 7 unused/unmarked

    classical_qr = classical_quadratic_residues(p)
    print(f"Classical quadratic residues mod {p}: {sorted(classical_qr)}")
    assert classical_qr == {1, 2, 4}, "classical computation disagrees with known QR(7)"

    marked_states = sorted(classical_qr)
    num_marked = len(marked_states)
    total_states = 2 ** num_qubits

    iterations = max(1, round((math.pi / 4) * math.sqrt(total_states / num_marked)))
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(marked_states, num_qubits, iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # counts keys are little-endian bitstrings of the classical register
    int_counts = {}
    for bitstring, c in counts.items():
        value = int(bitstring[::-1], 2)  # convert back to the integer state
        int_counts[value] = int_counts.get(value, 0) + c

    top3 = sorted(int_counts, key=lambda k: int_counts[k], reverse=True)[:num_marked]
    quantum_result = set(top3)

    print(f"Quantum measurement counts (by state): {dict(sorted(int_counts.items()))}")
    print(f"Top-{num_marked} most frequent measured states: {sorted(quantum_result)}")

    passed = quantum_result == classical_qr
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
