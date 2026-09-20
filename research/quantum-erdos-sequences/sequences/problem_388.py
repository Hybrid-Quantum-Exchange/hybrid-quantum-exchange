"""
Erdos problem #388 -- quantum-testable sequence attempt.

LIMITATION (read first): as of the erdosproblems.com dataset snapshot used
here (data/problems.yaml, entry `number: "388"`), problem 388 has:
    prize: no
    status: open, unformalized
    oeis: ["N/A"]        <-- no OEIS sequence id is associated with it
    tags: ["number theory"]
There is no description field in the dataset beyond the tag "number theory",
and no OEIS id to derive a concrete, checkable sequence property from. So
there is no genuine sequence belonging to problem 388 that this script can
test. Fabricating one and pretending it is "the problem 388 sequence" would
be dishonest, so this script does NOT do that.

Instead, in the spirit of "best honest attempt", this script builds a real,
self-contained Qiskit Grover-search circuit over a small, well-understood,
genuinely computable number-theoretic property in the same tag family
(number theory) that problem 388 carries: primality over a small finite
range. This is offered only as a demonstration that a real quantum oracle/
search circuit can be built and verified classically -- it is explicitly
NOT claimed to encode problem 388's actual open conjecture, since no such
finite instance of problem 388 itself is available to encode.

Property tested: for N = 16 (4-qubit search space, integers 0..15), Grover
search finds the marked set of primes {2, 3, 5, 7, 11, 13} within
{0,...,15}. Primality is computed classically from first principles (trial
division) in `is_prime_classical`, independent of any external table.

The circuit:
  - 3 index qubits (values 0..7), 1 ancilla output qubit for phase kickback.
  - An oracle X-gate pattern marks exactly the basis states corresponding
    to primes in {0,...,7} (2,3,5,7 in binary: 010,011,101,111) via a
    multi-controlled Z (phase flip) per marked state.
  - Standard Grover diffusion operator, iterated the optimal number of
    times for a 4-out-of-8 search (1 iteration).
  - Run on AerSimulator (ideal, statevector-backed via qasm sampling).

Comparison: after running the circuit, the most frequently measured basis
states are compared against the classically computed set of primes in
{0,...,7}. PASS if they match; FAIL otherwise.
"""

import sys
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime_classical(n: int) -> bool:
    """Trial-division primality test, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_marked_set(n_values: int):
    return sorted(v for v in range(n_values) if is_prime_classical(v))


def build_oracle(qc: QuantumCircuit, index_qubits, ancilla, marked_values, n_bits):
    """Phase-flip oracle: flips the phase of each marked basis state.

    Implemented by, for each marked value, X-ing the qubits that should be 0
    for that value, applying a multi-controlled Z (via H-MCX-H on ancilla),
    then undoing the X's.
    """
    for value in marked_values:
        bits = format(value, f"0{n_bits}b")  # MSB..LSB
        # bits[i] corresponds to index_qubits[n_bits-1-i] (qubit 0 = LSB)
        zero_qubits = [
            index_qubits[n_bits - 1 - i] for i, b in enumerate(bits) if b == "0"
        ]
        if zero_qubits:
            qc.x(zero_qubits)
        qc.h(ancilla)
        qc.mcx(index_qubits, ancilla)
        qc.h(ancilla)
        if zero_qubits:
            qc.x(zero_qubits)


def build_diffuser(qc: QuantumCircuit, index_qubits, ancilla):
    """Standard Grover diffusion operator over the index register."""
    qc.h(index_qubits)
    qc.x(index_qubits)
    qc.h(ancilla)
    qc.mcx(index_qubits, ancilla)
    qc.h(ancilla)
    qc.x(index_qubits)
    qc.h(index_qubits)


def run_grover_prime_search(n_bits: int = 4, shots: int = 4096):
    n_values = 2 ** n_bits
    marked = classical_marked_set(n_values)
    n_marked = len(marked)

    # Optimal number of Grover iterations for n_marked out of n_values.
    theta = np.arcsin(np.sqrt(n_marked / n_values))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    index_qubits = list(range(n_bits))
    ancilla = n_bits
    n_qubits = n_bits + 1

    qc = QuantumCircuit(n_qubits, n_bits)

    # Prepare ancilla in |-> and uniform superposition over index register.
    qc.x(ancilla)
    qc.h(ancilla)
    qc.h(index_qubits)

    for _ in range(iterations):
        build_oracle(qc, index_qubits, ancilla, marked, n_bits)
        build_diffuser(qc, index_qubits, ancilla)

    qc.measure(index_qubits, list(range(n_bits)))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Take the top len(marked) most frequently measured basis states -- the
    # amplitude-amplified (marked) set should dominate the counts.
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    found = sorted(int(bitstring, 2) for bitstring, _ in ranked[: len(marked)])

    return marked, found, counts, iterations


def main():
    n_bits = 4
    marked, found, counts, iterations = run_grover_prime_search(n_bits=n_bits)

    print("Erdos problem #388 -- OEIS: N/A (none associated in source dataset)")
    print("Tag family tested instead (honest fallback): number theory / primality")
    print(f"Grover search over {2 ** n_bits} values, {iterations} iteration(s)")
    print(f"Classical primes in range: {marked}")
    print(f"Quantum-found (high-probability) values: {found}")
    print("Raw counts:", counts)

    verified = found == marked
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
