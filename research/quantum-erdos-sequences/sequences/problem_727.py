"""
Erdos problem #727 -- quantum-testable instance.

OEIS id used: A002503.
  A002503: numbers k such that binomial(2*k, k) is divisible by (k+1)^2.
  (First terms: 5, 14, 27, 41, 44, 65, 76, 90, 109, 125, ...)

Classical property tested:
  For k in {1, 2, ..., 8} (an 8-element search space, indexed by 3 qubits,
  index i <-> k = i + 1), find the unique k with
      C(2k, k) mod (k+1)^2 == 0.
  k = 0 is excluded because it is a trivial degenerate solution
  (binomial(0,0)=1 is divisible by 1^2 for any definition), which is why the
  published sequence itself starts at 5, not 0.

  This script first computes, from first principles (Python's exact integer
  arithmetic, no external number theory library), the value of
  C(2k, k) mod (k+1)^2 for every k in 1..8, and confirms there is exactly one
  k in that range satisfying the property: k = 5 (index 4 in the 0..7 search
  space), matching A002503's first listed term. That gives an unambiguous,
  single-marked-item Grover search instance over 3 qubits.

Quantum approach:
  Grover's algorithm on 3 qubits (search space size N = 8). The oracle marks
  the unique basis state |100> (index 4, i.e. k = 5) with a phase flip,
  implemented directly as a multi-controlled-Z on the bit pattern that
  encodes the classically-precomputed solution index (not by re-deriving
  binomial coefficients inside the circuit -- the oracle target is a fixed,
  classically-verified constant, exactly the kind of "look up a known marked
  item" oracle Grover search is normally run against). One diffusion round
  (optimal for N=8, single solution: floor(pi/4 * sqrt(8)) ~= 2 iterations)
  is applied, then all 3 qubits are measured. The script checks that the
  quantum result (the most frequently measured index) equals the classically
  computed marked index.

PASS/FAIL: prints PASS if the most-frequent measured index equals the
classically derived marked index (5-1=4), FAIL otherwise.
"""

from math import comb, pi, floor, sqrt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_marked_index():
    """Brute-force, from first principles, the unique k in 1..8 with
    C(2k,k) % (k+1)^2 == 0, and return its 0-based search-space index."""
    marked = []
    for k in range(1, 9):
        value = comb(2 * k, k)
        modulus = (k + 1) ** 2
        if value % modulus == 0:
            marked.append(k)
    if len(marked) != 1:
        raise RuntimeError(
            f"Expected exactly one marked k in 1..8 for a single-solution "
            f"Grover instance, found {marked}"
        )
    k = marked[0]
    index = k - 1  # 0-based index into the 8-element search space
    return k, index


def build_grover_circuit(marked_index, n_qubits=3):
    """Standard Grover search over n_qubits qubits with a single marked
    computational basis state given by marked_index (0..2^n_qubits - 1)."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition
    qc.h(range(n_qubits))

    bits = format(marked_index, f"0{n_qubits}b")  # MSB..LSB string

    def apply_oracle(circuit):
        # Flip qubits that should be 0 in the marked pattern, so the
        # controlled-Z fires exactly on |marked_index>.
        for i, b in enumerate(reversed(bits)):  # reversed: qubit 0 = LSB
            if b == "0":
                circuit.x(i)
        if n_qubits == 1:
            circuit.z(0)
        else:
            circuit.h(n_qubits - 1)
            circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            circuit.h(n_qubits - 1)
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                circuit.x(i)

    def apply_diffuser(circuit):
        circuit.h(range(n_qubits))
        circuit.x(range(n_qubits))
        if n_qubits == 1:
            circuit.z(0)
        else:
            circuit.h(n_qubits - 1)
            circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            circuit.h(n_qubits - 1)
        circuit.x(range(n_qubits))
        circuit.h(range(n_qubits))

    n_iterations = max(1, floor((pi / 4) * sqrt(2 ** n_qubits)))
    for _ in range(n_iterations):
        apply_oracle(qc)
        apply_diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, n_iterations


def run_grover(marked_index, n_qubits=3, shots=2048):
    qc, n_iterations = build_grover_circuit(marked_index, n_qubits)
    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()
    # Qiskit bitstrings are little-endian in the classical register string
    # (c[n-1]...c[0]); convert back to an integer index consistently.
    best_bits = max(counts, key=counts.get)
    best_index = int(best_bits, 2)
    return best_index, counts, n_iterations


def main():
    k, classical_index = classical_marked_index()
    print(
        f"Classical result: unique k in 1..8 with C(2k,k) % (k+1)^2 == 0 "
        f"is k = {k} (search-space index {classical_index})"
    )

    quantum_index, counts, n_iterations = run_grover(classical_index)
    print(f"Grover iterations used: {n_iterations}")
    print(f"Measurement counts: {counts}")
    print(f"Quantum result (most frequent measured index): {quantum_index}")

    if quantum_index == classical_index:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
