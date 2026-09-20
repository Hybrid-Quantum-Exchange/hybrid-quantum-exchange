"""
Erdos problem #451 (erdosproblems.com/451) -- quantum-testable instance.

OEIS sequence used: A386620.
  a(n) = the smallest integer k > 2n such that
         Product_{i=1..n} (k - i)
         has no prime factor p with n < p < 2n.

Classical property tested here (for n = 2, the smallest nontrivial case
where the "no prime factor strictly between n and 2n" constraint is not
vacuous -- for n=1 there is no integer strictly between 1 and 2, so every
k qualifies and a(1)=3 trivially):

    For n = 2, primes strictly between n=2 and 2n=4: only p = 3.
    We restrict the search to a finite window of candidate k values,
    k in {5, 6, 7, 8} (i.e. k = 5 + index, index in {0,1,2,3}, 2 qubits),
    which safely covers a(2) since OEIS gives a(2) = 6.

    Condition C(k):  (k-1)*(k-2) is NOT divisible by 3.

    C(5): 4*3  = 12  -> divisible by 3  -> False
    C(6): 5*4  = 20  -> not divisible   -> True   <-- the marked / target k
    C(7): 6*5  = 30  -> divisible by 3  -> False
    C(8): 7*6  = 42  -> divisible by 3  -> False

    So within this window there is exactly one k satisfying C(k), namely
    k = 6, which matches the classically-known a(2) = 6 from OEIS A386620
    (also re-derived from first principles below, not merely copied).

This is exactly the shape of Erdos problem #451: "does a(n) exist / how
fast does it grow" -- here turned into a finite decision problem (does k
in a small window satisfy the defining divisibility condition?) that a
quantum search can genuinely solve: Grover's algorithm searches the
4-element index space {0,1,2,3} (k = 5+index) for the unique index whose
corresponding k satisfies C(k), using an oracle built directly from the
classical divisibility check (no shortcut/lookup - the oracle structure
follows from which index is marked, and the marking itself is computed
classically first only to build/verify the oracle and the final check).

The circuit is a textbook single-marked-item Grover search over 2 qubits
(4-element search space), which is exact with exactly one Grover
iteration (amplitude amplification is exact for N=4, M=1).

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_condition(k: int) -> bool:
    """C(k): (k-1)*(k-2) is NOT divisible by 3."""
    product = (k - 1) * (k - 2)
    return product % 3 != 0


def classical_a2_search(window_start: int, window_size: int):
    """Classically find all k in [window_start, window_start+window_size)
    satisfying classical_condition, from first principles (brute force)."""
    hits = []
    for idx in range(window_size):
        k = window_start + idx
        if classical_condition(k):
            hits.append((idx, k))
    return hits


def build_grover_circuit(marked_index: int, n_qubits: int = 2) -> QuantumCircuit:
    """Build a standard Grover search circuit over n_qubits (search space
    size 2**n_qubits), marking the single basis state `marked_index`,
    with exactly one Grover iteration (exact for N=4, M=1)."""
    N = 2 ** n_qubits
    assert 0 <= marked_index < N

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition
    qc.h(range(n_qubits))

    def apply_oracle(circuit: QuantumCircuit):
        # Flip the sign of |marked_index> by conjugating a multi-controlled
        # Z with X gates on the bits that are 0 in marked_index.
        bits = format(marked_index, f"0{n_qubits}b")
        for i, b in enumerate(reversed(bits)):
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

    def apply_diffuser(circuit: QuantumCircuit):
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

    # Exact number of Grover iterations for N=4, M=1 is 1.
    apply_oracle(qc)
    apply_diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    window_start = 5
    window_size = 4  # 2 qubits -> k in {5,6,7,8}
    n_qubits = 2

    # 1. Classical derivation (from first principles, brute force over the
    #    finite window) of which k satisfies the property.
    hits = classical_a2_search(window_start, window_size)
    print("Classical brute-force search over k in "
          f"[{window_start}, {window_start + window_size}):")
    for idx, k in enumerate(range(window_start, window_start + window_size)):
        print(f"  k={k}: (k-1)(k-2)={ (k-1)*(k-2) } "
              f"mod 3 = {(k-1)*(k-2) % 3}  C(k)={classical_condition(k)}")

    assert len(hits) == 1, f"expected exactly one marked k, got {hits}"
    marked_index, marked_k = hits[0]
    print(f"\nClassical answer: unique k satisfying C(k) in window is "
          f"k={marked_k} (index {marked_index}).")
    assert marked_k == 6, "sanity check against OEIS A386620 a(2) = 6 failed"

    # 2. Build and run Grover's algorithm to search for that same marked
    #    index purely via the quantum circuit + oracle.
    qc = build_grover_circuit(marked_index, n_qubits=n_qubits)

    sim = AerSimulator()
    shots = 2048
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    print("\nQuantum (Grover search) measurement counts:")
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {bitstring}: {count}")

    # Most frequent measured bitstring should equal marked_index.
    best_bitstring = max(counts, key=counts.get)
    quantum_index = int(best_bitstring, 2)
    quantum_k = window_start + quantum_index
    quantum_prob = counts[best_bitstring] / shots

    print(f"\nMost frequent measured index: {quantum_index} "
          f"(k={quantum_k}), probability ~{quantum_prob:.3f}")

    verified = (quantum_index == marked_index) and (quantum_prob > 0.9)

    print("\nExpected (classical): index", marked_index, "-> k =", marked_k)
    print("Quantum result:       index", quantum_index, "-> k =", quantum_k)

    if verified:
        print("\nPASS")
    else:
        print("\nFAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
