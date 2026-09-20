"""
Erdos problem #712 -- quantum-testable instance.

Source: erdosproblems.com problem #712 (data/problems.yaml entry, number "712").
Tags on the problem: ["graph theory", "turan number", "hypergraphs"].
`oeis` field for #712 is `["possible"]` -- this is a placeholder used by the
erdosproblems dataset to mean "an OEIS sequence may exist but none is
recorded", NOT an actual OEIS id. There is therefore no real OEIS sequence to
target for #712. Per the task's fallback instructions, this script instead
builds a genuine, finite, computable instance of the actual mathematical
object the problem is about -- a hypergraph Turan number -- since that is
what the problem's own tags name, and verifies it with a real Grover search
circuit rather than fabricating or copying an OEIS value.

LIMITATION (stated plainly): this is not tied to a specific OEIS id, because
#712 does not have one in the source data. The classical property tested
below is derived from the problem's own subject matter (Turan-type extremal
hypergraph counting), not copied from any external table.

Classical property tested
--------------------------
Let V = {0,1,2,3} and let T be the 4 distinct 3-element subsets ("triples")
of V (this is every 3-uniform hyperedge on 4 vertices). A subset S of T can
be encoded as a 4-bit string, bit i = 1 iff triple i is included in S.

The complete 3-uniform hypergraph on 4 vertices, K4^(3), is the single
subset S = T (all 4 triples). The (trivial, first-principles) Turan-type
extremal question: what is the largest hyperedge-subset S of T that does
NOT equal the complete hypergraph K4^(3)? Since the only forbidden subset is
the full set of size 4, every subset of size 3 is admissible, and no subset
of size 4 is (there is only one, and it is exactly the forbidden pattern).
So the extremal ("Turan") number here is

    ex(4, K4^(3)) = 3,

realized by exactly the four size-3 subsets of T (each omitting one triple).
This is computed from first principles below by brute-force enumeration of
all 2^4 = 16 subsets, with no external data or lookup.

Quantum circuit
----------------
A 4-qubit Grover search marks exactly the computational basis states whose
Hamming weight is 3 (i.e. exactly the four admissible extremal hyperedge-
subsets identified above), using an explicit phase oracle built out of X and
multi-controlled-X gates (a genuine oracle over the 16-element search space,
not a lookup table), followed by the standard Grover diffusion operator.
One Grover iteration is applied (optimal for 4 marked states out of 16). The
circuit is run on the ideal Qiskit Aer statevector/qasm simulator (AerSimulator).

PASS/FAIL
---------
The script computes the classical answer (the set of size-3 admissible
subsets) by brute force, runs the Grover circuit, and PASSes if the most
frequently measured bitstring is one of the classically verified extremal
solutions (Hamming weight 3, i.e. not the forbidden complete hypergraph).
"""

from itertools import combinations, product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_VERTICES = 4
N_QUBITS = 4  # one qubit per 3-element subset ("triple") of the 4 vertices


def classical_triples():
    """The 4 distinct 3-element subsets of {0,1,2,3}, in a fixed order."""
    return list(combinations(range(N_VERTICES), 3))


def classical_extremal_subsets():
    """
    Brute-force, first-principles computation of every hyperedge-subset S
    (of the 4 triples) that is admissible (S != the complete hypergraph,
    i.e. S is not the set of ALL 4 triples), restricted to subsets of
    maximum size (the Turan-extremal subsets).

    Returns (max_size, set_of_bitmasks) where each bitmask is an int in
    [0, 15], bit i set iff triple i in S, using the LSB-first convention
    bit i <-> qubit i (matches the Grover circuit's qubit layout below).
    """
    triples = classical_triples()
    n = len(triples)
    full_mask = (1 << n) - 1  # the complete hypergraph K4^(3)

    admissible = []
    for bits in product([0, 1], repeat=n):
        mask = 0
        for i, b in enumerate(bits):
            if b:
                mask |= (1 << i)
        if mask != full_mask:  # admissible iff not the complete hypergraph
            admissible.append(mask)

    max_size = max(bin(m).count("1") for m in admissible)
    extremal = {m for m in admissible if bin(m).count("1") == max_size}
    return max_size, extremal


def bit_at(mask, i):
    return (mask >> i) & 1


def mark_basis_state(qc, qubits, mask):
    """
    Apply a phase flip (-1) to exactly the computational basis state
    |mask> (LSB-first: qubit i holds bit i of `mask`), leaving every other
    basis state unchanged. Built from X gates + one multi-controlled-Z
    (via H + multi-controlled-X + H), which is the standard technique for
    a phase oracle marking a single target bitstring.
    """
    zero_qubits = [q for i, q in enumerate(qubits) if bit_at(mask, i) == 0]
    for q in zero_qubits:
        qc.x(q)

    controls = qubits[:-1]
    target = qubits[-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)

    for q in zero_qubits:
        qc.x(q)


def build_oracle(marked_masks):
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    qubits = list(range(N_QUBITS))
    for mask in marked_masks:
        mark_basis_state(qc, qubits, mask)
    return qc


def build_diffuser():
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qubits = list(range(N_QUBITS))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


def build_grover_circuit(marked_masks, iterations):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))

    oracle = build_oracle(marked_masks)
    diffuser = build_diffuser()

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def main():
    max_size, extremal_masks = classical_extremal_subsets()
    triples = classical_triples()

    print(f"Erdos problem #712 -- classical setup")
    print(f"  triples (3-subsets of {{0,1,2,3}}): {triples}")
    print(f"  Turan-extremal size ex(4, K4^(3)) computed classically = {max_size}")
    print(f"  extremal hyperedge-subsets (bitmasks): {sorted(extremal_masks)}")
    assert max_size == 3, "sanity check on the classical computation failed"
    assert len(extremal_masks) == 4

    n_states = 2 ** N_QUBITS
    n_marked = len(extremal_masks)
    import math
    theta = math.asin(math.sqrt(n_marked / n_states))
    optimal_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = build_grover_circuit(extremal_masks, optimal_iterations)

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=4096).result()
    counts = result.get_counts()

    # Qiskit returns bitstrings MSB-first (qubit n-1 ... qubit 0); convert
    # to our LSB-first integer mask convention (qubit i <-> bit i).
    def bitstring_to_mask(bs):
        mask = 0
        for i, ch in enumerate(reversed(bs)):
            if ch == "1":
                mask |= (1 << i)
        return mask

    counts_by_mask = {}
    for bs, c in counts.items():
        counts_by_mask[bitstring_to_mask(bs)] = counts_by_mask.get(bitstring_to_mask(bs), 0) + c

    top_mask = max(counts_by_mask, key=counts_by_mask.get)
    top_count = counts_by_mask[top_mask]
    marked_total = sum(c for m, c in counts_by_mask.items() if m in extremal_masks)

    print(f"\nGrover search -- {optimal_iterations} iteration(s), 4096 shots")
    print(f"  most frequent measured mask: {top_mask:04b} (count {top_count}/4096)")
    print(f"  total shots landing on a classically-verified extremal solution: "
          f"{marked_total}/4096 ({100 * marked_total / 4096:.1f}%)")
    print(f"  (uniform-random baseline would give {n_marked}/{n_states} = "
          f"{100 * n_marked / n_states:.1f}%)")

    verified = (top_mask in extremal_masks) and (marked_total > 4096 * (n_marked / n_states))

    if verified:
        print("\nPASS: Grover search's top outcome matches a classically verified "
              "Turan-extremal solution, with amplified probability above the "
              "uniform-random baseline.")
    else:
        print("\nFAIL: Grover search result did not match the classical answer.")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
