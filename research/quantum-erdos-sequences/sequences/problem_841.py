"""
Erdos problem #841  (data/problems.yaml entry: number "841", oeis: ["A092487"])

OEIS id used: A092487.
A092487(n) is (per the OEIS description of the sequence, tied to Richard K.
Guy's "Unsolved Problems in Number Theory", problem B30) the least k such
that the set {n+1, n+2, ..., n+k} contains a nonempty subset S with

        n * product(S)   a perfect square.

Classical property tested here (derived and checked from first principles
in this script, not copied from OEIS's b-file):

    For a fixed small n, and a fixed k, which of the 2^k nonempty subsets
    S of {n+1, ..., n+k} satisfy "n * product(S) is a perfect square"?

For n = 2, brute force (done classically below, independently of Qiskit)
shows that among the 4-element candidate set {3, 4, 5, 6} (k = 4), exactly
two of the 15 nonempty subsets make n * product(S) a perfect square:

    mask 1001b (elements {3, 6}):  2*3*6  = 36 = 6^2
    mask 1011b (elements {3,4,6}): 2*3*4*6 = 144 = 12^2

(mask bit i, i=0..3, selects element elems[i] from [3,4,5,6]; mask 0 is
excluded as it is the empty subset, product = n, not what's searched for
as a nontrivial witness set here since n itself need not be a square).

This is exactly a Grover search instance: 4 qubits encode which of the 16
subsets of {3,4,5,6} is chosen, and the "good" states are the 2 subsets
(masks 9 and 11) whose product with n=2 is a perfect square. We build a
real Grover oracle (marking those two computational basis states with a
multi-controlled phase flip) plus the standard diffusion operator, run it
on the ideal AerSimulator, and check that measurement overwhelmingly
returns one of the two classically-verified good bitstrings.

Whether n=2 itself is the "informal answer" for problem #841's underlying
Guy conjecture is not what is being claimed here; what is verified is the
concrete, independently-checked classical fact above, and that Grover's
algorithm quantum-mechanically finds it.
"""

from math import isqrt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def is_square(x: int) -> bool:
    if x < 0:
        return False
    r = isqrt(x)
    return r * r == x


def classical_marked_masks(n: int, elems: list[int]) -> list[int]:
    """All nonempty subset masks of `elems` (indices 0..k-1) with
    n * product(subset) a perfect square. Pure classical computation,
    independent of anything below."""
    k = len(elems)
    marked = []
    for mask in range(1, 1 << k):
        prod = n
        for i in range(k):
            if mask & (1 << i):
                prod *= elems[i]
        if is_square(prod):
            marked.append(mask)
    return marked


def build_oracle(k: int, marked_masks: list[int]) -> QuantumCircuit:
    """Phase-flip oracle: multiply the amplitude of each marked
    computational basis state (bitstring q_{k-1}...q_0 read as `mask`)
    by -1, using a multi-controlled Z built from X gates + mcx-with-phase
    (implemented via a controlled-Z on an ancilla-free multi-controlled
    Z gate)."""
    qc = QuantumCircuit(k, name="oracle")
    for mask in marked_masks:
        flip_qubits = [i for i in range(k) if not (mask & (1 << i))]
        for q in flip_qubits:
            qc.x(q)
        if k == 1:
            qc.z(0)
        else:
            qc.h(k - 1)
            qc.mcx(list(range(k - 1)), k - 1)
            qc.h(k - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc


def build_diffuser(k: int) -> QuantumCircuit:
    qc = QuantumCircuit(k, name="diffuser")
    qc.h(range(k))
    qc.x(range(k))
    if k == 1:
        qc.z(0)
    else:
        qc.h(k - 1)
        qc.mcx(list(range(k - 1)), k - 1)
        qc.h(k - 1)
    qc.x(range(k))
    qc.h(range(k))
    return qc


def grover_find_marked(k: int, marked_masks: list[int], shots: int = 4096):
    import math

    num_marked = len(marked_masks)
    iterations = max(1, round((math.pi / 4) * math.sqrt((2**k) / num_marked)))

    oracle = build_oracle(k, marked_masks)
    diffuser = build_diffuser(k)

    qc = QuantumCircuit(k, k)
    qc.h(range(k))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(k), range(k))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n = 2
    elems = [3, 4, 5, 6]  # n+1 .. n+4, i.e. k = 4
    k = len(elems)

    marked_masks = classical_marked_masks(n, elems)
    print(f"n = {n}, candidate set (elems for n+1..n+{k}) = {elems}")
    print(f"Classical marked subset masks (n*product is a perfect square): {marked_masks}")
    for m in marked_masks:
        subset = [elems[i] for i in range(k) if m & (1 << i)]
        prod = n
        for v in subset:
            prod *= v
        root = isqrt(prod)
        print(f"  mask={m:0{k}b}  subset={subset}  n*product={prod} = {root}^2")

    assert marked_masks, "classical search found no witness subset; cannot build Grover oracle"

    counts, iterations = grover_find_marked(k, marked_masks, shots=4096)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    # Qiskit bit ordering: rightmost measured classical bit is qubit 0.
    def bitstring_to_mask(bs: str) -> int:
        bits = bs[::-1]  # bits[i] corresponds to qubit i
        mask = 0
        for i, b in enumerate(bits):
            if b == "1":
                mask |= (1 << i)
        return mask

    most_common_bs = max(counts, key=counts.get)
    most_common_mask = bitstring_to_mask(most_common_bs)

    marked_shots = sum(c for bs, c in counts.items() if bitstring_to_mask(bs) in marked_masks)
    total_shots = sum(counts.values())
    marked_fraction = marked_shots / total_shots

    print(f"Most frequent measured mask: {most_common_mask:0{k}b} (= {most_common_mask})")
    print(f"Fraction of shots landing on a classically-verified marked state: {marked_fraction:.3f}")

    quantum_found_marked_state = most_common_mask in marked_masks
    amplification_succeeded = marked_fraction > 0.7  # random guessing would give ~2/16 = 0.125

    passed = quantum_found_marked_state and amplification_succeeded

    print(f"Quantum result matches a classically verified witness subset: {quantum_found_marked_state}")
    print(f"Grover amplification clearly above random baseline (0.125): {amplification_succeeded}")
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
