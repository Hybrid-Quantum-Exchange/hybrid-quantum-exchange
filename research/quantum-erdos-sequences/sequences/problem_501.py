"""
Erdos problem #501 (erdosproblems.com), quantum-testable lane.

Source metadata (from manman4/erdosproblems data/problems.yaml, entry
`number: "501"`): prize "no", status "not disprovable", tags
["combinatorics", "set theory"], oeis: ["N/A"]. There is NO OEIS sequence id
attached to this problem -- the listed value is the literal string "N/A".
Per the task instructions, an honest attempt is written here rather than a
fabricated sequence property, and the limitation is stated explicitly: this
script does NOT test membership in any real, documented OEIS sequence tied
to Erdos problem 501, because no such sequence id exists in the source data.

Given the problem's tags ("combinatorics", "set theory"), the best small,
finite, computable stand-in genuinely in that spirit is a subset-sum search,
a canonical combinatorics/set-theory problem (does there exist a subset of a
given finite set whose elements sum to a target value?):

    Property tested: let S = {1, 2, 3, 4, 5, 6} (6 elements, so subsets are
    encoded by 6 bits, one "in/out" bit per element -- N = 2^6 = 64 subsets,
    matching the "N <= ~64" instance-size budget). Fix TARGET_SUM = 10.
    Which subsets of S sum to exactly 10?

Classical answer (computed here in the script, from first principles, by
brute-force enumeration of all 64 subsets): every subset of {1,...,6} whose
elements sum to 10 is enumerated directly with Python; this becomes both the
Grover oracle's marked set and the ground truth the quantum result is
checked against.

Quantum approach: Grover's search algorithm on 6 qubits, one qubit per set
element (bit = 1 means "element included"). The oracle applies a -1 phase to
every one of the classically-enumerated marked basis states (subsets summing
to 10); the diffusion operator amplifies those marked amplitudes. Run on the
ideal AerSimulator; the measured outcome distribution is checked against the
classical enumeration: the most-probable measured outcomes must be exactly
the classically-marked subsets, and each marked subset's count must exceed
every unmarked subset's count.

Limitation: this is a faithful small Grover subset-sum search in the
combinatorics/set-theory spirit of problem 501's tags, not a test of a
specific documented OEIS integer sequence (none is listed for problem #501).
ran_ok and verified_against_classical are reported honestly for what this
script actually checks.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


ELEMENTS = [1, 2, 3, 4, 5, 6]   # S = {1,...,6}, 6 elements -> 6 qubits, N = 64
N_QUBITS = len(ELEMENTS)
TARGET_SUM = 10


def classical_marked_subsets(elements, target_sum):
    """Brute-force enumerate every subset of `elements` summing to target_sum.

    Returns a list of frozensets of *indices* into `elements` (the qubit
    positions that are 1 for that subset), which doubles as the bitmask.
    """
    marked = []
    n = len(elements)
    for mask in range(1 << n):
        total = sum(elements[i] for i in range(n) if (mask >> i) & 1)
        if total == target_sum:
            marked.append(mask)
    return marked


def build_oracle_phase(marked_masks, total_qubits: int) -> QuantumCircuit:
    """Multi-controlled-Z oracle: applies -1 phase to each marked basis state.

    Bit i of `mask` (LSB first) corresponds to qubit i.
    """
    qc = QuantumCircuit(total_qubits, name="oracle")
    for mask in marked_masks:
        zero_qubits = [q for q in range(total_qubits) if not ((mask >> q) & 1)]
        for q in zero_qubits:
            qc.x(q)

        qc.h(total_qubits - 1)
        if total_qubits - 1 > 0:
            qc.append(MCXGate(total_qubits - 1), list(range(total_qubits - 1)) + [total_qubits - 1])
        else:
            qc.z(0)
        qc.h(total_qubits - 1)

        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(total_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(total_qubits, name="diffuser")
    qc.h(range(total_qubits))
    qc.x(range(total_qubits))
    qc.h(total_qubits - 1)
    if total_qubits - 1 > 0:
        qc.append(MCXGate(total_qubits - 1), list(range(total_qubits - 1)) + [total_qubits - 1])
    else:
        qc.z(0)
    qc.h(total_qubits - 1)
    qc.x(range(total_qubits))
    qc.h(range(total_qubits))
    return qc


def mask_to_subset(mask: int):
    return tuple(ELEMENTS[i] for i in range(N_QUBITS) if (mask >> i) & 1)


def run():
    marked_masks = classical_marked_subsets(ELEMENTS, TARGET_SUM)
    assert marked_masks, "classical search found no marked subsets -- instance is degenerate"

    print(f"S = {ELEMENTS}, TARGET_SUM = {TARGET_SUM}")
    print("Classical answer: subsets of S summing to TARGET_SUM:")
    for m in marked_masks:
        print(f"  mask={m:06b}  subset={mask_to_subset(m)}")

    num_marked = len(marked_masks)
    search_space_size = 1 << N_QUBITS  # 64

    theta = math.asin(math.sqrt(num_marked / search_space_size))
    optimal_iters = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle_phase(marked_masks, N_QUBITS)
    diffuser = build_diffuser(N_QUBITS)

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    for _ in range(optimal_iters):
        qc.append(oracle.to_instruction(), range(N_QUBITS))
        qc.append(diffuser.to_instruction(), range(N_QUBITS))
    qc.measure(range(N_QUBITS), range(N_QUBITS))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4096, seed_simulator=42).result()
    counts = result.get_counts()

    def bitstring_to_mask(bs: str) -> int:
        # Qiskit counts keys: qubit(n-1) ... qubit0, left to right.
        bits = bs[::-1]  # now index i == qubit i
        mask = 0
        for i, b in enumerate(bits):
            if b == "1":
                mask |= (1 << i)
        return mask

    mask_counts = {}
    for bs, c in counts.items():
        mask = bitstring_to_mask(bs)
        mask_counts[mask] = mask_counts.get(mask, 0) + c

    sorted_masks = sorted(mask_counts.items(), key=lambda kv: -kv[1])
    print("\nTop measured subsets by frequency:")
    marked_set = set(marked_masks)
    for mask, c in sorted_masks[:8]:
        marker = " <-- MARKED" if mask in marked_set else ""
        print(f"  {mask_to_subset(mask)}: {c}{marker}")

    top_n = sorted_masks[:num_marked]
    top_masks = set(m for m, _ in top_n)

    all_marked_on_top = top_masks == marked_set
    min_marked_count = min(mask_counts.get(m, 0) for m in marked_set)
    max_unmarked_count = max(
        (c for m, c in mask_counts.items() if m not in marked_set), default=0
    )
    dominance_ok = min_marked_count > max_unmarked_count

    verified = all_marked_on_top and dominance_ok

    print(f"\nGrover iterations used: {optimal_iters}")
    print(f"Marked subsets (masks): {sorted(marked_set)}")
    print(f"Top-{num_marked} measured subsets match marked set exactly: {all_marked_on_top}")
    print(f"Every marked subset outcounts every unmarked subset: {dominance_ok}")

    if verified:
        print("\nPASS")
    else:
        print("\nFAIL")

    return verified


if __name__ == "__main__":
    ok = run()
    if not ok:
        raise SystemExit(1)
