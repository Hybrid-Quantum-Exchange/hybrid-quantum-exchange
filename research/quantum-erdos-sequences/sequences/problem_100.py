"""
Erdos problem #100 (erdosproblems.com), quantum-testable lane.

Source metadata (from manman4/erdosproblems data/problems.yaml, entry
`number: "100"`): prize "no", status "open", tags ["geometry", "distances"],
oeis: ["N/A"]. There is NO OEIS sequence id attached to this problem — the
listed value is the literal string "N/A". Per the task instructions, an
honest attempt is written here rather than a fabricated sequence property,
and the limitation is noted explicitly: this script does NOT test any real
term membership of an actual Erdos-problem-100 OEIS sequence, because none
exists in the source data.

Given the problem's tags ("geometry", "distances"), the best small,
finite, computable stand-in property genuinely in that spirit is a basic
distance-search problem on integer points on a line:

    Property tested: for points 0..7 on the integer line (3-bit index,
    N = 8), find a pair (i, j) with i < j whose distance |i - j| equals a
    fixed target distance d = 3. This is a real combinatorial "distance"
    search problem (finding points realizing a prescribed pairwise
    distance, the flavor of Erdos distance-type questions), on a small
    finite search space (all C(8,2) = 28 pairs, encoded as 6-bit strings:
    3 bits for i, 3 bits for j).

Classical answer (computed here in the script, from first principles):
enumerate all pairs (i, j) with 0 <= i < j <= 7 and |i - j| == 3, listing
every marked pair. This is used both to build the Grover oracle and as the
ground truth the quantum result is checked against.

Quantum approach: Grover's search algorithm on 6 qubits (3 for i, 3 for j),
oracle marks exactly the basis states encoding an ordered pair (i, j) with
i < j and j - i == 3, diffusion operator amplifies those marked amplitudes,
and the ideal AerSimulator statevector/counts are checked against the
classical enumeration: the most-probable measured outcomes must be exactly
the classically-marked pairs, and each marked pair's probability must
exceed all unmarked pairs' probabilities.

Limitation: this is a faithful small Grover search over a distance
property, not a test of a specific documented OEIS integer sequence
(none is listed for problem #100). ran_ok and verified_against_classical
are reported honestly for what this script actually checks.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_BITS_PER_COORD = 3          # i, j in 0..7
N = 1 << N_BITS_PER_COORD      # 8
TARGET_DISTANCE = 3


def classical_marked_pairs(n: int, target_d: int):
    """Enumerate all (i, j), i < j <= n-1, with j - i == target_d."""
    marked = []
    for i, j in itertools.combinations(range(n), 2):
        if (j - i) == target_d:
            marked.append((i, j))
    return marked


def pair_to_bitstring(i: int, j: int, bits_per_coord: int) -> str:
    """Qiskit bit ordering: qubit 0 is the rightmost character.

    Layout: qubits [0:bits_per_coord) encode i (LSB first),
    qubits [bits_per_coord:2*bits_per_coord) encode j (LSB first).
    """
    i_bits = format(i, f"0{bits_per_coord}b")[::-1]
    j_bits = format(j, f"0{bits_per_coord}b")[::-1]
    full = j_bits + i_bits  # qubit index increases left->right in this string
    # Build the standard big-endian-printed bitstring Qiskit counts use:
    # counts keys are printed MSB(last qubit) ... LSB(qubit 0), so reverse.
    return full[::-1]


def build_oracle_phase(bits_per_coord: int, marked_pairs, total_qubits: int) -> QuantumCircuit:
    """Correct multi-controlled-Z oracle: applies -1 phase to each marked basis state."""
    qc = QuantumCircuit(total_qubits, name="oracle")
    for (i, j) in marked_pairs:
        i_bits = format(i, f"0{bits_per_coord}b")[::-1]
        j_bits = format(j, f"0{bits_per_coord}b")[::-1]
        pattern = list(i_bits) + list(j_bits)

        zero_qubits = [q for q, b in enumerate(pattern) if b == "0"]
        for q in zero_qubits:
            qc.x(q)

        # Multi-controlled Z on all qubits: flips phase of |11...1>
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


def run():
    bits_per_coord = N_BITS_PER_COORD
    total_qubits = 2 * bits_per_coord  # 6 qubits: encode (i, j) directly, no ordering constraint bit

    marked_pairs = classical_marked_pairs(N, TARGET_DISTANCE)
    assert marked_pairs, "classical search found no marked pairs — instance is degenerate"
    print(f"Classical answer: pairs (i,j) in 0..{N-1} with j-i == {TARGET_DISTANCE}:")
    for p in marked_pairs:
        print(f"  {p}")

    num_marked = len(marked_pairs)
    search_space_size = N * N  # we search over all ordered (i,j) 6-bit strings; marked = ordered pairs with j-i==d
    # Redefine marked set over the ordered-pair search space actually used by the oracle:
    ordered_marked = [(i, j) for i in range(N) for j in range(N) if (j - i) == TARGET_DISTANCE]
    assert set(ordered_marked) == set(marked_pairs), "ordered marked set must match classical i<j pairs"

    theta = math.asin(math.sqrt(num_marked / search_space_size))
    optimal_iters = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle_phase(bits_per_coord, ordered_marked, total_qubits)
    diffuser = build_diffuser(total_qubits)

    qc = QuantumCircuit(total_qubits, total_qubits)
    qc.h(range(total_qubits))
    for _ in range(optimal_iters):
        qc.append(oracle.to_instruction(), range(total_qubits))
        qc.append(diffuser.to_instruction(), range(total_qubits))
    qc.measure(range(total_qubits), range(total_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4096, seed_simulator=42).result()
    counts = result.get_counts()

    def bitstring_to_pair(bs: str):
        # counts keys: qubit(total-1) ... qubit0, left to right
        bits = bs[::-1]  # now index 0 == qubit0
        i_bits = bits[0:bits_per_coord]
        j_bits = bits[bits_per_coord:2 * bits_per_coord]
        i_val = int(i_bits[::-1], 2)
        j_val = int(j_bits[::-1], 2)
        return (i_val, j_val)

    pair_counts = {}
    for bs, c in counts.items():
        pair = bitstring_to_pair(bs)
        pair_counts[pair] = pair_counts.get(pair, 0) + c

    sorted_pairs = sorted(pair_counts.items(), key=lambda kv: -kv[1])
    print("\nTop measured (i,j) pairs by frequency:")
    for pair, c in sorted_pairs[:8]:
        marker = " <-- MARKED" if pair in set(ordered_marked) else ""
        print(f"  {pair}: {c}{marker}")

    top_n = sorted_pairs[:num_marked]
    top_pairs = set(p for p, _ in top_n)
    marked_set = set(ordered_marked)

    all_marked_on_top = top_pairs == marked_set
    min_marked_count = min(pair_counts.get(p, 0) for p in marked_set)
    max_unmarked_count = max(
        (c for p, c in pair_counts.items() if p not in marked_set), default=0
    )
    dominance_ok = min_marked_count > max_unmarked_count

    verified = all_marked_on_top and dominance_ok

    print(f"\nGrover iterations used: {optimal_iters}")
    print(f"Marked pairs (ordered search space): {sorted(marked_set)}")
    print(f"Top-{num_marked} measured pairs match marked set exactly: {all_marked_on_top}")
    print(f"Every marked pair outcount every unmarked pair: {dominance_ok}")

    if verified:
        print("\nPASS")
    else:
        print("\nFAIL")

    return verified


if __name__ == "__main__":
    ok = run()
    if not ok:
        raise SystemExit(1)
