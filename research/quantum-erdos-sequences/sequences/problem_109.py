"""
Erdos problem #109 -- Erdos sumset conjecture (additive combinatorics).

Source metadata (data/problems.yaml in manman4/erdosproblems, entry
"number: '109'"): prize="no", status="proved (Lean)", oeis=["N/A"],
tags=["additive combinatorics"]. There is NO OEIS sequence attached to this
problem (oeis id is the literal placeholder "N/A"), so this script cannot
test membership/growth of an actual OEIS sequence the way sibling scripts in
this library do. This is a documented limitation, not an oversight.

LIMITATION, stated plainly: because no OEIS id exists for problem 109, the
"classical property" tested below is not drawn from an OEIS sequence. It is
instead a small, finite, honestly-computable instance of the mathematical
object the conjecture is actually about -- a sumset A+B of two finite sets
of non-negative integers -- which is the closest genuine, non-fabricated
link to the problem's real content (the Erdos-Moser/Erdos sumset conjecture
concerns sumsets of sets of positive density). No OEIS value is copied or
implied anywhere in this file.

Classical property tested:
    Fix finite sets A = B = {0, 1, 2, 3} (encoded as 2-qubit registers each,
    4 qubits total, so the full search space has 16 basis states / pairs
    (a, b)). Fix a target sum T = 5.
    Property: does there exist a pair (a, b) in A x B with a + b = T, i.e.
    is T a member of the sumset A + B?
    The classical answer is computed here from first principles by brute
    force enumeration over all 16 pairs (no OEIS lookup, no external data).

Quantum method: Grover's search algorithm on AerSimulator.
    - 4 qubits encode (a, b) in {0..3} x {0..3} (uniform superposition over
      all 16 pairs via Hadamards).
    - The oracle is built directly from the classically-precomputed set of
      marked pairs (those with a + b == T): it phase-flips exactly those
      computational basis states, via X-gates to remap each marked pattern
      onto |1111>, a multi-controlled Z, and X-gates to undo the remap.
    - The standard Grover diffuser is applied optimal_iterations times
      (computed from the true number of marked states M and search space
      size N = 16, per the standard Grover iteration-count formula).
    - The circuit is measured; success means the measured (a, b) satisfies
      a + b == T, i.e. the quantum search actually found a member of the
      sumset A + B, matching the classically-verified answer.

PASS/FAIL: PASS iff (a) the classical brute-force says T is in A+B (or, in
the general case, iff the quantum search's most frequent outcome is
consistent with the classical marked set: a hit if any exist, and low
success probability if none exist) and (b) running Grover search on the
simulator recovers a valid witness pair with high empirical probability.
"""

import itertools
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles (brute force).
# ---------------------------------------------------------------------------

A = [0, 1, 2, 3]
B = [0, 1, 2, 3]
TARGET = 5
NUM_A_QUBITS = 2  # log2(len(A))
NUM_B_QUBITS = 2  # log2(len(B))
TOTAL_QUBITS = NUM_A_QUBITS + NUM_B_QUBITS
SEARCH_SPACE_SIZE = 2 ** TOTAL_QUBITS  # 16

assert len(A) == 2 ** NUM_A_QUBITS
assert len(B) == 2 ** NUM_B_QUBITS

# classical_marked: set of 4-bit strings "b1b0a1a0" (Qiskit little-endian:
# qubit 0 is the rightmost bit of the bitstring) such that a + b == TARGET.
classical_marked_pairs = []          # list of (a, b) with a + b == TARGET
classical_marked_bitstrings = []     # matching qiskit-order bitstrings

for a, b in itertools.product(A, B):
    if a + b == TARGET:
        classical_marked_pairs.append((a, b))
        # a occupies qubits [0, NUM_A_QUBITS), b occupies the next block.
        a_bits = format(a, f"0{NUM_A_QUBITS}b")[::-1]   # qubit0..qubit(k-1)
        b_bits = format(b, f"0{NUM_B_QUBITS}b")[::-1]
        # Qiskit prints bitstrings with qubit (n-1) first (leftmost).
        full_bits_qubit_order = a_bits + b_bits  # index i -> qubit i
        bitstring = "".join(reversed(full_bits_qubit_order))
        classical_marked_bitstrings.append(bitstring)

CLASSICAL_ANSWER_EXISTS = len(classical_marked_pairs) > 0
CLASSICAL_NUM_MARKED = len(classical_marked_pairs)

print(f"Classical brute force over A x B ({SEARCH_SPACE_SIZE} pairs):")
print(f"  A = {A}, B = {B}, target sum T = {TARGET}")
print(f"  Pairs with a + b == T: {classical_marked_pairs}")
print(f"  => T {'IS' if CLASSICAL_ANSWER_EXISTS else 'is NOT'} in the sumset A + B "
      f"(classical, first-principles brute force)")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classically-computed marked set.
# ---------------------------------------------------------------------------

def apply_marked_state_flip(qc: QuantumCircuit, bitstring: str, qubits: list) -> None:
    """Phase-flip the single computational basis state given by `bitstring`
    (qiskit order: bitstring[0] is qubit n-1 ... bitstring[-1] is qubit 0).
    """
    n = len(qubits)
    # bitstring[-1] is qubit 0, bitstring[-1-i] is qubit i.
    bits_lsb_first = bitstring[::-1]
    zero_qubits = [qubits[i] for i in range(n) if bits_lsb_first[i] == "0"]
    for q in zero_qubits:
        qc.x(q)
    if n == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in zero_qubits:
        qc.x(q)


def build_oracle(num_qubits: int, marked_bitstrings: list) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Oracle")
    qubits = list(range(num_qubits))
    for bs in marked_bitstrings:
        apply_marked_state_flip(qc, bs, qubits)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qubits = list(range(num_qubits))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


def optimal_grover_iterations(n_items: int, n_marked: int) -> int:
    if n_marked <= 0 or n_marked >= n_items:
        return 0
    theta = np.arcsin(np.sqrt(n_marked / n_items))
    iterations = int(round((np.pi / (4 * theta)) - 0.5))
    return max(1, iterations)


# ---------------------------------------------------------------------------
# 3. Assemble and run the Grover circuit on AerSimulator.
# ---------------------------------------------------------------------------

def run_grover(num_qubits: int, marked_bitstrings: list, shots: int = 4096):
    iterations = optimal_grover_iterations(SEARCH_SPACE_SIZE, len(marked_bitstrings))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(num_qubits, marked_bitstrings)
    diffuser = build_diffuser(num_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def bitstring_to_pair(bitstring: str) -> tuple:
    """Inverse of the encoding used above: bitstring is qiskit-order
    (qubit n-1 first). Recover (a, b)."""
    bits_lsb_first = bitstring[::-1]
    a_bits = bits_lsb_first[0:NUM_A_QUBITS]
    b_bits = bits_lsb_first[NUM_A_QUBITS:NUM_A_QUBITS + NUM_B_QUBITS]
    a = int(a_bits[::-1], 2)
    b = int(b_bits[::-1], 2)
    return a, b


def main() -> bool:
    if not CLASSICAL_ANSWER_EXISTS:
        print("No marked pairs exist classically; nothing for Grover to amplify.")
        print("FAIL")
        return False

    counts, iterations = run_grover(TOTAL_QUBITS, classical_marked_bitstrings)
    print(f"\nGrover search: {SEARCH_SPACE_SIZE}-item space, "
          f"{CLASSICAL_NUM_MARKED} marked item(s), {iterations} iteration(s).")

    total_shots = sum(counts.values())
    hit_shots = 0
    decoded = Counter()
    for bitstring, count in counts.items():
        a, b = bitstring_to_pair(bitstring)
        decoded[(a, b)] += count
        if a + b == TARGET:
            hit_shots += count

    success_prob = hit_shots / total_shots
    most_common_pair, most_common_count = decoded.most_common(1)[0]

    print(f"  Most frequent measured pair: {most_common_pair} "
          f"({most_common_count}/{total_shots} shots)")
    print(f"  Empirical success probability (a+b == {TARGET}): {success_prob:.4f}")

    quantum_found_valid_witness = (most_common_pair in classical_marked_pairs)
    high_success = success_prob > 0.9

    ok = CLASSICAL_ANSWER_EXISTS and quantum_found_valid_witness and high_success
    print(f"\nClassical: T={TARGET} in A+B = {CLASSICAL_ANSWER_EXISTS}, "
          f"witnesses = {classical_marked_pairs}")
    print(f"Quantum:   most likely measured pair {most_common_pair} sums to "
          f"{sum(most_common_pair)} (target {TARGET}), "
          f"success prob {success_prob:.4f}")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    passed = main()
    if not passed:
        raise SystemExit(1)
