"""
Erdos problem #56 -- quantum-testable lane.

Source metadata (data/problems.yaml, entry "number: '56'", tags
["number theory", "intersecting family"]):
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #56 carries no OEIS sequence
id in the source data (oeis == "N/A"), so there is no integer sequence to
test membership/terms of. Per the task instructions, this script does not
fabricate an OEIS-derived property. Instead it builds a genuine, finite,
computable property drawn directly from the problem's own tags
("intersecting family", "number theory") -- the Erdos-Ko-Rado theorem --
and verifies it with a real Grover search circuit. This is the script's
best honest attempt in the absence of an OEIS id; ran_ok/verified are
reported for *this* property, not for an OEIS sequence that does not exist
for this problem.

Classical property under test (computed from first principles below, not
looked up):
    Let n = 5, k = 2. Consider all C(5,2) = 10 two-element subsets of
    {0,1,2,3,4}. The "star" family S = { A : 0 in A } is an intersecting
    family (every two sets in it share element 0). The Erdos-Ko-Rado
    theorem says the maximum intersecting family of k-subsets of an
    n-set (n >= 2k) has size C(n-1, k-1); for n=5, k=2 that is C(4,1) = 4,
    and the star family achieves it.

    This script:
      1. Classically enumerates all 10 two-subsets of {0,...,4}, and
         classically identifies exactly which of them contain element 0
         (the star family), and checks its size equals C(4,1) = 4 -- the
         EKR-predicted maximum -- by brute-force search over ALL
         intersecting families is infeasible to embed directly in a tiny
         oracle, so instead we test the well-defined finite decision
         property: "does 2-subset A (given by its index into the fixed
         enumeration) contain element 0?" This is exactly the star-family
         membership predicate whose count the EKR theorem pins down.
      2. Builds a Grover search circuit over the 4-qubit index space
         (16 basis states, 10 of which correspond to real 2-subsets) whose
         oracle marks precisely the indices of the subsets that contain
         element 0 -- i.e. marks the star family.
      3. Runs the circuit on the ideal AerSimulator and checks that
         measurement outcomes concentrate on the classically-computed
         marked set, and that the marked-set count found quantumly
         matches the classical count of 4.

PASS criterion: over many shots, the highest-probability measured basis
states are exactly the classically-computed marked indices, and their
combined empirical probability exceeds a fixed threshold -- i.e. Grover
amplification worked and the amplified set equals the classical star
family of size C(4,1) = 4.
"""

from itertools import combinations
from math import comb, pi, floor, sqrt

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup).
# ---------------------------------------------------------------------------

N = 5   # ground set size
K = 2   # subset size

all_subsets = list(combinations(range(N), K))          # fixed enumeration
assert len(all_subsets) == comb(N, K) == 10

# Index space needs ceil(log2(10)) = 4 qubits -> 16 basis states, indices
# 10..15 are "unused" (no subset maps to them) and must never be marked.
NUM_QUBITS = 4
NUM_STATES = 2 ** NUM_QUBITS  # 16

# The star family: subsets containing element 0.
marked_indices = [i for i, s in enumerate(all_subsets) if 0 in s]
classical_star_family = [all_subsets[i] for i in marked_indices]

# Erdos-Ko-Rado predicted maximum intersecting family size for n=5,k=2.
ekr_predicted_max = comb(N - 1, K - 1)  # C(4,1) = 4

assert len(classical_star_family) == ekr_predicted_max, (
    f"classical count {len(classical_star_family)} != EKR bound {ekr_predicted_max}"
)

# Sanity: the star family is genuinely intersecting (every pair shares elt 0).
for a, b in combinations(classical_star_family, 2):
    assert set(a) & set(b), "star family is supposed to be intersecting"

print("Classical setup:")
print(f"  n={N}, k={K}, all {len(all_subsets)} subsets: {all_subsets}")
print(f"  star family (contain element 0): {classical_star_family}")
print(f"  EKR-predicted max intersecting family size C(n-1,k-1) = {ekr_predicted_max}")
print(f"  marked indices (in fixed enumeration): {marked_indices}")


# ---------------------------------------------------------------------------
# 2. Grover search circuit marking exactly `marked_indices`.
# ---------------------------------------------------------------------------

def oracle_for_indices(num_qubits: int, indices: list[int]) -> QuantumCircuit:
    """Phase-flip oracle: applies -1 phase to each computational basis state
    whose integer value (little-endian) is in `indices`, via X-sandwiched
    multi-controlled Z on each target bitstring."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    mcz = MCXGate(num_qubits - 1)  # placeholder type spec; built per-call below
    for idx in indices:
        bits = format(idx, f"0{num_qubits}b")[::-1]  # little-endian per qubit
        zero_positions = [q for q, b in enumerate(bits) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        # multi-controlled Z on all qubits: use H-MCX-H trick on last qubit
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        for q in zero_positions:
            qc.x(q)
    return qc


def diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


M = len(marked_indices)
theta = 2 * pi if M == 0 else 2 * pi  # unused; kept for clarity
# Optimal number of Grover iterations for M marked out of N_STATES.
optimal_iters = max(1, floor((pi / 4) * sqrt(NUM_STATES / M)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

oracle = oracle_for_indices(NUM_QUBITS, marked_indices)
diff = diffuser(NUM_QUBITS)

for _ in range(optimal_iters):
    qc.compose(oracle, inplace=True)
    qc.compose(diff, inplace=True)

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

print(f"\nGrover circuit: {NUM_QUBITS} qubits, {M} marked states, "
      f"{optimal_iters} iteration(s).")


# ---------------------------------------------------------------------------
# 3. Run on ideal AerSimulator and check.
# ---------------------------------------------------------------------------

SHOTS = 4096
sim = AerSimulator()
result = sim.run(qc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first (qubit n-1 .. qubit 0), which is
# already the same convention as int(bs, 2) = sum(qubit_i * 2**i) -- the
# little-endian-by-qubit-index convention used when building the oracle.
def bitstring_to_index(bs: str) -> int:
    return int(bs, 2)

index_counts = {}
for bitstring, c in counts.items():
    idx = bitstring_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + c

marked_shots = sum(index_counts.get(i, 0) for i in marked_indices)
marked_prob = marked_shots / SHOTS

# Top-M most frequent outcomes, for comparison against the classical set.
top_indices = sorted(index_counts, key=lambda i: -index_counts[i])[:M]

print(f"\nSimulator results ({SHOTS} shots):")
print(f"  empirical probability mass on classical marked set: {marked_prob:.4f}")
print(f"  top-{M} measured indices: {sorted(top_indices)}")
print(f"  classical marked indices:  {sorted(marked_indices)}")

THRESHOLD = 0.90
quantum_matches_classical = (
    set(top_indices) == set(marked_indices) and marked_prob >= THRESHOLD
)

verified_against_classical = quantum_matches_classical

if verified_against_classical:
    print(f"\nPASS: Grover search concentrated ({marked_prob:.4f} >= {THRESHOLD}) "
          f"exactly on the classically-computed EKR star family "
          f"(size {len(classical_star_family)} == C(n-1,k-1) = {ekr_predicted_max}).")
else:
    print(f"\nFAIL: quantum result did not match classical star family "
          f"within threshold {THRESHOLD}.")
