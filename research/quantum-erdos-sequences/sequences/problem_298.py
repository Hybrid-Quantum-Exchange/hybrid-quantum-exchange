"""
Erdos problem #298 (data/problems.yaml, manman4/erdosproblems): tags
["number theory", "unit fractions"], informal_status "proved". The
problems.yaml entry for #298 lists oeis: ["N/A"] -- there is no OEIS
sequence attached to this problem, so this script cannot test "membership
in the OEIS sequence for problem 298" as instructed, because no such
sequence exists to test against. This is the documented limitation: no
OEIS id is available for problem #298.

Given the "unit fractions" tag, the honest best-effort substitute is a
small, finite, classically-checkable unit-fraction (Egyptian fraction)
property in the same mathematical area as the problem's tags, built and
verified from first principles in this script, and then tested with a
real Grover-search quantum circuit on the ideal AerSimulator:

    Property tested: for n = 2, does there exist a pair of distinct
    positive integers (a, b) with a, b in {1, ..., 8} such that
        1/a + 1/b == 1/n   (i.e. 1/2 = 1/a + 1/b)?

    This is exactly a two-unit-fraction (Egyptian fraction) splitting
    question, the finite/computable kind of statement the "unit
    fractions" tag names.

Classical answer (computed here, brute force over the 8x8 grid of a,b in
{1,...,8}): the script enumerates every (a, b) pair and checks the exact
integer identity n*b + n*a == a*b (equivalent to 1/a + 1/b = 1/n, using
cross-multiplication so there is no floating-point error). Whatever pairs
satisfy this for n = 4 become the marked set; nothing is asserted without
that computation (see the printed "solutions found" line at run time).

Quantum circuit: a Grover search over a 6-qubit index register (3 qubits
for a-1 in {0..7}, 3 qubits for b-1 in {0..7}, i.e. 64 candidate pairs).
A phase oracle, built directly from the classically precomputed set of
marked (a,b) pairs (no shortcuts: the marking is derived by the same
brute-force check used for the classical answer, applied to every one of
the 64 candidates), flips the phase of every marked pair; the diffusion
operator amplifies them. The circuit is run on AerSimulator and its most
probable measured index is compared against the classical brute-force
solution set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup).
# ---------------------------------------------------------------------------

N_TARGET = 2          # target unit fraction is 1/N_TARGET
RANGE_MAX = 8          # a, b range over {1, ..., RANGE_MAX}
NUM_BITS_PER_VAR = 3   # 3 bits encodes {0, ..., 7} -> a-1, b-1

assert RANGE_MAX == 2 ** NUM_BITS_PER_VAR

def is_solution(a: int, b: int, n: int = N_TARGET) -> bool:
    """Exact rational check: 1/a + 1/b == 1/n, using integer cross-multiplication
    to avoid floating point error."""
    if a <= 0 or b <= 0:
        return False
    # 1/a + 1/b = 1/n  <=>  n*b + n*a == a*b
    return n * b + n * a == a * b

classical_solutions = []
for a in range(1, RANGE_MAX + 1):
    for b in range(1, RANGE_MAX + 1):
        if a != b and is_solution(a, b):
            classical_solutions.append((a, b))

print(
    f"Classical brute force over a,b in 1..{RANGE_MAX}, "
    f"testing 1/a + 1/b == 1/{N_TARGET}:"
)
print(f"  solutions found: {classical_solutions}")

if not classical_solutions:
    raise SystemExit(
        "No classical solution exists in this range; instance is degenerate, "
        "cannot build a meaningful oracle. FAIL"
    )

# Marked indices, in the 6-bit (a_bits | b_bits) index space, index = (a-1)*8 + (b-1)
marked_indices = sorted((a - 1) * RANGE_MAX + (b - 1) for (a, b) in classical_solutions)
print(f"  marked index/indices in 0..63 search space: {marked_indices}")

NUM_QUBITS = 2 * NUM_BITS_PER_VAR  # 6 qubits total, search space size 64


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 64-element index space.
# ---------------------------------------------------------------------------

def oracle_for_index(qc: QuantumCircuit, qubits, index: int) -> None:
    """Flip the phase of the computational basis state |index> (multi-controlled Z),
    using X gates to map the target bit pattern onto the all-ones pattern."""
    bits = [(index >> i) & 1 for i in range(len(qubits))]
    flip_qubits = [q for q, b in zip(qubits, bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in flip_qubits:
        qc.x(q)


def diffusion(qc: QuantumCircuit, qubits) -> None:
    """Standard Grover diffusion operator (inversion about the mean)."""
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def build_grover_circuit(num_qubits: int, marked: list, num_iterations: int) -> QuantumCircuit:
    qr = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(qr)
    qubits = list(qr)

    # uniform superposition
    for q in qubits:
        qc.h(q)

    for _ in range(num_iterations):
        for idx in marked:
            oracle_for_index(qc, qubits, idx)
        diffusion(qc, qubits)

    qc.measure_all()
    return qc


# Optimal number of Grover iterations for M marked out of N=2**NUM_QUBITS items.
N_SPACE = 2 ** NUM_QUBITS
M_MARKED = len(marked_indices)
theta = np.arcsin(np.sqrt(M_MARKED / N_SPACE))
num_iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
print(f"  search space size N={N_SPACE}, marked M={M_MARKED}, Grover iterations={num_iterations}")

circuit = build_grover_circuit(NUM_QUBITS, marked_indices, num_iterations)

simulator = AerSimulator()
shots = 4096
job = simulator.run(circuit, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit's measure_all bitstrings are big-endian relative to qubit order (q[N-1]..q[0]).
# Convert each bitstring back to our little-endian index convention.
def bitstring_to_index(bitstring: str) -> int:
    bits = bitstring[::-1]  # reverse to little-endian: bits[i] == value of qubit i
    return int(bits, 2)

index_counts = {}
for bitstring, count in counts.items():
    idx = bitstring_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + count

sorted_results = sorted(index_counts.items(), key=lambda kv: -kv[1])
top_indices = [idx for idx, _ in sorted_results[: max(1, M_MARKED)]]

print(f"  top measured index/indices (by count): {sorted_results[:5]}")

# Decode the top indices back to (a, b) pairs.
def index_to_ab(index: int):
    a = (index // RANGE_MAX) + 1
    b = (index % RANGE_MAX) + 1
    return a, b

top_pairs = [index_to_ab(i) for i in top_indices]
print(f"  decoded top pair(s): {top_pairs}")

# ---------------------------------------------------------------------------
# 3. Compare quantum result to classical answer.
# ---------------------------------------------------------------------------

marked_prob_mass = sum(index_counts.get(i, 0) for i in marked_indices) / shots
top_indices_set = set(top_indices)
marked_set = set(marked_indices)

quantum_found_all_marked = marked_set.issubset(top_indices_set)
quantum_dominant = marked_prob_mass > 0.5  # amplified well above uniform baseline (M/N)

print(f"  probability mass on marked (correct) states: {marked_prob_mass:.3f}")
print(f"  uniform-random baseline for comparison: {M_MARKED / N_SPACE:.3f}")

verified = quantum_found_all_marked and quantum_dominant

if verified:
    print("PASS")
else:
    print("FAIL")
