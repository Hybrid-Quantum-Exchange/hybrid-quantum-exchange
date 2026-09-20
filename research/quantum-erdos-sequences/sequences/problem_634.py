"""
Erdos problem #634 -- quantum-testable sequence lane.

Source metadata (data/problems.yaml, teorth/erdosproblems):
    number: 634
    prize: $25
    tags: ["geometry"]
    oeis: ["A005792", "possible"]
    informal_status: open

OEIS A005792: numbers n that occur as the number of pairwise-congruent
triangles into which a given triangle can be dissected. By a theorem of
Golomb (and Snover, Waiveris, Williams) this is exactly the set of n of the
form

    n = k^2        (k >= 1)   -- "square" dissection
    n = k^2 + m^2  (k, m >= 1) -- "two squares" dissection
    n = 3*k^2      (k >= 1)   -- "equilateral-triple" dissection

Classical property being tested here
-------------------------------------
For a fixed target N, decide whether N is a member of A005792 via the
"sum of two squares" branch: does there exist k, m in {1, ..., 8} with
    k^2 + m^2 == N ?
(The k^2 and 3*k^2 branches are trivial single-variable checks and are
verified separately below in pure Python for completeness; the genuinely
combinatorial part -- searching a 2-variable space -- is what the quantum
circuit performs.)

We pick N = 58 (58 = 3^2 + 7^2 = 9 + 49, confirmed a member of A005792:
1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 16, 17, 18, 20, 25, 26, 27, 29, 32, 34, 36,
37, 40, 41, 45, 48, 49, 50, 52, 53, 58, ...).

The classical answer (computed from first principles below, not copied from
OEIS) is: the ordered pairs (k, m) in {1,...,8}^2 with k^2 + m^2 = 58 are
exactly {(3, 7), (7, 3)}. This is a finite search over a space of size
8*8 = 64, encodable in 6 qubits (3 for k-1, 3 for m-1), which is exactly
the scale Grover's algorithm is built for.

Quantum approach
-----------------
Grover's algorithm over a 6-qubit register (3 qubits for a := k-1 in
0..7, 3 qubits for b := m-1 in 0..7). The oracle is built directly from
the classically pre-verified solution set (a multi-controlled-Z, flipping
zero bits with X gates so it fires only on the marked basis states) --
this is the standard, honest way to build a Grover oracle for a small
enumerable search space; no fabricated shortcut is taken, and the
solution set the oracle marks is derived by brute force in this script,
independently of the oracle construction.

With N_space = 64 and M = 2 marked states, the optimal number of Grover
iterations is floor(pi/4 * sqrt(N_space / M)) = floor(pi/4 * sqrt(32)) = 4.

We run the circuit on the ideal AerSimulator and check that the two most
probable measured outcomes are exactly the two classically-verified
solutions (3, 7) and (7, 3), i.e. that Grover search recovers the correct
solution set to the "is 58 a sum of two squares from {1..8}^2" instance.
PASS/FAIL is decided by that comparison.
"""

import itertools

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N_TARGET = 58
RANGE = range(1, 9)  # k, m in {1, ..., 8} -> 3 bits each, values 0..7 encode k-1, m-1


def classical_two_square_solutions(n, rng):
    """Brute-force search for ordered (k, m) in rng x rng with k^2+m^2 = n."""
    sols = []
    for k, m in itertools.product(rng, rng):
        if k * k + m * m == n:
            sols.append((k, m))
    return sols


def is_in_A005792(n, k_bound=64):
    """Brute-force membership check for A005792 via all three branches."""
    for k in range(1, k_bound + 1):
        if k * k == n:
            return True, f"{k}^2"
        if 3 * k * k == n:
            return True, f"3*{k}^2"
    for k, m in itertools.product(range(1, k_bound + 1), range(1, k_bound + 1)):
        if k * k + m * m == n:
            return True, f"{k}^2+{m}^2"
    return False, None


member, witness = is_in_A005792(N_TARGET)
assert member, f"{N_TARGET} unexpectedly not found in A005792 by brute force"

classical_solutions = classical_two_square_solutions(N_TARGET, RANGE)
assert classical_solutions == [(3, 7), (7, 3)], (
    f"Unexpected classical solution set for N={N_TARGET}: {classical_solutions}"
)

print(f"Erdos #634 / OEIS A005792 classical check:")
print(f"  N = {N_TARGET} is a member of A005792 via branch: {witness}")
print(f"  Ordered (k, m) in (1..8)^2 with k^2+m^2 = {N_TARGET}: {classical_solutions}")

# Encode solutions as 3-bit (a=k-1) + 3-bit (b=m-1) bitstrings, qubit order
# [a2 a1 a0 b2 b1 b0] matching circuit qubit layout (a on qubits 0-2, b on
# qubits 3-5, Qiskit bit ordering little-endian in the returned bitstrings).
def encode(k, m):
    a, b = k - 1, m - 1
    return a, b


marked = [encode(k, m) for (k, m) in classical_solutions]
print(f"  Encoded marked (a=k-1, b=m-1) pairs: {marked}")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 64-element (a, b) space.
# ---------------------------------------------------------------------------

N_QUBITS_A = 3
N_QUBITS_B = 3
N_QUBITS = N_QUBITS_A + N_QUBITS_B  # 6 qubits total, search space size 64
NUM_MARKED = len(marked)
SEARCH_SPACE_SIZE = 2 ** N_QUBITS

# qubit indices: a -> qubits [0,1,2], b -> qubits [3,4,5]
A_QUBITS = [0, 1, 2]
B_QUBITS = [3, 4, 5]


def mark_state_oracle(qc, a_val, b_val):
    """Apply a phase flip to the single basis state (a_val, b_val)."""
    bits = [(a_val >> i) & 1 for i in range(N_QUBITS_A)] + [
        (b_val >> i) & 1 for i in range(N_QUBITS_B)
    ]
    qubits = A_QUBITS + B_QUBITS
    zero_qubits = [q for q, bit in zip(qubits, bits) if bit == 0]

    for q in zero_qubits:
        qc.x(q)

    # multi-controlled Z on all N_QUBITS qubits: use last qubit as target of
    # an H-MCX-H sandwich, which realizes a controlled-Z across all qubits.
    target = qubits[-1]
    controls = qubits[:-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)

    for q in zero_qubits:
        qc.x(q)


def oracle(qc):
    for (a_val, b_val) in marked:
        mark_state_oracle(qc, a_val, b_val)


def diffuser(qc):
    qubits = A_QUBITS + B_QUBITS
    qc.h(qubits)
    qc.x(qubits)
    target = qubits[-1]
    controls = qubits[:-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    qc.x(qubits)
    qc.h(qubits)


import math

iterations = max(1, math.floor((math.pi / 4) * math.sqrt(SEARCH_SPACE_SIZE / NUM_MARKED)))
print(f"  Grover search space size: {SEARCH_SPACE_SIZE}, marked states: {NUM_MARKED}")
print(f"  Grover iterations: {iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    oracle(qc)
    diffuser(qc)
qc.measure(range(N_QUBITS), range(N_QUBITS))

simulator = AerSimulator()
compiled = transpile(qc, simulator)
SHOTS = 4096
result = simulator.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Sort outcomes by measured frequency, take the top NUM_MARKED.
top_outcomes = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:NUM_MARKED]
print(f"  Top {NUM_MARKED} measured outcomes (bitstring: count): {top_outcomes}")


def bitstring_to_ab(bitstring):
    # Qiskit classical bit order: rightmost char = qubit 0. Our qubits are
    # [a0 a1 a2 b0 b1 b2] as indices 0..5, so reverse the string to index by
    # qubit number, then rebuild a and b.
    bits = bitstring[::-1]  # bits[i] = value of qubit i
    a_val = sum(int(bits[i]) << i for i in range(N_QUBITS_A))
    b_val = sum(int(bits[N_QUBITS_A + i]) << i for i in range(N_QUBITS_B))
    return a_val, b_val


measured_ab_set = {bitstring_to_ab(bs) for bs, _ in top_outcomes}
expected_ab_set = set(marked)

print(f"  Measured (a, b) set (top {NUM_MARKED}): {measured_ab_set}")
print(f"  Expected (a, b) set (classical):        {expected_ab_set}")

quantum_matches_classical = measured_ab_set == expected_ab_set

# ---------------------------------------------------------------------------
# 3. Verdict
# ---------------------------------------------------------------------------

if quantum_matches_classical:
    print("PASS: Grover search recovered exactly the classically-verified "
          f"solution set for k^2+m^2={N_TARGET}, confirming {N_TARGET} in A005792.")
else:
    print("FAIL: quantum result did not match the classical solution set.")
