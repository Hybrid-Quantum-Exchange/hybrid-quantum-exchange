"""
Erdos problem #435 -- quantum-testable instance.

OEIS id used: A389479
  a(n) = Frobenius number of the set { binomial(m,k) : k = 1..m-1 },
  where m = A024619(n) is the n-th positive integer that is NOT a
  prime power. The smallest such m is 6 (1,2,3,4,5,7,8,9 are all prime
  powers, 6 is not), so A024619(1) = 6 and A389479(1) = 49, i.e. the
  Frobenius number of the coin set { C(6,1),...,C(6,5) } = {6,15,20}
  is 49. This is the first published term of A389479.

Classical property tested here (computed from first principles, not
copied from OEIS):
  For the coin set S = {6, 15, 20} (derived above), the target value
  N = 50 = frobenius_number(S) + 1 IS representable as a non-negative
  integer combination 6*a + 15*b + 20*c, while N-1 = 49 is NOT. We
  classically brute-force the reachability table for 0..2000 with a
  standard coin/Frobenius dynamic program to confirm:
      frobenius_number(S) == 49   (matches A389479(1))
      49 is NOT representable, 50 IS representable.

Quantum computation:
  We build a genuine Grover search circuit over the bounded search
  space a in [0,7] (3 qubits), b in [0,3] (2 qubits), c in [0,3]
  (2 qubits) -- 7 qubits total, 128 basis states -- searching for
  assignments (a,b,c) with 6a + 15b + 20c == 50.
  The marked set is computed classically (this is the "database" a
  real Grover oracle would encode structurally; here it is built as
  an explicit multi-controlled-Z phase oracle over the marked
  bitstrings, which is a standard, legitimate way to realize a Grover
  oracle for an arbitrary boolean predicate on a small register).
  The number of Grover iterations is chosen from the classical count
  of marked states via the standard formula. After running on the
  ideal AerSimulator, we take the most-sampled outcome(s) and verify
  in software that each highly-sampled bitstring:
    (1) decodes to a valid (a,b,c) with 6a+15b+20c == 50, and
    (2) is a member of the classical marked-state set.
  PASS requires the quantum search to concentrate its probability
  mass (a chosen majority threshold) on true solutions of the
  equation 6a+15b+20c = 50, i.e. the circuit actually found a
  representation of 50 by the coin set that produced the Frobenius
  number 49 = A389479(1).
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# Step 1: classical derivation of the OEIS term and the coin set.
# ---------------------------------------------------------------------

def is_prime_power(n: int) -> bool:
    if n < 2:
        return False
    x = n
    d = 2
    factors = set()
    while d * d <= x:
        while x % d == 0:
            factors.add(d)
            x //= d
        d += 1
    if x > 1:
        factors.add(x)
    return len(factors) == 1


def smallest_non_prime_power(start: int = 2) -> int:
    m = start
    while is_prime_power(m):
        m += 1
    return m


def frobenius_number(coins, limit: int) -> int:
    reach = [False] * (limit + 1)
    reach[0] = True
    for i in range(1, limit + 1):
        for c in coins:
            if c <= i and reach[i - c]:
                reach[i] = True
                break
    unreachable = [i for i in range(limit + 1) if not reach[i]]
    if not unreachable:
        raise ValueError("limit too small to find Frobenius number")
    return max(unreachable), reach


m = smallest_non_prime_power(2)          # -> 6, first non-prime-power >= 2
assert m == 6, f"expected A024619(1) = 6, got {m}"

coins = sorted({math.comb(m, k) for k in range(1, m)})   # {6, 15, 20}
assert coins == [6, 15, 20], coins

FROB_LIMIT = 2000
frob, reach_table = frobenius_number(coins, FROB_LIMIT)
assert frob == 49, f"expected A389479(1) = 49, got {frob}"
assert reach_table[49] is False
assert reach_table[50] is True

N = frob + 1  # 50 -- the classical target value to represent with the coins


# ---------------------------------------------------------------------
# Step 2: classically enumerate the bounded search space and the
# marked (satisfying) assignments for the Grover oracle.
# ---------------------------------------------------------------------

A_BITS, B_BITS, C_BITS = 3, 2, 2         # a in [0,7], b in [0,3], c in [0,3]
A_MAX, B_MAX, C_MAX = 2 ** A_BITS, 2 ** B_BITS, 2 ** C_BITS
TOTAL_QUBITS = A_BITS + B_BITS + C_BITS  # 7 qubits, 128 basis states

marked_states = []          # list of bitstrings (MSB..LSB over [a|b|c])
marked_triples = []
for a in range(A_MAX):
    for b in range(B_MAX):
        for c in range(C_MAX):
            if coins[0] * a + coins[1] * b + coins[2] * c == N:
                bits = format(a, f"0{A_BITS}b") + format(b, f"0{B_BITS}b") + format(c, f"0{C_BITS}b")
                marked_states.append(bits)
                marked_triples.append((a, b, c))

assert marked_states, "classical search found no representation of N -- unexpected"
print(f"Classical: coins={coins}, Frobenius number={frob}, target N={N}")
print(f"Classical: {len(marked_states)} marked assignment(s) in bounded search space: {marked_triples}")


# ---------------------------------------------------------------------
# Step 3: build the Grover oracle and diffusion operator.
# ---------------------------------------------------------------------

def apply_multi_controlled_z(qc: QuantumCircuit, qubits):
    """Flip the phase of the |1..1> state on the given qubits (multi-controlled Z)."""
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def oracle(qc: QuantumCircuit, qubits, bitstring: str):
    """Phase-flip exactly the computational basis state `bitstring` (MSB first)."""
    zero_positions = [i for i, bit in enumerate(bitstring) if bit == "0"]
    # qubits[0] is the most-significant bit position of the string.
    target_qubits = [qubits[i] for i in range(len(bitstring))]
    flip_qubits = [qubits[i] for i in zero_positions]
    for q in flip_qubits:
        qc.x(q)
    apply_multi_controlled_z(qc, target_qubits)
    for q in flip_qubits:
        qc.x(q)


def diffusion(qc: QuantumCircuit, qubits):
    qc.h(qubits)
    qc.x(qubits)
    apply_multi_controlled_z(qc, qubits)
    qc.x(qubits)
    qc.h(qubits)


n = TOTAL_QUBITS
qc = QuantumCircuit(n, n)
all_qubits = list(range(n))

# uniform superposition
qc.h(all_qubits)

# number of Grover iterations for M marked states out of 2^n
Nstates = 2 ** n
Mmarked = len(marked_states)
theta = math.asin(math.sqrt(Mmarked / Nstates))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

for _ in range(iterations):
    for bitstring in marked_states:
        oracle(qc, all_qubits, bitstring)
    diffusion(qc, all_qubits)

qc.measure(all_qubits, all_qubits)

print(f"Quantum: {n} qubits, {Nstates} basis states, {Mmarked} marked, {iterations} Grover iteration(s)")


# ---------------------------------------------------------------------
# Step 4: run on the ideal AerSimulator and verify against the
# classical marked-state set.
# ---------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
SHOTS = 4096
result = sim.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's bit ordering in the returned bitstring is reversed relative to
# qubit index (classical register bit c[i] <-> qubit i, printed MSB..LSB
# by *qubit index descending*). Our circuit's qubit 0 is the MSB of `a`
# (see `all_qubits` construction), and Qiskit prints c[n-1]...c[0], i.e.
# qubit (n-1) first. Reverse to recover our [a|b|c] bit order.
def qiskit_to_problem_bits(qiskit_bitstring: str) -> str:
    return qiskit_bitstring[::-1]

marked_set = set(marked_states)
sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])

total_marked_shots = 0
top_bits_are_marked = True
for qiskit_bits, cnt in sorted_counts[: max(1, Mmarked)]:
    problem_bits = qiskit_to_problem_bits(qiskit_bits)
    is_marked = problem_bits in marked_set
    if is_marked:
        total_marked_shots += cnt
    else:
        top_bits_are_marked = False

fraction_marked_in_top = total_marked_shots / SHOTS

print("Top measured outcomes (problem bit order, count):")
for qiskit_bits, cnt in sorted_counts[:5]:
    problem_bits = qiskit_to_problem_bits(qiskit_bits)
    a = int(problem_bits[:A_BITS], 2)
    b = int(problem_bits[A_BITS:A_BITS + B_BITS], 2)
    c = int(problem_bits[A_BITS + B_BITS:], 2)
    print(f"  {problem_bits}  a={a} b={b} c={c}  6a+15b+20c={coins[0]*a+coins[1]*b+coins[2]*c}  count={cnt}")

print(f"Fraction of shots among the top-{Mmarked} outcomes landing on true solutions: {fraction_marked_in_top:.3f}")

# Success criterion: the Grover search concentrated at least 50% of all
# shots onto genuine solutions of 6a+15b+20c == N == 50 (the classical
# Frobenius-number-derived instance), i.e. it actually found a way to
# represent 50 by the coin set {6,15,20} whose Frobenius number is 49.
PASS = top_bits_are_marked and fraction_marked_in_top >= 0.5

if PASS:
    print("PASS")
else:
    print("FAIL")
