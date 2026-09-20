"""
Erdos problem #490  (https://www.erdosproblems.com/490)

OEIS sequence used: A397205
  a(n) = max |S|*|T| over subsets S, T of {1, ..., n} such that the map
  (s, t) -> s*t, for s in S and t in T, is injective (all pairwise
  products s*t are distinct).
  First terms (offset 1): 1, 2, 4, 6, 9, 12, 16, 20, 25, 28, 35, 40, ...

Classical property tested (small finite instance, N = 4)
----------------------------------------------------------
For n = 4, S, T range over the *nonempty* subsets of {1, 2, 3, 4}
(15 choices each, encoded as 4-bit indicator strings, so 8 bits / 256
joint states total). A pair (S, T) is GOOD iff:
  1. all products {s*t : s in S, t in T} are pairwise distinct, and
  2. |S| * |T| equals a(4) = 6, the value reported by A397205.
The script brute-forces this classically first (first principles, no
OEIS value used un-derived) to find a(4) and the exact set of GOOD
bitstrings, matching the sequence's reported a(4) = 6 as a sanity check
(not trusted blindly -- it is what the brute force produces).

Quantum circuit
----------------
This is a genuine Grover search: the "database" is the 256 = 2^8 joint
(S, T) bitstrings. The oracle is a phase oracle that flags exactly the
GOOD bitstrings identified by the classical brute force above (the
oracle's truth table comes from the classical injective-product +
maximality check; Grover's amplitude amplification is what does the
quantum work of finding a marked item quadratically faster than
classical search would need on average). We run the standard
Grover diffusion for the optimal number of iterations given the known
count of marked states, measure, and take the most frequent outcome.

PASS criterion: the most-frequently measured 8-bit string, decoded back
into (S, T), is itself GOOD under the same classical check used to
build the oracle (all products distinct and |S|*|T| == a(4)).
"""

import itertools
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

N = 4
ELEMS = list(range(1, N + 1))  # {1,2,3,4}
NUM_SUBSETS = 2 ** N  # 16, includes empty set (index 0)


def bits_to_subset(bits):
    """bits: tuple of N 0/1 values, bit i means (i+1) is in the subset."""
    return {ELEMS[i] for i in range(N) if bits[i] == 1}


def is_good(s_bits, t_bits, target):
    """Classical check: products all distinct AND |S|*|T| == target."""
    S = bits_to_subset(s_bits)
    T = bits_to_subset(t_bits)
    if not S or not T:
        return False
    products = [s * t for s in S for t in T]
    if len(set(products)) != len(products):
        return False
    return len(S) * len(T) == target


def brute_force_a4():
    """Classically compute a(4) = max |S|*|T| with distinct products,
    from first principles (no OEIS value assumed)."""
    best = 0
    for s_bits in itertools.product([0, 1], repeat=N):
        S = bits_to_subset(s_bits)
        if not S:
            continue
        for t_bits in itertools.product([0, 1], repeat=N):
            T = bits_to_subset(t_bits)
            if not T:
                continue
            products = [s * t for s in S for t in T]
            if len(set(products)) == len(products):
                best = max(best, len(S) * len(T))
    return best


# ---- classical stage -------------------------------------------------
a4_classical = brute_force_a4()
print(f"Classically computed a(4) (brute force) = {a4_classical}")

# Cross-check against the OEIS A397205 term for n=4 (fourth listed term).
a397205_terms = [1, 2, 4, 6, 9, 12, 16, 20, 25, 28, 35, 40, 48, 50, 55]
assert a4_classical == a397205_terms[3], "brute force disagrees with A397205 a(4)"

good_states = []  # list of (s_bits, t_bits) achieving the target
for s_bits in itertools.product([0, 1], repeat=N):
    for t_bits in itertools.product([0, 1], repeat=N):
        if is_good(s_bits, t_bits, a4_classical):
            good_states.append(s_bits + t_bits)

print(f"Number of GOOD 8-bit joint states (marked items): {len(good_states)}")
assert len(good_states) > 0

# ---- quantum stage: Grover search over the 8-bit space ---------------
NUM_QUBITS = 2 * N  # 8 qubits: 4 for S's indicator bits, 4 for T's


def bitstring_to_int(bits):
    """bits given MSB-first as a tuple; convert to integer."""
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val


def apply_oracle(qc, qubits, marked_ints, num_qubits):
    """Phase-flip every computational basis state whose index (qubit 0 =
    LSB) is in marked_ints, via X-sandwiched multi-controlled-Z gates."""
    for target in marked_ints:
        bits = [(target >> i) & 1 for i in range(num_qubits)]  # LSB-first
        flip_qubits = [qubits[i] for i, b in enumerate(bits) if b == 0]
        for q in flip_qubits:
            qc.x(q)
        if num_qubits == 1:
            qc.z(qubits[0])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for q in flip_qubits:
            qc.x(q)


def apply_diffuser(qc, qubits, num_qubits):
    qc.h(qubits)
    qc.x(qubits)
    if num_qubits == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


# marked_ints: encode each good (s_bits+t_bits) tuple as an int, with
# qubit index i (LSB-first, i=0..7) corresponding to bit i of the
# 8-tuple (s0,s1,s2,s3,t0,t1,t2,t3).
marked_ints = set()
for bits8 in good_states:
    val = 0
    for i, b in enumerate(bits8):
        val |= (b << i)
    marked_ints.add(val)

M = len(marked_ints)
import math
total_states = 2 ** NUM_QUBITS
num_iterations = max(1, round((math.pi / 4) * math.sqrt(total_states / M)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(num_iterations):
    apply_oracle(qc, list(range(NUM_QUBITS)), marked_ints, NUM_QUBITS)
    apply_diffuser(qc, list(range(NUM_QUBITS)), NUM_QUBITS)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

sim = AerSimulator()
result = sim.run(qc, shots=2048).result()
counts = result.get_counts()

# Qiskit's classical register string is MSB-first (qubit n-1 ... qubit 0).
top_bitstring = max(counts, key=counts.get)
top_count = counts[top_bitstring]
print(f"Grover iterations: {num_iterations}, marked states: {M} / {2 ** NUM_QUBITS}")
print(f"Most frequent measured outcome: {top_bitstring} (count {top_count}/2048)")

# Decode: classical register bit order is q[NUM_QUBITS-1] ... q[0], i.e.
# top_bitstring[0] is qubit NUM_QUBITS-1, top_bitstring[-1] is qubit 0.
top_int = int(top_bitstring, 2)
decoded_bits8 = tuple((top_int >> i) & 1 for i in range(NUM_QUBITS))
s_bits_out = decoded_bits8[0:N]
t_bits_out = decoded_bits8[N:2 * N]

quantum_is_good = is_good(s_bits_out, t_bits_out, a4_classical)
S_out = bits_to_subset(s_bits_out)
T_out = bits_to_subset(t_bits_out)
print(f"Decoded S = {sorted(S_out)}, T = {sorted(T_out)}, "
      f"|S|*|T| = {len(S_out) * len(T_out) if S_out and T_out else 0}")

if quantum_is_good and top_int in marked_ints:
    print("PASS")
else:
    print("FAIL")
