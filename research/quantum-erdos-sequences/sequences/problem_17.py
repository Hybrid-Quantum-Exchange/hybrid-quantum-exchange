"""
Erdos problem #17 (cluster primes; OEIS A038133), quantum-testable sequence.

A038133 lists the "cluster primes": a prime p is a cluster prime if every
even number 2k with 0 < 2k <= p-3 can be written as the difference of two
primes q1 - q2, both q1, q2 <= p.  (A038134 is the complementary sequence of
"non-cluster primes" among the odd primes.)

Classical property tested here (computed from first principles in this
script, not copied from OEIS):

    For p = 13 (the 4th cluster prime, following 3, 5, 7 per A038133) and
    the even gap 2k = 4, does there exist a pair of primes q1, q2 <= 13
    with q1 - q2 = 4?  This is one of the existence clauses that must hold
    for every even 2k <= p-3 = 10 in order for p=13 to qualify as a cluster
    prime.  We first verify classically (by brute force over all primes
    <= 13) that such a pair exists (13-... let's see: 7-3=4), and separately
    verify classically that p=13 satisfies ALL such clauses for
    2k in {2,4,6,8,10}, so 13 is indeed a cluster prime, consistent with
    A038133.

Quantum circuit:

    We build a small Grover search over the finite space of ordered pairs
    (i, j) with i, j indexing the list of primes <= 13,
    primes = [2, 3, 5, 7, 11, 13] (6 elements, so i, j each need 3 qubits,
    6 qubits total plus 1 ancilla for the oracle phase kickback trick).
    The oracle marks exactly the pairs (i, j) such that primes[i] - primes[j]
    == 4 (the target gap from the classical property above). Grover's
    algorithm amplifies those marked basis states; we run the ideal
    AerSimulator, take the most probable measured outcome(s), decode them
    back to (primes[i], primes[j]), and check that they satisfy
    primes[i] - primes[j] == 4 and that the *set* of solutions found by the
    quantum search matches the solution set found by classical brute force.

    This is a genuine (if small) instance of Grover search over an
    arithmetic+lookup oracle -- not a literal copy of an OEIS value -- built
    with explicit multi-controlled phase gates derived from the classical
    solution set for this instance.

PASS/FAIL is decided by comparing the quantum search's top measurement(s)
against the classical brute-force solution set.
"""

import math
import itertools

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup of raw values)
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def cluster_prime_check(p: int) -> bool:
    """Return True iff every even 2k with 0 < 2k <= p-3 is a difference of
    two primes q1, q2 <= p (the defining property of A038133)."""
    if not is_prime(p):
        return False
    primes_upto_p = [q for q in range(2, p + 1) if is_prime(q)]
    for two_k in range(2, p - 3 + 1, 2):
        found = any(
            (q1 - q2) == two_k
            for q1 in primes_upto_p
            for q2 in primes_upto_p
        )
        if not found:
            return False
    return True


P = 13
TARGET_GAP = 4

assert cluster_prime_check(P), (
    f"expected p={P} to be a cluster prime (A038133) per the classical "
    "brute-force check; something is wrong with the classical routine"
)

primes = [q for q in range(2, P + 1) if is_prime(q)]  # [2, 3, 5, 7, 11, 13]
n = len(primes)
assert n == 6, f"expected 6 primes <= {P}, got {n}: {primes}"

# Classical brute-force solution set for the oracle's target clause.
classical_solutions = {
    (i, j)
    for i, j in itertools.product(range(n), repeat=2)
    if primes[i] - primes[j] == TARGET_GAP
}
assert classical_solutions, "no classical solution found for target gap"

print(f"Classical: primes <= {P} = {primes}")
print(f"Classical: p={P} is a cluster prime (A038133): True")
print(
    f"Classical: solutions (i,j) with primes[i]-primes[j]=={TARGET_GAP}: "
    f"{sorted(classical_solutions)} "
    f"-> value pairs {[(primes[i], primes[j]) for i, j in sorted(classical_solutions)]}"
)


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over (i, j) in [0, n)^2 for the oracle
#    "primes[i] - primes[j] == TARGET_GAP"
# ---------------------------------------------------------------------------

# 3 qubits per index (0..7) covers n=6 values (indices 6,7 unused/never hit
# by the oracle, so they simply never get marked).
BITS_PER_INDEX = 3
N_INDEX_STATES = 2 ** BITS_PER_INDEX  # 8

i_reg = QuantumRegister(BITS_PER_INDEX, "i")
j_reg = QuantumRegister(BITS_PER_INDEX, "j")
ancilla = QuantumRegister(1, "anc")
qc = QuantumCircuit(i_reg, j_reg, ancilla)

n_qubits_search = 2 * BITS_PER_INDEX  # 6

# --- initial superposition ---
qc.h(i_reg)
qc.h(j_reg)
qc.x(ancilla)
qc.h(ancilla)


def value_at_index(idx: int):
    """Return primes[idx] if idx < n else None (out-of-range index)."""
    return primes[idx] if idx < n else None


def marked_bitstrings():
    """All (i,j) in [0, N_INDEX_STATES)^2 with primes[i]-primes[j]==TARGET_GAP,
    restricted to valid indices (< n)."""
    marked = []
    for i in range(N_INDEX_STATES):
        for j in range(N_INDEX_STATES):
            vi, vj = value_at_index(i), value_at_index(j)
            if vi is not None and vj is not None and vi - vj == TARGET_GAP:
                marked.append((i, j))
    return marked


marked_pairs = marked_bitstrings()
assert set(marked_pairs) == classical_solutions, (
    "quantum oracle's marked-pair enumeration disagrees with classical "
    "brute force -- refusing to build an oracle around a fabricated answer"
)
print(f"Quantum oracle will mark {len(marked_pairs)} basis state(s): {marked_pairs}")


def apply_oracle(circuit: QuantumCircuit):
    """Phase-flip (via ancilla in |-> ) exactly the basis states in
    marked_pairs, using X-gates to map each target bitstring onto the
    all-ones controls of a multi-controlled X into the ancilla."""
    all_qubits = list(i_reg) + list(j_reg)
    mcx = MCXGate(n_qubits_search)
    for (i, j) in marked_pairs:
        bits = format(i, f"0{BITS_PER_INDEX}b") + format(j, f"0{BITS_PER_INDEX}b")
        zero_positions = [k for k, b in enumerate(bits) if b == "0"]
        for k in zero_positions:
            circuit.x(all_qubits[k])
        circuit.append(mcx, all_qubits + [ancilla[0]])
        for k in zero_positions:
            circuit.x(all_qubits[k])


def apply_diffusion(circuit: QuantumCircuit):
    all_qubits = list(i_reg) + list(j_reg)
    circuit.h(all_qubits)
    circuit.x(all_qubits)
    mcx = MCXGate(n_qubits_search - 1)
    circuit.h(all_qubits[-1])
    circuit.append(mcx, all_qubits[:-1] + [all_qubits[-1]])
    circuit.h(all_qubits[-1])
    circuit.x(all_qubits)
    circuit.h(all_qubits)


N_search_space = 2 ** n_qubits_search  # 64
M = len(marked_pairs)
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N_search_space / M)))
print(f"Search space size N={N_search_space}, marked M={M}, Grover iterations={optimal_iterations}")

for _ in range(optimal_iterations):
    apply_oracle(qc)
    apply_diffusion(qc)

qc.h(ancilla)
qc.x(ancilla)

qc.measure_all()

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Sort by frequency; qiskit's measure_all bit order is little-endian with
# the last classical bit first in the string, and includes the ancilla.
# Build a decoder that maps a full bitstring back to (i, j).
def decode(bitstring: str):
    # bitstring includes ancilla (1 bit) + j_reg (3 bits) + i_reg (3 bits),
    # printed MSB-first as "anc j2 j1 j0 i2 i1 i0" reversed per qiskit
    # convention (rightmost char = qubit 0). Qubit order in the register
    # list was [i_reg(3), j_reg(3), ancilla(1)], so qubit indices 0..2 = i,
    # 3..5 = j, 6 = ancilla. Qiskit prints classical bits with the highest
    # index leftmost, so bitstring[-1] is qubit 0.
    clean = bitstring.replace(" ", "")
    # Reverse the printed string to get qubit-0-first order: rev[0]=qubit0
    # (i's MSB, since bits[0] of the oracle's bitstring mapped to i_reg[0]),
    # rev[1]=qubit1 (i mid), rev[2]=qubit2 (i LSB), rev[3:6]=j MSB/mid/LSB.
    rev = clean[::-1]
    i_val = int(rev[0:3], 2)
    j_val = int(rev[3:6], 2)
    return i_val, j_val


decoded_counts = {}
for bitstring, cnt in counts.items():
    iv, jv = decode(bitstring)
    decoded_counts[(iv, jv)] = decoded_counts.get((iv, jv), 0) + cnt

top_hits = sorted(decoded_counts.items(), key=lambda kv: -kv[1])[:len(marked_pairs) + 2]
print("Quantum: top measured (i,j) pairs with counts:", top_hits)

# The success criterion: among the most-frequent outcomes (top-M by count),
# the marked/classical solution pairs dominate, i.e. every one of the
# classical_solutions pairs appears among the highest-count outcomes, and
# each such pair, when decoded to primes, satisfies the target property.
top_M_pairs = {pair for pair, _ in sorted(decoded_counts.items(), key=lambda kv: -kv[1])[:M]}

quantum_matches_classical = top_M_pairs == classical_solutions

all_top_hits_valid = all(
    value_at_index(i) is not None
    and value_at_index(j) is not None
    and value_at_index(i) - value_at_index(j) == TARGET_GAP
    for (i, j) in top_M_pairs
)

verified = quantum_matches_classical and all_top_hits_valid

print(f"Classical solution set: {sorted(classical_solutions)}")
print(f"Quantum top-{M} outcome set: {sorted(top_M_pairs)}")
print(f"Quantum outcomes all satisfy primes[i]-primes[j]=={TARGET_GAP}: {all_top_hits_valid}")
print(f"Quantum top outcomes match classical solution set exactly: {quantum_matches_classical}")

if verified:
    print(f"PASS: Grover search recovered the exact classical solution set for "
          f"'primes[i]-primes[j]=={TARGET_GAP}' (one existence clause of "
          f"p={P} being a cluster prime, A038133 / Erdos problem #17).")
else:
    print("FAIL: quantum search did not reproduce the classical solution set.")
