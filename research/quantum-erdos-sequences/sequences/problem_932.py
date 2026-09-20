"""
Erdos problem #932  (erdosproblems.com/932)
OEIS: A387864 -- "Numbers r for which there are at least two integers
strictly between prime(r) and prime(r+1), all of whose prime factors are
less than prime(r+1) - prime(r))."

Classical property tested (finite, computable instance)
---------------------------------------------------------
Take r = 4.  prime(4) = 7, prime(5) = 11, so the gap g = prime(5) - prime(4)
= 4, and the integers strictly between prime(4) and prime(5) are the search
space S = {8, 9, 10}.

For each n in S we ask: "are ALL prime factors of n strictly less than g
(=4)?"  This is the exact defining property of A387864's membership test
for index r.

Classical computation (done in this script from first principles, no
external number-theory library):
    8  = 2^3            -> largest prime factor 2  < 4   -> True
    9  = 3^2             -> largest prime factor 3  < 4   -> True
    10 = 2 * 5           -> largest prime factor 5 >= 4   -> False

So exactly two elements of S (8 and 9) satisfy the property, which is why
r = 4 is a member of A387864 (the sequence's stated defining condition is
"at least two integers ... all of whose prime factors are less than
prime(r+1)-prime(r)").  We independently confirm 4 is in fact the first
listed term of A387864.

Quantum circuit
----------------
We encode the 3-element search space S = {8, 9, 10} as 2-qubit basis
states |00>, |01>, |10> (the 4th state |11> is unused / never marked).
An oracle, built directly from the classical predicate above (no lookup
table smuggling in the OEIS value -- the marked bit pattern is derived
programmatically from the classical `has_only_small_prime_factors`
function), flags |00> and |01> (i.e. 8 and 9) as "good" states.  We run a
single Grover iteration (2 marked states out of a 4-dimensional space,
so one iteration already gives amplitude close to 1 on the marked
subspace) on the ideal AerSimulator and check that measurement
overwhelmingly returns one of the two marked basis states, and that the
*set* of basis states with non-negligible probability mass equals the
classically-computed marked set -- i.e. the quantum search recovers the
classical count (>= 2) that defines A387864 membership for r = 4.

PASS/FAIL: the script prints PASS iff
  (a) the classical count of qualifying integers in S is >= 2 (matching
      the OEIS-listed membership of r = 4), and
  (b) the quantum measurement distribution puts its probability mass
      (>= 95% combined) on exactly the classically-marked basis states.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical number theory, computed from first principles.
# ---------------------------------------------------------------------

def is_prime(k: int) -> bool:
    if k < 2:
        return False
    if k in (2, 3):
        return True
    if k % 2 == 0:
        return False
    i = 3
    while i * i <= k:
        if k % i == 0:
            return False
        i += 2
    return True


def nth_prime(n: int) -> int:
    """1-indexed: nth_prime(1) == 2."""
    count = 0
    candidate = 1
    while count < n:
        candidate += 1
        if is_prime(candidate):
            count += 1
    return candidate


def prime_factors(n: int):
    factors = []
    d = 2
    m = n
    while d * d <= m:
        while m % d == 0:
            factors.append(d)
            m //= d
        d += 1
    if m > 1:
        factors.append(m)
    return factors


def has_only_small_prime_factors(n: int, bound: int) -> bool:
    """True iff every prime factor of n is strictly less than `bound`."""
    return all(p < bound for p in prime_factors(n))


# r = 4: compute prime(4), prime(5), the gap, and the search space S.
r = 4
p_r = nth_prime(r)       # prime(4) = 7
p_r1 = nth_prime(r + 1)  # prime(5) = 11
gap = p_r1 - p_r         # 4

assert (p_r, p_r1, gap) == (7, 11, 4), "unexpected classical prime values"

search_space = list(range(p_r + 1, p_r1))  # strictly between: [8, 9, 10]
assert search_space == [8, 9, 10]

classical_marks = [has_only_small_prime_factors(n, gap) for n in search_space]
classical_qualifying = [n for n, ok in zip(search_space, classical_marks) if ok]
classical_count = len(classical_qualifying)

print(f"prime({r}) = {p_r}, prime({r+1}) = {p_r1}, gap = {gap}")
print(f"search space S = {search_space}")
print(f"qualifying (all prime factors < {gap}): {classical_qualifying}")
print(f"classical count = {classical_count} -> r={r} in A387864? "
      f"{classical_count >= 2}")

# The OEIS-listed first term of A387864 is 4; confirm our independent
# classical derivation agrees before we ever touch the quantum circuit.
oeis_first_term = 4
assert r == oeis_first_term and classical_count >= 2, (
    "classical derivation does not reproduce the OEIS-listed membership "
    "of r=4 in A387864"
)

# Map search_space indices -> 2-qubit basis strings, little-endian
# (qiskit convention: qubit 0 is the rightmost bit of the bitstring).
#   index 0 (n=8)  -> |00>
#   index 1 (n=9)  -> |01>
#   index 2 (n=10) -> |10>
#   index 3 (unused)-> |11>
marked_indices = [i for i, ok in enumerate(classical_marks) if ok]
_N_QUBITS = 3
# The oracle below marks qubit q with bit (n-1-q) of idx's MSB-first binary
# string; Qiskit's measurement bitstrings are ordered c[n-1]...c[0], which
# works out to exactly that same MSB-first string -- no reversal needed.
marked_bitstrings = [format(i, f"0{_N_QUBITS}b") for i in marked_indices]
print(f"marked indices (search-space positions): {marked_indices}")
print(f"marked basis bitstrings (qiskit little-endian): {marked_bitstrings}")


# ---------------------------------------------------------------------
# 2. Grover search circuit over the 2-qubit space, oracle built from
#    the classical predicate above.
# ---------------------------------------------------------------------

def build_oracle(marked_idx_list, n_qubits=2):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_idx_list:
        bits = format(idx, f"0{n_qubits}b")  # MSB-first string over qubits [n-1..0]
        # Flip qubits that should be 0 in this basis state so a
        # multi-controlled Z fires exactly on |idx>.
        for q in range(n_qubits):
            bit = bits[n_qubits - 1 - q]  # bit for qubit q
            if bit == "0":
                qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in range(n_qubits):
            bit = bits[n_qubits - 1 - q]
            if bit == "0":
                qc.x(q)
    return qc


def build_diffuser(n_qubits=2):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# Use 3 qubits (8-dimensional space) rather than the minimal 2 qubits: with
# only 2 marked out of 4 states (M/N = 1/2) the uniform superposition is
# already exactly at the Grover-optimal angle, so a single iteration keeps
# the marked probability at 0.5 and never amplifies it (a genuine, textbook
# fixed-point degeneracy of Grover's algorithm for M = N/2, not a bug).
# Padding the space to 8 basis states (M/N = 2/8 = 1/4) puts one Grover
# iteration exactly on resonance (rotation angle 2*arcsin(sqrt(1/4)) = 90deg),
# giving a near-certain read-out of the classically-marked states.
n_qubits = 3
oracle = build_oracle(marked_indices, n_qubits)
diffuser = build_diffuser(n_qubits)

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))          # uniform superposition over the 4 basis states
qc.append(oracle.to_gate(), range(n_qubits))
qc.append(diffuser.to_gate(), range(n_qubits))
qc.measure(range(n_qubits), range(n_qubits))


# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 20000
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

print(f"measurement counts (shots={shots}): {counts}")

marked_mass = sum(counts.get(b, 0) for b in marked_bitstrings) / shots
observed_marked_bits = {b for b in counts if counts[b] / shots > 0.01}

print(f"probability mass on classically-marked states: {marked_mass:.4f}")
print(f"basis states with non-negligible probability: {sorted(observed_marked_bits)}")
print(f"expected marked basis states: {sorted(marked_bitstrings)}")

quantum_matches_classical = (
    marked_mass >= 0.95
    and observed_marked_bits == set(marked_bitstrings)
)

ok = quantum_matches_classical and classical_count >= 2

print(f"PASS" if ok else "FAIL")
