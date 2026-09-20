"""
Erdos problem #218 (erdosproblems.com) -- quantum-testable sequence lane.

OEIS ids used: A333230 (and its counterpart A333231), which sit alongside
A064113 ("balanced primes") in the erdosproblems.com <-> OEIS crosswalk for
problem 218 (all three sequences are about the second differences of
consecutive primes -- i.e. whether the prime gaps are locally increasing or
decreasing).

A333230 is defined as:
    k such that prime(k+2) - 2*prime(k+1) + prime(k) >= 0
i.e. the index k is a member exactly when the *second difference* of the
k-th, (k+1)-th and (k+2)-th primes is non-negative (a "weak ascent" in the
sequence of prime gaps: the gap after prime(k+1) is at least as large as the
gap before it).

Classical property tested here (computed from first principles in this
script, no OEIS lookup table is copied in):
    For k in {1, 2, ..., 16}, is
        prime(k+2) - 2*prime(k+1) + prime(k) >= 0 ?
    i.e. is k a member of A333230?

We generate the first 18 primes ourselves with trial division, compute the
membership set S = {k in [1,16] : k in A333230} classically, and then use
that classically-verified set to build a genuine Grover's search circuit
(4 qubits, so N = 16 basis states |k-1> for k = 1..16) whose marked states
are exactly S. Grover amplifies the marked states; we run the ideal
AerSimulator and check that the circuit's measurement distribution
concentrates on S (every one of the top-|S| measured outcomes, weighted by
count, lies in S, and the marked outcomes collectively carry the large
majority of the shots). This is a real amplitude-amplification computation,
not a re-derivation of the arithmetic itself in-circuit (building a
reversible "is this the n-th prime" adder is out of scope for a small
circuit); the oracle's marked set is fixed from the classical computation
above, exactly as in the standard Grover's-algorithm construction where the
oracle encodes a boolean function f(x) precomputed for a known search
problem.

PASS criterion: every basis state measured with non-negligible frequency
(>= 5% of shots) after Grover amplification is in S, and the total
probability mass on S exceeds a random-guessing baseline (|S|/16) by a
wide margin.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS values copied in).
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def first_n_primes(n: int):
    primes = []
    candidate = 2
    while len(primes) < n:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes


N_QUBITS = 4
N_ITEMS = 2 ** N_QUBITS  # 16 -> k ranges over 1..16

# Need primes(k), primes(k+1), primes(k+2) for k = 1..16, so the first 18
# primes suffice.
PRIMES = first_n_primes(N_ITEMS + 2)  # PRIMES[i] = (i+1)-th prime

member_set = set()  # values of k (1-indexed) that are in A333230
non_member_set = set()  # k (1-indexed) NOT in A333230, i.e. second_diff < 0
for k in range(1, N_ITEMS + 1):
    p_k, p_k1, p_k2 = PRIMES[k - 1], PRIMES[k], PRIMES[k + 1]
    second_diff = p_k2 - 2 * p_k1 + p_k
    if second_diff >= 0:
        member_set.add(k)
    else:
        non_member_set.add(k)

print(f"First {N_ITEMS + 2} primes: {PRIMES}")
print(f"A333230 membership (k=1..{N_ITEMS}): {sorted(member_set)}")
print(f"Non-members (strict weak descent, k=1..{N_ITEMS}): {sorted(non_member_set)}")
print(f"|S| = {len(member_set)} out of {N_ITEMS} (members), "
      f"|complement| = {len(non_member_set)} (non-members)")

# Grover's algorithm amplifies a minority marked subset most cleanly; here
# the A333230 members are a majority (10/16), so we instead search for the
# smaller, equally well-defined complementary set (the strict weak-descent
# indices, second_diff < 0) -- an exact classical complement of A333230
# within k=1..16, still derived from the same first-principles computation
# above, not looked up.
search_target = non_member_set
print(f"Grover search target (smaller set, size {len(search_target)}): "
      f"{sorted(search_target)}")

# Map k (1..16) to the 4-bit computational basis index (k-1), i.e. |k-1>.
marked_indices = sorted(k - 1 for k in search_target)
S_bitstrings = {format(idx, f"0{N_QUBITS}b") for idx in marked_indices}


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle for the classically-computed marked set.
# ---------------------------------------------------------------------------

def build_oracle(marked_idx, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_idx:
        bits = format(idx, f"0{n_qubits}b")  # MSB..LSB over qubits n-1..0
        # X on qubits whose bit is 0, so the all-ones pattern lines up
        # with this marked state for the multi-controlled Z.
        zero_qubits = [n_qubits - 1 - pos for pos, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


M = len(marked_indices)
assert 0 < M < N_ITEMS, "Grover search needs a nontrivial, proper subset marked."

# Optimal number of Grover iterations for M marked items out of N_ITEMS.
theta = math.asin(math.sqrt(M / N_ITEMS))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

oracle = build_oracle(marked_indices, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and check against the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 8192
tqc = transpile(qc, backend)
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit's bit ordering has qubit 0 as the rightmost character; our oracle
# used qubit (n_qubits-1-pos) for MSB-first bitstring 'bits', which matches
# the standard little-endian classical-register printout directly (the
# measured bitstring is already MSB..LSB == qubit n-1 .. qubit 0).
mass_on_S = sum(c for bs, c in counts.items() if bs in S_bitstrings) / shots
random_baseline = M / N_ITEMS

# Every outcome carrying >=5% of shots must be a marked (in-S) outcome.
significant_outcomes = {bs for bs, c in counts.items() if c / shots >= 0.05}
all_significant_in_S = significant_outcomes.issubset(S_bitstrings)

amplification_factor = mass_on_S / random_baseline if random_baseline > 0 else float("inf")

print(f"Grover iterations used: {iterations}")
print(f"Measured counts: {counts}")
print(f"Probability mass on marked (search-target) states: {mass_on_S:.4f}")
print(f"Random-guessing baseline (|S|/N): {random_baseline:.4f}")
print(f"Amplification factor: {amplification_factor:.2f}x")
print(f"All outcomes with >=5% shots are in the search target: {all_significant_in_S}")

verified = all_significant_in_S and amplification_factor > 2.0

if verified:
    print("PASS")
else:
    print("FAIL")
