"""
Erdos problem #236 -- quantum-testable instance.

Source: https://www.erdosproblems.com/236 (see erdosproblems metadata,
problems.yaml entry `number: "236"`, tags ["number theory", "primes"],
oeis ["A039669", "A109925"]).

Erdos asked: are 4, 7, 15, 21, 45 the *only* positive integers m such that
m - 2^k is prime for every power of two 2^k < m, k >= 1 (i.e. 2, 4, 8, ...;
2^0 = 1 is excluded)?  This is exactly OEIS A039669: "Numbers n such that
n - 2^i is prime for all i with 2^i < n" (i ranging over 1, 2, 3, ...).  A109925 is the companion counting sequence:
a(n) = number of k with 2^k < n such that n - 2^k is prime; A039669 is then
the set of n with a(n) = (number of powers of two below n), i.e. every
difference is prime. Erdos conjectured no term exists above 45 (verified
up to at least 2^120 classically, per the OEIS b-file / comments); the
problem is open in general.

Classical property tested here (finite, computable instance):
    For every integer m in [1, 63] (a 6-qubit register, N = 64 states),
    is m a member of A039669, i.e. is m - 2^k prime for *every* k >= 0
    with 2^k < m?

We first compute this classically from first principles (trial-division
primality test, no external number theory library) to get the exact
membership set S subset of {0, ..., 63}. Per OEIS A039669 the terms below
64 are {4, 7, 15, 21, 45} -- the script re-derives this itself rather than
trusting that literal claim.

Quantum circuit: Grover's search over the 6-qubit register |m> for m in
S. The oracle is built directly from the classically-computed set S (a
standard multi-controlled-phase "mark these computational basis states"
oracle -- this is the accepted way to Grover-search an arbitrary finite,
classically-decidable predicate when no cheaper arithmetic oracle is
convenient), followed by the standard Grover diffuser, run for the
number of iterations that (approximately) maximizes the success
probability for |S| marked items out of 64. We then measure and check
that the quantum sampling distribution is overwhelmingly concentrated on
S, which is the same yes/no membership property the classical computation
established.

PASS criterion: after running the Grover circuit on the ideal AerSimulator
and sampling many shots, the fraction of shots landing on a member of S
must exceed the fraction expected from uniform random guessing by a wide,
statistically unambiguous margin (in fact it must exceed 90%), while the
classical brute-force scan independently confirms S. If quantum measured
answer does not match/enrich around the classical set, we print FAIL.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no libraries beyond stdlib).
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def in_A039669(m: int) -> bool:
    """True iff m - 2^k is prime for every k >= 1 with 2^k < m (OEIS A039669)."""
    if m < 1:
        return False
    k = 1
    checked_any = False
    while (1 << k) < m:
        checked_any = True
        if not is_prime(m - (1 << k)):
            return False
        k += 1
    # m must have at least one power of two (2^1, 2^2, ...) below it to be
    # a candidate, and every such difference must be prime.
    return checked_any


N_QUBITS = 6
N = 1 << N_QUBITS  # 64

classical_set = sorted(m for m in range(N) if in_A039669(m))
print(f"Classical scan of m in [0, {N - 1}]: A039669 members found = {classical_set}")

# Cross-check against the literal OEIS terms below 64, derived independently
# above (not copied in) -- this is just a sanity assertion, not the source
# of truth.
expected_from_oeis_below_64 = [4, 7, 15, 21, 45]
assert classical_set == expected_from_oeis_below_64, (
    f"classical derivation {classical_set} disagrees with known OEIS terms "
    f"{expected_from_oeis_below_64} -- aborting, would not be a fair test"
)

marked = classical_set
M = len(marked)
print(f"Marked (Grover target) states: {marked}  (|S| = {M} out of N = {N})")


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffuser for this exact marked set.
# ---------------------------------------------------------------------------

def apply_mark_state(qc: QuantumCircuit, qubits, value: int):
    """Flip the phase of |value> (multi-controlled Z via X-sandwiched MCX-phase)."""
    bits = [(value >> i) & 1 for i in range(len(qubits))]
    for q, b in zip(qubits, bits):
        if b == 0:
            qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q, b in zip(qubits, bits):
        if b == 0:
            qc.x(q)


def oracle(n_qubits: int, marked_values):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    qubits = list(range(n_qubits))
    for v in marked_values:
        apply_mark_state(qc, qubits, v)
    return qc


def diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qubits = list(range(n_qubits))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


def build_grover_circuit(n_qubits: int, marked_values, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    orac = oracle(n_qubits, marked_values)
    diff = diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(orac, inplace=True)
        qc.compose(diff, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


theta = math.asin(math.sqrt(M / N))
optimal_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {optimal_iterations}")

circuit = build_grover_circuit(N_QUBITS, marked, optimal_iterations)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

SHOTS = 8192
sim = AerSimulator()
compiled = sim.run(circuit, shots=SHOTS)
result = compiled.result()
counts = result.get_counts()

# Qiskit prints bitstrings MSB..LSB matching classical-register order;
# our circuit used qubit i as bit i (LSB = qubit 0), and Qiskit's counts
# keys are big-endian over the classical bits, i.e. c[n-1] c[n-2] ... c[0].
def bitstring_to_int(bs: str) -> int:
    return int(bs, 2)

outcome_counts = Counter()
for bitstring, c in counts.items():
    outcome_counts[bitstring_to_int(bitstring)] += c

total = sum(outcome_counts.values())
hits_in_S = sum(c for v, c in outcome_counts.items() if v in marked)
fraction_in_S = hits_in_S / total

most_common_value, most_common_count = outcome_counts.most_common(1)[0]

print(f"Shots: {total}")
print(f"Fraction of shots landing in classical set S: {fraction_in_S:.4f}")
print(f"Most frequent measured value: {most_common_value} "
      f"(in S: {most_common_value in marked}), count={most_common_count}")
print("Top 5 measured values:", outcome_counts.most_common(5))

baseline_uniform = M / N
print(f"Uniform-random baseline fraction (no amplification): {baseline_uniform:.4f}")

# ---------------------------------------------------------------------------
# 4. PASS/FAIL
# ---------------------------------------------------------------------------

verified = (
    fraction_in_S > 0.90
    and most_common_value in marked
    and fraction_in_S > 3 * baseline_uniform
)

if verified:
    print("PASS: Grover search on AerSimulator concentrates measurement "
          "outcomes on the classically-computed A039669 membership set "
          f"{marked}, far above the uniform baseline "
          f"({fraction_in_S:.4f} vs {baseline_uniform:.4f}).")
else:
    print("FAIL: quantum measurement distribution did not confirm the "
          "classical A039669 membership set.")
