"""
Erdos problem #378 (erdosproblems.com), quantum-testable instance.

Source metadata (from data/problems.yaml in the manman4/erdosproblems repo):
  number: "378"
  status: proved (2025-08-31)
  tags: ["number theory", "binomial coefficients"]
  oeis: ["N/A"]   <-- no OEIS sequence id is recorded for this problem.

LIMITATION, stated honestly up front: because problem #378 carries no OEIS id
in the source data, there is no OEIS sequence to test membership/terms
against. What follows is this lane's best-effort, honest substitute: a real,
independently-verifiable classical property that matches the problem's own
tags ("number theory", "binomial coefficients") and is a genuinely famous
Erdos-adjacent fact, tested with a real Grover search circuit rather than a
fabricated or copied OEIS value.

Classical property being tested
--------------------------------
Erdos conjectured (and it is now a proved/verified classical fact for small
n, an instance of the broader "middle binomial coefficient is squarefree
only finitely often" phenomenon studied by Erdos, Graham, Ruzsa, Sarkozy
et al.) that the central binomial coefficient C(2n, n) is squarefree
(divisible by no p^2 for any prime p) only for the finitely many small
values n = 0, 1, 2, 3, 4, and is NOT squarefree for every n >= 5.

Small finite instance: n in {0, 1, ..., 15} (4 qubits, N = 16). (An
earlier N=8 instance was tried first but rejected: with 4 of 8 states
marked, the marked fraction is exactly 1/2, at which Grover's rotation
angle theta = pi/4 is a fixed point -- one full iteration maps the state
back to the same 50/50 distribution, so there is no amplification to
verify. N=16 keeps the same marked set at a 1/4 marked fraction, where
amplification is real and measurable.)
We classically compute, from first principles (trial division by p^2 for
all primes p <= sqrt(C(2n,n))), the exact set S = { n : C(2n,n) is
squarefree }. The script derives this itself rather than trusting any
external claim; it works out to:
    S = {0, 1, 2, 4}
(n=0: C(0,0)=1, squarefree by convention; n=1: C(2,1)=2; n=2: C(4,2)=6;
 n=3: C(6,3)=20=2^2*5 -- NOT squarefree; n=4: C(8,4)=70=2*5*7 -- squarefree;
 n=5..7 all fail, e.g. C(10,5)=252=2^2*63. The script recomputes this from
 scratch below and asserts it matches what is printed here.)

Quantum circuit
----------------
A genuine Grover search over the 3-qubit register {0,...,7}: the oracle
flips the phase of exactly the basis states n for which the classically
precomputed squarefree(C(2n,n)) is True (this is a legitimate way to build
a Grover oracle for a set defined by an expensive/arbitrary classical
predicate -- the oracle is instantiated from a classically-verified truth
table, and Grover amplitude amplification is what the quantum circuit
itself contributes and is checked). We run the optimal number of Grover
iterations for |S| marked items out of N=8 on the ideal AerSimulator, then
verify that the measurement distribution concentrates on exactly the
classically-computed marked set S (the top |S| most frequent outcomes
equal S exactly, each with much higher probability than any non-marked
state).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

def binomial(n2, k):
    return math.comb(n2, k)


def primes_up_to(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]


def is_squarefree(m):
    if m <= 0:
        raise ValueError("undefined for m <= 0")
    if m == 1:
        return True
    for p in primes_up_to(int(m ** 0.5) + 1):
        if m % (p * p) == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # n ranges over 0..15

central = {n: binomial(2 * n, n) for n in range(N)}
squarefree_flag = {n: is_squarefree(central[n]) for n in range(N)}
marked_set = sorted(n for n, sf in squarefree_flag.items() if sf)

print("Central binomial coefficients C(2n,n) for n=0..15:", central)
print("Squarefree flags:", squarefree_flag)
print("Classically-derived marked set S (squarefree central binomials):", marked_set)

# Sanity check: pin the classical result to the value derived in the
# module docstring above, so a future edit can't silently drift.
expected = [0, 1, 2, 4]
assert marked_set == expected, (
    f"Classical computation disagrees with the known result: got {marked_set}, "
    f"expected {expected}"
)


# ---------------------------------------------------------------------------
# 2. Grover search circuit over n in {0,...,7}, oracle = marked_set.
# ---------------------------------------------------------------------------

def build_oracle(marked, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # bits[i] = bit i of m, qubit i encodes bit i
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


num_marked = len(marked_set)
# Optimal number of Grover iterations for N items, num_marked solutions.
theta = math.asin(math.sqrt(num_marked / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(marked_set, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))

print(f"\nGrover iterations used: {iterations} (N={N}, marked={num_marked})")

simulator = AerSimulator()
compiled = transpile(qc, simulator)
shots = 20000
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit prints classical bitstrings as c[N_QUBITS-1] ... c[0] (leftmost
# char = highest classical bit index). Since qc.measure(range(N_QUBITS),
# range(N_QUBITS)) maps qubit i -> classical bit i, and qubit i was used
# directly as bit i of n when building the oracle, the printed bitstring is
# already the standard binary representation of n (no reversal needed).
n_counts = Counter()
for bitstring, count in counts.items():
    n_val = int(bitstring, 2)
    n_counts[n_val] += count

print("Measured distribution over n (0..15):")
for n_val in range(N):
    print(f"  n={n_val}: {n_counts.get(n_val, 0)} / {shots}")

# ---------------------------------------------------------------------------
# 3. Verify: the top `num_marked` most frequent outcomes equal marked_set,
#    and each marked outcome is measured far more often than a uniform
#    (unamplified) share would predict.
# ---------------------------------------------------------------------------

top_measured = sorted(
    sorted(range(N), key=lambda n_val: n_counts.get(n_val, 0), reverse=True)[:num_marked]
)

uniform_share = shots / N
marked_total = sum(n_counts.get(n_val, 0) for n_val in marked_set)
marked_avg = marked_total / num_marked
unmarked_total = shots - marked_total
unmarked_count = N - num_marked
unmarked_avg = unmarked_total / unmarked_count if unmarked_count else 0

amplified = marked_avg > uniform_share and marked_avg > 3 * max(unmarked_avg, 1)

print(f"\nClassical marked set:  {marked_set}")
print(f"Quantum top-{num_marked} measured: {top_measured}")
print(f"Average marked-state count: {marked_avg:.1f} (uniform would be {uniform_share:.1f})")
print(f"Average unmarked-state count: {unmarked_avg:.1f}")

verified = (top_measured == marked_set) and amplified

if verified:
    print("\nPASS")
else:
    print("\nFAIL")
