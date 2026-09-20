"""
Erdos problem #866 -- quantum-testable instance.

Source metadata (erdosproblems.com data, /home/user/manman4/erdosproblems/data/problems.yaml,
block "number: \"866\""):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["possible"]
    tags: ["number theory", "additive combinatorics"]

LIMITATION, stated honestly up front: problem #866's data block carries no real
OEIS sequence id -- the field literally contains the placeholder string
"possible", not an A-number. There is therefore no concrete integer sequence
from this problem to point a quantum circuit at, and this script does not
pretend otherwise or invent an OEIS id. What *is* real and usable from the
metadata is the pair of tags "number theory" and "additive combinatorics",
which name a well-defined, classical, finite decision problem in that area:
sumset membership for a small finite set of integers (the notion of a
"sum-free set" / sumset A+A is the standard additive-combinatorics object,
and is exactly the kind of finite/computable property Erdos-style additive
combinatorics problems are built from).

Concrete finite instance chosen here (N = 8, 3 qubits):
    A = {1, 2, 4, 6}  (universe indices 0..7, 3-qubit register)
    Sumset(A) = { a + b : a, b in A, a != b }
    Property tested: for x in {0, ..., 7}, is x an element of Sumset(A)
    that is itself also a member of A (i.e. x is both a candidate universe
    element 0..7 *and* expressible as a sum of two distinct elements of A)?

This is computed first-principles classically in `classical_sumset_hits()`
below, giving the exact marked set M subset of {0,...,7}. A Grover search
circuit (3 qubits, oracle built directly from M, diffusion operator) is then
run on the ideal AerSimulator to search for elements of M. The quantum
result (the most sampled bitstring(s) after the optimal number of Grover
iterations) is compared against the classical answer M for PASS/FAIL.

This is a generic small Grover instance whose oracle is derived from a real
additive-combinatorics computation (sumset membership), used here because
problem #866 supplies no OEIS sequence to search over directly. Reported
honestly: verified_against_classical is True only in the sense that the
Grover search's classical target set is itself computed and checked in this
script; it is not a verification of problem #866's actual open conjecture
(which is unformalized and open, and not something a small circuit can
settle).
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
import math


# ----------------------------------------------------------------------
# 1. Classical computation (first principles) of the marked set.
# ----------------------------------------------------------------------
def classical_sumset_hits(A, universe_size):
    """Return sorted list of x in [0, universe_size) such that x is in
    Sumset(A) = {a+b : a,b in A, a != b} AND x in A.
    Computed by brute force, no shortcuts."""
    sumset = set()
    for a in A:
        for b in A:
            if a != b:
                sumset.add(a + b)
    hits = sorted(x for x in range(universe_size) if x in sumset and x in A)
    return hits


A = [1, 2, 4, 6]
N_QUBITS = 3
UNIVERSE = 2 ** N_QUBITS  # 8

marked = classical_sumset_hits(A, UNIVERSE)
print(f"Classical marked set (x in A and x in Sumset(A)): {marked}")
assert len(marked) > 0, "instance must have at least one marked element"
assert len(marked) < UNIVERSE, "instance must not mark every element"


# ----------------------------------------------------------------------
# 2. Grover oracle + diffusion for this explicit marked set.
# ----------------------------------------------------------------------
def bits_of(x, n):
    return [(x >> i) & 1 for i in range(n)]


def append_oracle(qc, marked_values, n):
    """Phase-flip exactly the computational basis states in marked_values,
    via a multi-controlled-Z per marked value (X-sandwich to match 0-bits)."""
    for val in marked_values:
        bits = bits_of(val, n)
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for i in zero_positions:
            qc.x(i)


def append_diffusion(qc, n):
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


def build_grover_circuit(marked_values, n, iterations):
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        append_oracle(qc, marked_values, n)
        append_diffusion(qc, n)
    qc.measure(range(n), range(n))
    return qc


# Optimal number of Grover iterations for |M| marked items out of N total.
M = len(marked)
theta = math.asin(math.sqrt(M / UNIVERSE))
iterations = max(1, math.floor((math.pi / (4 * theta)) - 0.5))
print(f"|marked|={M}, N={UNIVERSE}, Grover iterations={iterations}")

qc = build_grover_circuit(marked, N_QUBITS, iterations)


# ----------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ----------------------------------------------------------------------
sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit's count keys already have qubit 0 as the rightmost (least
# significant) character, so a direct binary parse gives the register's
# integer value with no reversal needed.
int_counts = {}
for bitstring, c in counts.items():
    val = int(bitstring, 2)
    int_counts[val] = int_counts.get(val, 0) + c

sorted_counts = sorted(int_counts.items(), key=lambda kv: -kv[1])
print("Measurement distribution (value: count):", sorted_counts)

# Total probability mass landing on the classically-marked set.
marked_mass = sum(c for v, c in int_counts.items() if v in marked) / shots
print(f"Fraction of shots landing on classically-marked set: {marked_mass:.4f}")

# The single most frequent measured value(s) should be within the marked set,
# and the marked set should collectively dominate the distribution
# (Grover amplification), which is the real thing to check against the
# classical answer computed above.
top_value = sorted_counts[0][0]
top_value_in_marked = top_value in marked

passed = top_value_in_marked and marked_mass > 0.5

print(f"Top measured value: {top_value} -- in classical marked set {marked}? {top_value_in_marked}")

if passed:
    print("PASS")
else:
    print("FAIL")
