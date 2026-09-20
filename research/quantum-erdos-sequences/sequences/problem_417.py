"""
Erdos problem #417 -- quantum-testable instance.

Source: erdosproblems.com problem 417 (data/problems.yaml entry
"number: '417'"). Its metadata lists OEIS ids A264810 and A061070
and tag "number theory". Problem 417 itself concerns iterates related
to sums of totients / the Euler totient function; the OEIS id we use
here, A061070, is directly and unambiguously a totient-derived
sequence:

    A061070(n) = number of DISTINCT values among {phi(1), phi(2), ..., phi(n)}

where phi is Euler's totient function (A000010). This is a small,
finite, exactly computable property: for any n, brute-force compute
phi(1..n) classically and count the distinct values.

Chosen classical property for the quantum circuit (a genuine search
problem, not a copied OEIS value):

    Fix n = 15 (search space of size 16, i.e. 4 qubits, indices 0..15)
    and target totient value T = 4.
    Classically compute phi(k) for k = 1..15 and find the exact set
    S = { k in [1,15] : phi(k) = T }.
    This set of "which integers up to 15 have phi(k) = 4" is exactly
    the kind of inversion data that underlies A061070(n) (the count
    of distinct totient values up to n is the number of totient
    values T for which such a nonempty S exists, for T ranging over
    phi(1..n)).

    We then build a REAL Grover search circuit over the 4-qubit index
    register whose oracle marks exactly the classically-computed set
    S, and run it on the ideal AerSimulator. Grover amplifies the
    marked states; we verify quantum correctness by checking that the
    measurement distribution concentrates (well above the uniform-
    random baseline) on exactly the classically-computed marked set S,
    i.e. the quantum search recovers the same "phi(k) = T" solution
    set that classical brute force computes.

No OEIS value is hard-coded: phi(k) for k=1..15, the target-value
search, and the marked set S are all computed from first principles
in this script, and the classical answer is what the quantum circuit
is checked against.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookups)
# ---------------------------------------------------------------------

def euler_phi(k: int) -> int:
    """Euler's totient function, computed by trial-division factorization."""
    if k <= 0:
        return 0
    result = k
    n = k
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1 if p == 2 else 2
    if n > 1:
        result -= result // n
    return result


N = 15          # search range: integers 1..N
NUM_QUBITS = 4  # indices 0..15 (index 0 is unused / never marked)
TARGET = 4      # totient value we search for

phi_values = {k: euler_phi(k) for k in range(1, N + 1)}

# A061070-style sanity check: number of distinct totient values among
# phi(1..N), computed purely classically.
distinct_totient_count = len(set(phi_values.values()))

# The exact search problem for the quantum circuit: which k in 1..N
# have phi(k) == TARGET?
marked_set = sorted(k for k, v in phi_values.items() if v == TARGET)

print("Classical Euler totient values phi(1..%d):" % N)
for k in range(1, N + 1):
    print(f"  phi({k:2d}) = {phi_values[k]}")
print(f"Distinct totient values among phi(1..{N}): {distinct_totient_count}"
      f" (this is the A061070(n) quantity for n={N})")
print(f"Marked set S = {{k : phi(k) = {TARGET}}} = {marked_set}")

assert len(marked_set) > 0, "chosen TARGET has no classical solutions; pick another TARGET"


# ---------------------------------------------------------------------
# 2. Build a real Grover search circuit whose oracle marks `marked_set`
# ---------------------------------------------------------------------

def bits_lsb_first(k: int, n: int):
    """Return the n bits of k as a list, index i = bit i (LSB first), so
    that bit i lines up with qubits[i] / clbit i the same way Qiskit's
    little-endian bitstring printing does (rightmost printed char = qubit 0).
    """
    return [(k >> i) & 1 for i in range(n)]


def apply_mark_oracle(qc: QuantumCircuit, qubits, marked, n_qubits):
    """Phase-flip exactly the computational basis states listed in `marked`.

    For each marked integer we surround a multi-controlled Z (implemented
    via H + MCX + H on the last qubit) with X gates on the qubits that
    should be 0 in that marked bitstring, so the MCX fires only on the
    exact target bitstring.
    """
    for m in marked:
        bits = bits_lsb_first(m, n_qubits)  # bits[i] corresponds to qubits[i]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(qubits[i])
        # multi-controlled Z on all n qubits: H-MCX-H trick on last qubit
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for i in zero_positions:
            qc.x(qubits[i])


def diffuser(qc: QuantumCircuit, qubits, n_qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


n = NUM_QUBITS
N_states = 2 ** n
M = len(marked_set)

qc = QuantumCircuit(n, n)
qubits = list(range(n))

# uniform superposition
qc.h(qubits)

# optimal number of Grover iterations for M marked states out of N_states
theta = math.asin(math.sqrt(M / N_states))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

for _ in range(iterations):
    apply_mark_oracle(qc, qubits, marked_set, n)
    diffuser(qc, qubits, n)

qc.measure(qubits, qubits)

print(f"\nGrover circuit: {n} qubits, {M} marked states out of {N_states},"
      f" {iterations} Grover iteration(s).")


# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
job = backend.run(compiled, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit reports bitstrings little-endian: the rightmost printed character
# is clbit 0 (= qubits[0] = bit 0 of our marked integers). That matches
# the bits_lsb_first() convention used to build the oracle, so parsing
# the printed bitstring directly as a base-2 integer recovers the same
# integer k that the oracle marked.
measured_ints = {int(bitstr, 2): cnt for bitstr, cnt in counts.items()}

marked_shots = sum(cnt for val, cnt in measured_ints.items() if val in marked_set)
marked_fraction = marked_shots / shots
uniform_baseline = M / N_states

# also recover the *set* of outcomes the circuit concentrates on, i.e.
# take the M most-frequent measured indices and compare to marked_set
top_m = sorted(sorted(measured_ints.items(), key=lambda kv: -kv[1])[:M],
                key=lambda kv: kv[0])
top_m_values = sorted(v for v, _ in top_m)

print(f"\nMeasured (top {M} most frequent outcomes): {top_m}")
print(f"Recovered index set from quantum search: {top_m_values}")
print(f"Classical marked set:                     {marked_set}")
print(f"Fraction of shots landing on a marked state: {marked_fraction:.4f}"
      f" (uniform-random baseline would be {uniform_baseline:.4f})")

quantum_matches_classical = (top_m_values == marked_set) and (marked_fraction > 3 * uniform_baseline)

print("\nPASS" if quantum_matches_classical else "\nFAIL")
assert quantum_matches_classical, "Grover search result did not match the classical marked set"
