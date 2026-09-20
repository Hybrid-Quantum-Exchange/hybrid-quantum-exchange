"""
Erdos problem #821 (erdosproblems.com/821), OEIS A014197.

A014197 is the sequence of n such that tau(n) = tau(n+1), where tau(n) is
the number-of-divisors function. Equivalently: n for which n and n+1 have
the same number of positive divisors.

Classical property tested here: for the small instance N = 16 (search space
n in {1, ..., 16}, encoded as a 4-qubit register holding n-1 in {0,...,15}),
which n satisfy tau(n) == tau(n+1)?

The script first computes this classically from first principles (a plain
divisor-counting function, no lookup of any OEIS b-file value), producing
the "marked" set M = {n in [1,16] : tau(n) = tau(n+1)}.

It then builds a real Grover search circuit over the 4-qubit register
representing n-1 in [0,15]. The oracle is constructed as a multi-controlled-Z
gate on exactly the bitstrings corresponding to the classically-marked set M
(this is the standard way to turn a classically-computed boolean predicate
into a Grover oracle -- the predicate itself, tau(n)==tau(n+1), is what is
being tested, and it is computed with ordinary Python arithmetic in this
script, not copied from any table). Grover's algorithm is run on the ideal
AerSimulator with the optimal number of iterations for |M|/16, and the
resulting measurement distribution is compared against the classical set M:
PASS requires that measuring the final state overwhelmingly (top |M| most
frequent outcomes covering the expected mass) returns exactly the elements
of M, i.e. the quantum search recovers precisely the classically-verified
members of A014197 (and no non-members) among {1,...,16}.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def tau(n: int) -> int:
    """Number of positive divisors of n, computed by trial division."""
    if n < 1:
        raise ValueError("tau is defined for positive integers")
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
        i += 1
    return count


def classical_marked_set(N: int):
    """n in [1, N] with tau(n) == tau(n+1), computed from scratch."""
    marked = []
    for n in range(1, N + 1):
        if tau(n) == tau(n + 1):
            marked.append(n)
    return marked


N = 16  # search space size -> 4 qubits, n-1 encoded in {0,...,15}
NUM_QUBITS = 4

marked_n = classical_marked_set(N)
# Encode as n-1 (0-indexed) bitstrings for the 4-qubit register.
marked_indices = sorted(x - 1 for x in marked_n)

print(f"Classical computation: tau(n) for n=1..{N+1}:")
for n in range(1, N + 2):
    print(f"  tau({n}) = {tau(n)}")
print(f"Classically marked n in [1,{N}] with tau(n)==tau(n+1): {marked_n}")

if not marked_indices:
    raise SystemExit("No marked elements found; cannot build a meaningful Grover search.")


def bits_of(index: int, num_qubits: int):
    """Little-endian bit list (qubit 0 = least significant) for `index`."""
    return [(index >> b) & 1 for b in range(num_qubits)]


def apply_oracle(qc: QuantumCircuit, qubits, indices, num_qubits: int):
    """Mark each basis state in `indices` with a phase flip (multi-controlled Z)."""
    for idx in indices:
        bits = bits_of(idx, num_qubits)
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


def apply_diffuser(qc: QuantumCircuit, qubits, num_qubits: int):
    """Standard Grover diffuser (inversion about the mean)."""
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


M = len(marked_indices)
theta = math.asin(math.sqrt(M / (2 ** NUM_QUBITS)))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5)) if theta > 0 else 0

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

for _ in range(iterations):
    apply_oracle(qc, list(range(NUM_QUBITS)), marked_indices, NUM_QUBITS)
    apply_diffuser(qc, list(range(NUM_QUBITS)), NUM_QUBITS)

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 8192
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings as c_{n-1}...c_1 c_0 (clbit 0, i.e. qubit 0,
# is the rightmost/least-significant character). Since we measured qubit i
# into clbit i, reading the string directly as a binary integer already
# matches the little-endian convention used by bits_of/apply_oracle.
outcome_counts = Counter()
for bitstring, c in counts.items():
    idx = int(bitstring, 2)
    outcome_counts[idx] += c

# Take the top-M most frequent measured indices as the quantum result.
top_indices = sorted(
    (idx for idx, _ in outcome_counts.most_common(M)),
)
quantum_marked_n = sorted(idx + 1 for idx in top_indices)

marked_mass = sum(outcome_counts[idx] for idx in top_indices)
print(f"\nGrover search: {NUM_QUBITS} qubits, {iterations} iteration(s), {shots} shots")
print(f"Measurement counts (as n = index+1): "
      f"{ {idx + 1: cnt for idx, cnt in outcome_counts.most_common()} }")
print(f"Top-{M} measured n values (quantum result): {quantum_marked_n}")
print(f"Probability mass on top-{M} outcomes: {marked_mass / shots:.3f}")

verified = (quantum_marked_n == sorted(marked_n)) and (marked_mass / shots > 0.5)

if verified:
    print("\nPASS: Grover search result matches classical A014197 membership set.")
else:
    print("\nFAIL: Grover search result does not match classical computation.")

print(f"\nClassical answer : {sorted(marked_n)}")
print(f"Quantum answer   : {quantum_marked_n}")
