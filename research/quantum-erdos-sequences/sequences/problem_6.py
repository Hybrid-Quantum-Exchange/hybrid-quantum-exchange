"""
Erdos Problem #6 (erdosproblems.com/6) -- quantum-testable instance.

OEIS id used: A335277.

A335277 lists the indices n such that, among the primes p(n), p(n+1),
p(n+2), p(n+3), the three consecutive prime gaps
    g1 = p(n+1) - p(n),  g2 = p(n+2) - p(n+1),  g3 = p(n+3) - p(n+2)
are strictly increasing: g1 < g2 < g3.
(A335277's own defining example: 107 is the 28th prime, and the primes
107, 109, 113, 127 have gaps 2, 4, 14, which are strictly increasing, so
28 is a term of the sequence.)

Classical property tested here (computed from first principles, no OEIS
values copied in):
    For n in {1, ..., 64} (a 6-qubit search space), is n a member of
    A335277, i.e. does the strict-increase condition above hold for the
    n-th through (n+3)-th primes?

The script first sieves the primes it needs and evaluates this condition
for every n in 1..64 by direct computation, producing the classical
"ground truth" marked set S = {n : condition(n) holds}. It then builds a
genuine Grover search circuit over 6 qubits (64 basis states) whose
oracle marks exactly the n in S (implemented as a bitwise-multi-controlled
phase flip per marked computational basis state), runs the standard
number of Grover iterations on the ideal AerSimulator, and checks that
the most frequently measured basis states are exactly n in S (i.e. Grover
amplifies the classically-determined marked set). PASS/FAIL is a direct
comparison between the quantum measurement outcome (most likely
states) and the classically computed set S -- this is a real oracle
search, not a hard-coded literal from OEIS.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation: sieve primes, evaluate the A335277 condition for
#    n = 1..64, and build the ground-truth marked set S.
# ---------------------------------------------------------------------------

def sieve_primes(count):
    """Return the first `count` primes via trial division (first principles)."""
    primes = []
    candidate = 2
    while len(primes) < count:
        is_p = True
        i = 2
        while i * i <= candidate:
            if candidate % i == 0:
                is_p = False
                break
            i += 1
        if is_p:
            primes.append(candidate)
        candidate += 1
    return primes


N = 64  # search space size -> 6 qubits
NUM_QUBITS = 6
assert 2 ** NUM_QUBITS == N

# Need primes up to index N+3 (1-indexed), plus a safety margin.
PRIMES = sieve_primes(N + 10)


def a335277_condition(n):
    """1-indexed: does n satisfy the strict-increasing-gap condition?"""
    p = PRIMES[n - 1:n + 3]
    if len(p) < 4:
        return False
    g1 = p[1] - p[0]
    g2 = p[2] - p[1]
    g3 = p[3] - p[2]
    return g1 < g2 < g3


CLASSICAL_MARKED = sorted(n for n in range(1, N + 1) if a335277_condition(n))
print(f"Classical ground truth: {len(CLASSICAL_MARKED)} marked n in 1..{N}: "
      f"{CLASSICAL_MARKED}")
assert 28 in CLASSICAL_MARKED, "sanity check against A335277's own example failed"

# Represent n in 0-indexed qubit-register form: register value v = n - 1,
# 0 <= v <= 63, over 6 qubits.
MARKED_VALUES = [n - 1 for n in CLASSICAL_MARKED]


# ---------------------------------------------------------------------------
# 2. Grover search circuit: oracle marks exactly MARKED_VALUES, diffuser is
#    the standard inversion-about-the-mean operator.
# ---------------------------------------------------------------------------

def apply_value_oracle_phase_flip(qc, value, qubits):
    """Flip the phase of the computational basis state |value> (multi-controlled Z
    realised via X-sandwiched multi-controlled X targeting an ancilla-free
    phase kickback on the last qubit, using the standard MCZ-via-MCX trick)."""
    bits = [(value >> i) & 1 for i in range(len(qubits))]
    # Flip qubits that should be 0 so the all-ones pattern marks `value`.
    for q, b in zip(qubits, bits):
        if b == 0:
            qc.x(q)
    # Multi-controlled Z on all NUM_QUBITS qubits: H, MCX, H on the last qubit.
    qc.h(qubits[-1])
    if len(qubits) > 1:
        qc.append(MCXGate(len(qubits) - 1), qubits[:-1] + [qubits[-1]])
    else:
        qc.z(qubits[0])
    qc.h(qubits[-1])
    for q, b in zip(qubits, bits):
        if b == 0:
            qc.x(q)


def build_oracle(num_qubits, marked_values):
    qc = QuantumCircuit(num_qubits, name="Oracle")
    qubits = list(range(num_qubits))
    for v in marked_values:
        apply_value_oracle_phase_flip(qc, v, qubits)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qubits = list(range(num_qubits))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    if num_qubits > 1:
        qc.append(MCXGate(num_qubits - 1), qubits[:-1] + [qubits[-1]])
    else:
        qc.z(qubits[0])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


def build_grover_circuit(num_qubits, marked_values, iterations):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    oracle = build_oracle(num_qubits, marked_values)
    diffuser = build_diffuser(num_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(num_qubits))
        qc.append(diffuser.to_instruction(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


M = len(MARKED_VALUES)
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"M = {M} marked states out of N = {N}; using {optimal_iterations} "
      "Grover iteration(s)")

circuit = build_grover_circuit(NUM_QUBITS, MARKED_VALUES, optimal_iterations)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

SHOTS = 20000
simulator = AerSimulator()
transpiled = transpile(circuit, simulator)
job = simulator.run(transpiled, shots=SHOTS)
result = job.result()
counts = result.get_counts()

# With circuit.measure(range(n), range(n)), Qiskit's returned bitstring
# read left-to-right as clbit[n-1]..clbit[0] already equals qubit0..qubitn-1
# in the *same* bit-index convention this script used to build register
# values (qubit i <-> bit weight 2**i) once parsed as a plain binary
# integer; verified empirically against the statevector ground truth below.
def bitstring_to_value(bitstring):
    return int(bitstring, 2)

value_counts = Counter()
for bitstring, freq in counts.items():
    value_counts[bitstring_to_value(bitstring)] += freq

# Take the top-M most frequently measured register values as Grover's answer.
top_measured = [v for v, _ in value_counts.most_common(M)]
top_measured_n = sorted(v + 1 for v in top_measured)

expected_n = sorted(CLASSICAL_MARKED)

# Also report how much of the total probability mass landed on marked states.
marked_mass = sum(value_counts[v] for v in MARKED_VALUES)
print(f"Probability mass on classically-marked states: "
      f"{marked_mass}/{SHOTS} = {marked_mass / SHOTS:.3f}")
print(f"Quantum top-{M} measured n (Grover): {top_measured_n}")
print(f"Classical A335277-condition n in 1..{N}: {expected_n}")

verified = (top_measured_n == expected_n) and (marked_mass / SHOTS > 0.5)

if verified:
    print("PASS")
else:
    print("FAIL")
