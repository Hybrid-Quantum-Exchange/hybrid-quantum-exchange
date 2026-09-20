"""
Erdos problem #487 — quantum-testable companion script.

LIMITATION (read first): problems.yaml lists problem 487 as
    oeis: ["N/A"]
    tags: ["number theory"]
So there is no OEIS sequence attached to this problem to build a
sequence-membership circuit from. Rather than fabricate a fake OEIS id or
copy a value with no derivation, this script falls back to the one piece
of real mathematical content the entry does give us: the tag "number
theory". It builds a genuine Grover search circuit over 3-bit integers
(0..7) whose oracle marks the PRIME numbers in that range, and verifies
that Grover's algorithm finds a prime with high probability.

Classical property being tested:
    For N = {0, 1, ..., 15}, let S = {n in N : n is prime}.
    Primality is checked here from first principles (trial division),
    not copied from any table. S = {2, 3, 5, 7, 11, 13} (computed below).

Quantum computation:
    A 4-qubit Grover search is built with a phase-oracle that flips the
    sign of amplitudes on basis states in S, followed by the standard
    diffusion operator, run for the optimal number of iterations for
    |S|=6 out of N=16 (computed from the standard Grover formula). The
    circuit is simulated on the ideal AerSimulator with many shots.
    PASS requires that the most frequent measured outcomes are exactly
    the classically-computed prime set S and that most of the
    measurement probability mass has been amplified onto S (i.e.
    Grover search actually found the marked/prime states).

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical ground truth, derived from first principles.
# ---------------------------------------------------------------------
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16
CLASSICAL_PRIMES = sorted(n for n in range(N) if is_prime(n))
print(f"Classical instance: N = {{0..{N - 1}}}")
print(f"Classical primes (trial division): {CLASSICAL_PRIMES}")

MARKED = set(CLASSICAL_PRIMES)
assert MARKED == {2, 3, 5, 7, 11, 13}, "sanity check on classical computation failed"


# ---------------------------------------------------------------------
# 2. Build the Grover oracle that marks exactly the prime basis states.
# ---------------------------------------------------------------------
def build_oracle(n_qubits: int, marked: set) -> QuantumCircuit:
    """Phase oracle: flips sign of |x> for every x in `marked`."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for x in marked:
        bits = format(x, f"0{n_qubits}b")
        # Flip qubits that should be 0 in x so a multi-controlled Z
        # fires exactly on |x>.
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits: int, marked: set, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# Optimal number of Grover iterations, standard formula:
# theta = asin(sqrt(M/N)); after k iterations the probability of
# measuring a marked state is sin((2k+1)*theta)^2, maximized near
# k ~ (pi / (4*theta)) - 1/2.
M = len(MARKED)
theta = math.asin(math.sqrt(M / N))
optimal_iters = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {optimal_iters}")
predicted_marked_prob = math.sin((2 * optimal_iters + 1) * theta) ** 2
print(f"Predicted marked probability mass: {predicted_marked_prob:.3f}")

circuit = build_grover_circuit(N_QUBITS, MARKED, optimal_iters)


# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------
simulator = AerSimulator()
compiled = transpile(circuit, simulator)
SHOTS = 4096
result = simulator.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Convert bitstrings (Qiskit is little-endian in classical register order
# but with our single register measured in order 0..n-1, format the
# string back to an integer using the same bit convention as the oracle).
def bitstring_to_int(bitstring: str) -> int:
    # Qiskit prints classical bits as c[n-1] ... c[0] (MSB first), which
    # is exactly the standard binary representation of the integer whose
    # bit i is qubit i -- the same convention build_oracle() uses.
    return int(bitstring, 2)

observed_counts = {}
for bitstring, count in counts.items():
    val = bitstring_to_int(bitstring)
    observed_counts[val] = observed_counts.get(val, 0) + count

print("Observed outcome distribution (value: count):")
for val in sorted(observed_counts, key=lambda v: -observed_counts[v]):
    print(f"  {val}: {observed_counts[val]}")

# Take the top-|MARKED| most frequent outcomes as Grover's answer set.
top_outcomes = sorted(observed_counts, key=lambda v: -observed_counts[v])[:M]
quantum_answer = set(top_outcomes)

# Also check that the marked outcomes collectively carry most of the
# probability mass (amplification actually happened).
marked_mass = sum(observed_counts.get(v, 0) for v in MARKED)
marked_fraction = marked_mass / SHOTS
print(f"Fraction of shots landing on a prime (marked) state: {marked_fraction:.3f}")


# ---------------------------------------------------------------------
# 4. Compare quantum result to the classical answer and report.
# ---------------------------------------------------------------------
verified = (quantum_answer == MARKED) and (marked_fraction > 0.75)

print(f"Classical marked set : {sorted(MARKED)}")
print(f"Quantum top-{M} outcomes: {sorted(quantum_answer)}")

if verified:
    print("PASS")
else:
    print("FAIL")
