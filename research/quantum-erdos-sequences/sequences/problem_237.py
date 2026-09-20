#!/usr/bin/env python3
"""
Erdos problem #237 -- quantum-testable companion script.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
"number: 237"): prize=no, status="proved (Lean)", tags=["number theory",
"primes"], oeis=["N/A"].

LIMITATION, stated honestly up front: problem #237 carries NO OEIS sequence
id in the source metadata (oeis: ["N/A"]), and the source gives no numeric
formula or finite instance to target directly. There is therefore no single
"OEIS term" this script can honestly claim to reproduce from problem #237's
own data. Per the task's fallback instructions, this is the best-effort
substitute: the metadata's only concrete mathematical content is the tag
"primes", so this script builds a genuine, self-contained finite/computable
number-theory property in that spirit -- primality over a small finite
range -- and tests it with a real Grover-search quantum circuit rather than
inventing or copying an unrelated OEIS value.

Classical property tested
--------------------------
Search space: integers 0..15 (4 qubits).
Marked set:   the primes in that range, i.e. {n in [0,16) : n is prime}.
This is computed from first principles in `is_prime()` / `classical_primes()`
below -- no literature value is copied.

For a 4-qubit search space, Grover's algorithm is run with an oracle that
flags exactly the classically-computed prime states, using the standard
optimal number of Grover iterations for the marked-count computed here. The
resulting most-probable measured state is compared against the classically
computed prime set: PASS if the top measured outcome(s) are all primes.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS lookup)
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # search space 0..15

classical_primes = sorted(n for n in range(N) if is_prime(n))
print(f"Classical property: primes in [0, {N}) = {classical_primes}")


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classical prime states
# ---------------------------------------------------------------------------

def bits_of(n: int, width: int):
    return [(n >> i) & 1 for i in range(width)]


def build_oracle(marked, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = bits_of(m, n_qubits)
        # flip qubits that are 0 in this marked state, so the all-ones
        # pattern conditions the multi-controlled Z
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
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


def grover_circuit(marked, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, range(n_qubits), inplace=True)
        qc.compose(diffuser, range(n_qubits), inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# optimal iteration count for Grover: floor(pi/4 * sqrt(N/M))
M = len(classical_primes)
iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover search: N={N} states, M={M} marked (primes), iterations={iterations}")

circuit = grover_circuit(classical_primes, N_QUBITS, iterations)

sim = AerSimulator(method="statevector")
shots = 4096
# Note: deliberately not transpiling here. The circuit is already expressed
# in basic gates (h/x/mcx via compose), and transpile() is free to permute
# qubit layout for optimization, which silently scrambles the correspondence
# between classical measurement bits and the integer values used above.
# Pinning method="statevector" keeps the ideal, exact-amplitude simulator
# path (rather than an automatically-chosen approximate/stabilizer method).
result = sim.run(circuit, shots=shots).result()
counts = result.get_counts()

# get_counts() bitstrings already parse directly (via int(bitstring, 2))
# to the same integer convention used by bits_of()/is_prime() above; this
# was verified empirically against an unmeasured Statevector run of the
# identical oracle+diffuser circuit (which unambiguously amplifies exactly
# the classical prime states) before settling on this parsing.
value_counts = Counter()
for bitstring, c in counts.items():
    value = int(bitstring, 2)
    value_counts[value] += c

sorted_values = value_counts.most_common()
print("Top measured outcomes (value: count):")
for v, c in sorted_values[:8]:
    print(f"  {v:2d} (prime={is_prime(v)}): {c}")

# Amplification success metric: total probability mass landing on any
# classically prime state.
prime_mass = sum(c for v, c in value_counts.items() if v in classical_primes)
prime_fraction = prime_mass / shots
print(f"Fraction of shots landing on a classical prime: {prime_fraction:.3f}")

# The top-1 measured value should itself be prime, and overall amplification
# should clearly exceed the uniform baseline M/N.
baseline = M / N
top_value = sorted_values[0][0]
top_is_prime = is_prime(top_value)
amplified = prime_fraction > baseline * 1.5  # comfortably above uniform baseline

verified = top_is_prime and amplified

print(f"Baseline (uniform) prime fraction would be: {baseline:.3f}")
print(f"Top measured value {top_value} is prime: {top_is_prime}")
print(f"Amplification above baseline: {amplified}")

if verified:
    print("PASS")
else:
    print("FAIL")
