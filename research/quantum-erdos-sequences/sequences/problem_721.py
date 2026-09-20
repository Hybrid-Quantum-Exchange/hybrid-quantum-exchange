"""
Erdos problem #721 -- quantum-testable instance.

Source: manman4/erdosproblems data/problems.yaml, entry "number: '721'".
That entry's tags are ["number theory", "additive combinatorics", "ramsey
theory"] and its OEIS id is A171081: the two-color van der Waerden numbers
w(3, n) -- the minimum N such that every 2-coloring of {1, ..., N} contains
a monochromatic arithmetic progression of length 3 (a mono "AP3"). OEIS
A171081 gives w(3, 2) = 9.

Classical property tested (computed from first principles below, not copied
from OEIS): for N = 8 = w(3, 2) - 1, there EXISTS a 2-coloring of
{1, ..., 8} with no monochromatic 3-term arithmetic progression (this is
exactly what "w(3,2) = 9" as opposed to "w(3,2) <= 8" asserts: 8 is not yet
forced). The script:

  1. Classically enumerates all 12 length-3 arithmetic progressions inside
     {1, ..., 8} and, by brute force over all 2^8 = 256 colorings, finds the
     set of "good" colorings (no monochromatic AP3 among the 12 triples).
     This is done honestly by direct enumeration, not asserted.
  2. Builds a genuine Grover search circuit over 8 qubits (one qubit per
     integer 1..8, computational basis value = its color) whose oracle
     phase-flips exactly the good colorings identified in step 1, and whose
     diffuser is the standard Grover diffusion operator. The number of
     Grover iterations is chosen from the classically-known count of good
     states (6) and total search space (256): floor(pi/4 * sqrt(256/6)).
  3. Runs the circuit on the ideal AerSimulator, measures, and takes the
     most frequently sampled bitstring.
  4. PASS iff that bitstring, checked against the SAME classical AP3
     predicate (recomputed independently at verification time), is indeed a
     coloring with no monochromatic 3-term AP -- i.e. the quantum search
     found a genuine witness for w(3,2) > 8, consistent with the OEIS value
     w(3,2) = 9.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N = 8  # instance size: w(3,2) - 1, per OEIS A171081 (w(3,2) = 9)


def arithmetic_progressions(n):
    """All (a, a+d, a+2d) with 1 <= a < a+d < a+2d <= n."""
    triples = []
    for d in range(1, n):
        a = 1
        while a + 2 * d <= n:
            triples.append((a, a + d, a + 2 * d))
            a += 1
    return triples


TRIPLES = arithmetic_progressions(N)
assert len(TRIPLES) == 12, "sanity check on the classical enumeration"


def has_mono_ap3(bits, triples):
    """bits[i] is the color (0/1) of integer i+1. True if some triple is monochromatic."""
    for (a, b, c) in triples:
        va, vb, vc = bits[a - 1], bits[b - 1], bits[c - 1]
        if va == vb == vc:
            return True
    return False


def bits_of(x, n):
    return [(x >> i) & 1 for i in range(n)]


# Step 1: classical brute force over all 2^N colorings.
good_states = []
for x in range(2 ** N):
    bits = bits_of(x, N)
    if not has_mono_ap3(bits, TRIPLES):
        good_states.append(x)

num_good = len(good_states)
print(f"Classical brute force: {num_good} good colorings out of {2 ** N} "
      f"(no monochromatic AP3 among {len(TRIPLES)} triples in 1..{N})")
assert num_good > 0, "w(3,2) would have to be <= 8 if this were empty -- contradicts A171081"


# Step 2: build the Grover oracle that phase-flips exactly `good_states`.
def add_marker(qc, state, n):
    """Phase-flip the computational basis state `state` (n-bit int, qubit i <-> bit i)."""
    bits = bits_of(state, n)
    zero_qubits = [i for i, b in enumerate(bits) if b == 0]
    if zero_qubits:
        qc.x(zero_qubits)
    # multi-controlled Z on qubit n-1, controlled by qubits 0..n-2, via H-MCX-H
    controls = list(range(n - 1))
    target = n - 1
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    if zero_qubits:
        qc.x(zero_qubits)


def build_oracle(states, n):
    qc = QuantumCircuit(n, name="oracle")
    for s in states:
        add_marker(qc, s, n)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(good_states, N)
diffuser = build_diffuser(N)

iterations = max(1, math.floor((math.pi / 4) * math.sqrt((2 ** N) / num_good)))
print(f"Grover iterations: {iterations} (M={num_good}, search space={2 ** N})")

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N))
    qc.append(diffuser.to_gate(), range(N))
qc.measure(range(N), range(N))

# Step 3: run on the ideal AerSimulator.
backend = AerSimulator()
tqc = transpile(qc, backend)
result = backend.run(tqc, shots=2048).result()
counts = result.get_counts()

# Qiskit bit-string order is q_{n-1}...q_0; reverse to get qubit index i at position i.
best_bitstring_qiskit = max(counts, key=counts.get)
best_bitstring = best_bitstring_qiskit[::-1]
measured_int = int(best_bitstring, 2)
measured_bits = [int(b) for b in best_bitstring]

top5 = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
print("Top measured outcomes (qiskit order, counts):", top5)
print(f"Most frequent outcome as coloring bits (position 1..{N}): {measured_bits}, "
      f"integer encoding = {measured_int}")

# Step 4: verify the measured witness against the classical predicate,
# recomputed independently here (not reusing `good_states` directly).
is_valid_witness = (
    not has_mono_ap3(measured_bits, arithmetic_progressions(N))
    and measured_int in good_states
)

print(f"Classically known good colorings (witnesses that w(3,2) > {N}): {good_states}")
print(f"Measured witness is a genuine no-mono-AP3 coloring: {is_valid_witness}")

if is_valid_witness:
    print("PASS")
else:
    print("FAIL")
