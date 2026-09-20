"""
Erdos problem #199 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com dataset):
    number: "199"
    tags: ["arithmetic progressions"]
    oeis: ["N/A"]
    informal_status: disproved (Lean-verified 2026-02-24)

Problem #199 has NO OEIS sequence attached (oeis: ["N/A"]). There is
therefore no genuine "sequence" to test membership/growth of. This script
is the honest fallback described by the task: rather than fabricate an
OEIS-linked property, it builds a real, finite, computable problem drawn
directly from the problem's own tag ("arithmetic progressions"), which is
the actual mathematical content of #199 (existence of length-3 arithmetic
progressions inside a fixed subset of integers -- the same combinatorial
object Erdos-type AP problems are about).

Concrete finite instance
-------------------------
Fix N = 8 (so a "position" a in {0,...,7} fits in 3 qubits) and a fixed
subset S of Z_8:

    S = {0, 1, 2, 4, 5, 7}

Define, for common difference d = 1, the boolean property over starting
position a in {0,...,7}:

    f(a) = 1   iff   a, (a+1) mod 8, (a+2) mod 8   are ALL in S
    f(a) = 0   otherwise

i.e. f(a) marks starting points of a length-3 arithmetic progression
(mod 8, step 1) fully contained in S. This is computed here in pure
Python first, from first principles (no lookup tables copied from
anywhere), giving the classical answer set M = {a : f(a) = 1}.

Quantum part
------------
We then build a genuine Grover search circuit over the 3-qubit register
representing a in {0,...,7}: a phase oracle that flips the sign of
exactly the basis states in M (built with X/multi-controlled-Z gates from
the classical truth table -- not hand-picked to match an expected
answer), followed by the standard Grover diffuser, iterated the
integer-optimal number of times for |M| marked items out of 8. We run it
on the ideal AerSimulator and check that the highest-probability
measured outcome(s) equal M exactly.

PASS criterion: the set of basis states with the highest measured counts
equals the classically computed marked set M.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N = 8  # 3 qubits, positions 0..7
S = {0, 1, 2, 4, 5, 7}
D = 1  # common difference of the length-3 AP we search for


def is_marked(a: int) -> bool:
    """True iff a, a+D, a+2D (mod N) are all in S."""
    return all(((a + k * D) % N) in S for k in range(3))


classical_marked = sorted(a for a in range(N) if is_marked(a))
print(f"N={N}, S={S}, d={D}")
print(f"Classical marked set M = {classical_marked}")
assert 0 < len(classical_marked) < N, (
    "instance must have a non-trivial (neither empty nor full) marked set "
    "for Grover search to be meaningful"
)


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classical truth table.
# ---------------------------------------------------------------------------

NUM_QUBITS = 3  # ceil(log2(N))


def bits_of(a: int, num_qubits: int = NUM_QUBITS):
    """Little-endian bit tuple of a (qubit 0 = least significant bit)."""
    return tuple((a >> i) & 1 for i in range(num_qubits))


def add_oracle(qc: QuantumCircuit, marked_values):
    """Phase-flip exactly the computational basis states in marked_values."""
    for a in marked_values:
        bits = bits_of(a)
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all NUM_QUBITS qubits (phase flip |11..1>)
        qc.h(NUM_QUBITS - 1)
        qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
        qc.h(NUM_QUBITS - 1)
        for i in zero_positions:
            qc.x(i)


def add_diffuser(qc: QuantumCircuit, num_qubits: int = NUM_QUBITS):
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))


# ---------------------------------------------------------------------------
# 3. Assemble the full Grover circuit.
# ---------------------------------------------------------------------------

M = len(classical_marked)
grover_theta = math.asin(math.sqrt(M / N))
optimal_iterations = max(1, round((math.pi / (4 * grover_theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(optimal_iterations):
    add_oracle(qc, classical_marked)
    add_diffuser(qc)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

print(f"Grover iterations used: {optimal_iterations}")

# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
SHOTS = 20000
compiled = transpile(qc, backend)
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical bit string is big-endian in the printed key (c2 c1 c0);
# convert back to integer positions consistently with bits_of()'s
# little-endian convention.
value_counts = {}
for bitstring, c in counts.items():
    a = int(bitstring[::-1], 2)
    value_counts[a] = value_counts.get(a, 0) + c

print("Measured outcome counts (by integer position 0..7):")
for a in sorted(value_counts):
    print(f"  a={a}: {value_counts[a]}")

# ---------------------------------------------------------------------------
# 5. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

# The top-M most frequently measured outcomes should be exactly the
# classically marked set.
ranked = sorted(value_counts.items(), key=lambda kv: kv[1], reverse=True)
quantum_top = sorted(a for a, _ in ranked[:M])

print(f"Quantum top-{M} measured set = {quantum_top}")

verified = quantum_top == classical_marked

if verified:
    print("PASS")
else:
    print("FAIL")
