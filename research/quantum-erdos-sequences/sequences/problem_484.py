"""
Erdos problem #484 -- quantum-testable instance
=================================================

Source: erdosproblems.com problem #484 (data/problems.yaml entry
`number: "484"`), tags ["number theory", "additive combinatorics",
"ramsey theory"], informal_status "proved (Lean)".

LIMITATION (read before trusting the "verified" claim below): the
problems.yaml record for #484 lists `oeis: ["N/A"]` -- there is no OEIS
sequence attached to this problem, and no problem statement text is
available in the read-only clone used to build this library entry. So
this script cannot test a property *of Erdos problem #484's own
sequence*, because it has none on record here. What follows is instead
an honest, self-contained quantum test of a small, finite, genuinely
computable instance of the *kind* of statement #484's tags point at --
a monochromatic Schur triple (Ramsey-type additive-combinatorics)
question closely related to Schur's theorem -- chosen because it is
small enough for a real Grover search circuit and its answer is
checked from first principles in this script, not copied from a
lookup table. Treat this as a best-effort stand-in for #484, not a
verification of #484 itself.

The classical property tested
------------------------------
Take the 4-element ground set {1, 2, 3, 4}. A 2-colouring is a function
colour: {1,2,3,4} -> {0,1}, encoded as a 4-bit string b3 b2 b1 b0 where
bit i is the colour of element i+1. A Schur triple is (x, y, z) with
x <= y and x + y = z, all in {1,2,3,4}; there are exactly four of them:
(1,1,2), (1,2,3), (1,3,4), (2,2,4). A colouring is "valid" iff it
contains NO monochromatic Schur triple (i.e. no triple with
colour(x) == colour(y) == colour(z)).

This script:
  1. Enumerates all 16 colourings classically and computes the exact
     set V of valid (Schur-triple-avoiding) colourings and its size M.
  2. Builds a genuine Grover search circuit over the 4 colour qubits
     whose oracle marks exactly the bitstrings in V (via an explicit
     diagonal +/-1 phase oracle derived from V, not hard-coded from
     any external answer), with the standard Grover diffuser, run for
     the correct integer number of iterations for N=16, M=|V|.
  3. Runs the circuit on the ideal AerSimulator and checks that the
     search concentrates its measurement probability on exactly the
     classically-computed set V.
  4. Prints PASS if the quantum search recovers the classical answer,
     FAIL otherwise.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------

ELEMENTS = [1, 2, 3, 4]
N_QUBITS = 4
N_STATES = 2 ** N_QUBITS  # 16 colourings

SCHUR_TRIPLES = []
for x in ELEMENTS:
    for y in ELEMENTS:
        if x <= y:
            z = x + y
            if z in ELEMENTS:
                SCHUR_TRIPLES.append((x, y, z))

assert SCHUR_TRIPLES == [(1, 1, 2), (1, 2, 3), (1, 3, 4), (2, 2, 4)]


def colour_of(bitstring_int, element):
    """Colour (0/1) of `element` (1..4) under the colouring encoded by
    `bitstring_int`, where bit (element-1) of the integer is the colour."""
    return (bitstring_int >> (element - 1)) & 1


def is_valid_colouring(bitstring_int):
    for (x, y, z) in SCHUR_TRIPLES:
        cx, cy, cz = colour_of(bitstring_int, x), colour_of(bitstring_int, y), colour_of(bitstring_int, z)
        if cx == cy == cz:
            return False
    return True


VALID_SET = sorted(c for c in range(N_STATES) if is_valid_colouring(c))
M = len(VALID_SET)

print(f"Ground set: {ELEMENTS}")
print(f"Schur triples (x,y,z) with x+y=z: {SCHUR_TRIPLES}")
print(f"Total colourings N = {N_STATES}")
print(f"Classically valid (Schur-triple-free) colourings V = {VALID_SET}  (M = {M})")

if M == 0 or M == N_STATES:
    raise SystemExit(
        "Degenerate instance (M=0 or M=N): Grover search is not meaningful here; "
        "this should not happen for this fixed 4-element instance."
    )

# ---------------------------------------------------------------------
# 2. Build a genuine Grover search circuit whose oracle marks exactly
#    VALID_SET, derived programmatically from the classical computation
#    above (not hand-copied from any external source).
# ---------------------------------------------------------------------

# Diagonal phase oracle: -1 on marked (valid) states, +1 elsewhere.
# Qiskit's little-endian convention: qubit 0 is the least-significant
# bit, matching how we defined colour_of() above (bit i-1 = element i),
# so basis-state index == bitstring_int directly.
diag = [(-1.0 if i in VALID_SET else 1.0) for i in range(N_STATES)]
oracle_gate = DiagonalGate(diag)

# Standard Grover diffuser (inversion about the mean) on N_QUBITS qubits.
def diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


num_iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M)))
print(f"Grover iterations used: {num_iterations} (N={N_STATES}, M={M})")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(num_iterations):
    qc.append(oracle_gate, range(N_QUBITS))
    qc.append(diffuser(N_QUBITS).to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports counts as bitstrings 'b3 b2 b1 b0' (qubit N-1 .. qubit 0);
# convert back to the same integer encoding used classically.
int_counts = {}
for bitstring, n in counts.items():
    value = int(bitstring, 2)
    int_counts[value] = int_counts.get(value, 0) + n

print("Measured outcome counts (as integers):", dict(sorted(int_counts.items(), key=lambda kv: -kv[1])))

# ---------------------------------------------------------------------
# 4. Verify: the measurement probability mass on VALID_SET should be
#    the large majority (Grover-amplified), matching the classical set.
# ---------------------------------------------------------------------

hits_on_valid = sum(n for v, n in int_counts.items() if v in VALID_SET)
prob_on_valid = hits_on_valid / SHOTS

# Also require that the single most frequent outcome is itself a
# classically-valid colouring (the search actually found a solution).
top_outcome = max(int_counts.items(), key=lambda kv: kv[1])[0]

print(f"P(measured colouring in classical VALID_SET) = {prob_on_valid:.4f}")
print(f"Most frequent measured colouring: {top_outcome:04b}  (in VALID_SET: {top_outcome in VALID_SET})")

verified = (prob_on_valid > 0.90) and (top_outcome in VALID_SET)

if verified:
    print("PASS")
else:
    print("FAIL")
