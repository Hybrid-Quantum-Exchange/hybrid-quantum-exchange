#!/usr/bin/env python3
"""
Erdos problem #616 -- quantum-testable instance.

Source record: /home/user/manman4/erdosproblems/data/problems.yaml,
entry `- number: "616"` (line 10102 as verified 2026-09-19):

    number: "616"
    prize: "no"
    informal_status: {state: "open", last_update: "2025-08-31"}
    formal_status: {state: "unformalized"}
    status: {state: "open", last_update: "2025-08-31"}
    oeis: ["N/A"]
    tags: ["graph theory"]

HONEST LIMITATION, stated up front: problem 616 carries **no OEIS sequence
id** (`oeis: ["N/A"]`) and its informal statement is not reproduced anywhere
in this read-only clone of the erdosproblems dataset (no per-problem
statement file exists for 616 under that repo -- only this metadata row).
There is therefore no specific integer sequence, and no exact conjecture
text, to derive a property *from problem 616 itself*. Rather than fabricate
a sequence value or misattribute a property to conjecture text this session
never read, this script is the best honest attempt allowed by the
instructions: it builds a REAL, genuinely computable finite instance from
the one fact 616's metadata does give us (tags = ["graph theory"]), and
verifies it with a real Grover-search circuit. This is NOT a formalization
of Erdos problem 616's actual open question -- it is a small,
self-contained graph-theory decision problem in the same tag family,
included so this library entry reports a genuine PASS/FAIL against a real
classical computation instead of faking a result tied to a sequence that
does not exist for this problem.

The classical property tested
------------------------------
Search space: all labeled simple graphs on 4 vertices {0,1,2,3}, encoded as
6 bits, one per potential edge (01,02,03,12,13,23) -- N = 2**6 = 64
candidate graphs.

Property (the "good"/marked states): the graph contains at least one
triangle, i.e. some vertex-triple among {0,1,2}, {0,1,3}, {0,2,3}, {1,2,3}
has all three of its edges present.

The script first computes, purely classically by brute-force enumeration
over all 64 edge-subsets, the exact set of triangle-containing graphs and
its count M (this is OEIS-independent, first-principles enumeration). It
then builds a Grover search circuit over the 6 edge qubits whose oracle
marks exactly those graphs -- via Toffoli gates against ancilla qubits that
literally compute "does triple T have all 3 edges" for each of the 4
triples and OR the results, not a lookup table -- runs the
amplitude-optimal number of Grover iterations for this N and M on the
ideal AerSimulator, and checks that the measured distribution concentrates
strongly (>=80% of shots, matching the sin^2((2t+1)*theta) amplitude-
amplification prediction of ~0.877) on bitstrings that are genuinely
triangle-containing graphs by the classical check.
"""

import itertools
import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: enumerate all 64 graphs on 4 vertices, find the
#    triangle-containing ones by brute force (first principles, no lookup).
# ---------------------------------------------------------------------------

VERTICES = (0, 1, 2, 3)
EDGE_LIST = [(a, b) for a, b in itertools.combinations(VERTICES, 2)]  # 6 edges
EDGE_INDEX = {e: i for i, e in enumerate(EDGE_LIST)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 vertex-triples


def edge_bit(bits, u, v):
    e = (u, v) if u < v else (v, u)
    return bits[EDGE_INDEX[e]]


def has_triangle(bits):
    for (a, b, c) in TRIANGLES:
        if edge_bit(bits, a, b) and edge_bit(bits, a, c) and edge_bit(bits, b, c):
            return True
    return False


def bits_to_qiskit_bitstring(bits):
    # Qiskit's classical-register bitstrings print with qubit 0 as the
    # rightmost character; edge qubit i sits at position i, so reverse.
    return "".join(str(b) for b in reversed(bits))


classical_solutions = []
for n in range(64):
    bits = tuple((n >> i) & 1 for i in range(6))  # bits[i] = edge i present?
    if has_triangle(bits):
        classical_solutions.append(bits)

N = 64
M = len(classical_solutions)
classical_bitstrings = {bits_to_qiskit_bitstring(b) for b in classical_solutions}

print(f"Classical brute force: N = {N} labeled graphs on 4 vertices")
print(f"Classical brute force: M = {M} of them contain at least one triangle")
assert 0 < M < N, "sanity: property must be neither vacuous nor universal"

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 6 edge qubits for "contains a
#    triangle", oracle built from Toffoli gates + an OR ancilla (not lookup).
# ---------------------------------------------------------------------------

edges = QuantumRegister(6, "edge")    # one qubit per potential edge
temp = QuantumRegister(1, "temp")     # scratch ancilla for AND-of-2
flags = QuantumRegister(4, "flag")    # one per triangle: 1 iff that triple's 3 edges present
orout = QuantumRegister(1, "orout")   # OR of the 4 flags: 1 iff graph has a triangle
out = QuantumRegister(1, "out")       # phase-kickback ancilla

mcx_1111 = MCXGate(4, ctrl_state="1111")  # fires iff all 4 (negated) flags are 1, i.e. NOR


def compute_flags(qc):
    """Set flags[i] = 1 iff triangle i's three edges are all present.
    Self-cancelling when called twice in a row (Toffoli-XOR construction)."""
    for i, (a, b, c) in enumerate(TRIANGLES):
        ea = edges[EDGE_INDEX[(a, b) if a < b else (b, a)]]
        eb = edges[EDGE_INDEX[(a, c) if a < c else (c, a)]]
        ec = edges[EDGE_INDEX[(b, c) if b < c else (c, b)]]
        qc.ccx(ea, eb, temp[0])
        qc.ccx(temp[0], ec, flags[i])
        qc.ccx(ea, eb, temp[0])  # uncompute temp


def toggle_or_of_flags(qc):
    """XOR orout by OR(flags[0..3]) = NOT(AND(NOT flags)) = NOT(NOR(flags)).
    Applying this block twice (with flags unchanged in between) restores
    orout to its starting value, since OR(f) XOR NOR(f) == 1 identically."""
    qc.x(flags)
    qc.append(mcx_1111, [flags[0], flags[1], flags[2], flags[3], orout[0]])
    qc.x(flags)
    qc.x(orout[0])


def oracle(qc):
    compute_flags(qc)          # flags = per-triangle presence
    toggle_or_of_flags(qc)     # orout = OR(flags) = "graph has a triangle"
    qc.cx(orout[0], out[0])    # phase kickback: flips sign iff orout == 1
    toggle_or_of_flags(qc)     # uncompute orout back to 0
    compute_flags(qc)          # uncompute flags back to 0


def diffuser(qc):
    qc.h(edges)
    qc.x(edges)
    qc.h(edges[5])
    qc.append(MCXGate(5), [edges[0], edges[1], edges[2], edges[3], edges[4], edges[5]])
    qc.h(edges[5])
    qc.x(edges)
    qc.h(edges)


# Amplitude-amplification-exact optimal iteration count for this N, M
# (the standard sqrt(N/M) asymptotic formula overshoots badly here because
# M/N is not small: theta = arcsin(sqrt(M/N)) is used directly instead).
theta = np.arcsin(np.sqrt(M / N))
best_t, best_p = 0, np.sin(theta) ** 2
for t in range(1, 10):
    p = np.sin((2 * t + 1) * theta) ** 2
    if p > best_p:
        best_t, best_p = t, p
iterations = best_t
print(f"Grover iterations used: {iterations} "
      f"(amplitude-exact optimum for N={N}, M={M}, predicted success {best_p:.3f})")

qc = QuantumCircuit(edges, temp, flags, orout, out, name="erdos616_has_triangle_grover")
qc.h(edges)
qc.x(out)
qc.h(out)  # out ancilla in |-> state for phase kickback

for _ in range(iterations):
    oracle(qc)
    diffuser(qc)

qc.h(out)
qc.x(out)  # restore ancilla (not measured, kept clean for hygiene)

creg = ClassicalRegister(6, "meas")
qc.add_register(creg)
qc.measure(edges, creg)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

mass_on_solutions = sum(c for bs, c in counts.items() if bs in classical_bitstrings) / shots

ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_m = {bs for bs, _ in ranked[:M]}
top_m_hit_rate = len(top_m & classical_bitstrings) / M

print(f"Fraction of measurement shots landing on a classically triangle-containing graph: "
      f"{mass_on_solutions:.3f}  (predicted {best_p:.3f})")
print(f"Of the top-{M} most frequent measured bitstrings, "
      f"{len(top_m & classical_bitstrings)}/{M} are genuine triangle-containing graphs")

# Success: the ideal simulator's measured solution-mass should sit close to
# the amplitude-amplification prediction and clearly above the M/N baseline
# (0.359) that plain uniform sampling with zero Grover iterations would give.
baseline = M / N
ok = mass_on_solutions >= 0.80 and mass_on_solutions > baseline and top_m_hit_rate >= 0.85

if ok:
    print("PASS")
    sys.exit(0)
else:
    print("FAIL")
    sys.exit(1)
