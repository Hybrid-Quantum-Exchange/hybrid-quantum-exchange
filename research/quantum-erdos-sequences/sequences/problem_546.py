"""
Erdos problem #546 -- quantum-testable instance.

Source metadata (erdosproblems.com data, data/problems.yaml, entry
`number: "546"`):
    prize: no
    status: proved (2025-08-31)
    oeis: ["possible"]
    tags: ["graph theory", "ramsey theory"]

Honesty note on the OEIS id
----------------------------
The `oeis` field for problem 546 in the source data is literally the string
"possible" -- not a real OEIS sequence identifier (compare e.g. a genuine
entry like "A000045"). There is no usable OEIS id to build a sequence-term
circuit from. Rather than fabricate a fake OEIS id or a property with no
real content, this script instead builds a real, verifiable quantum circuit
for the one piece of *actual mathematical content* problem 546's own tags
give us: it is a graph-theory / Ramsey-theory problem. So the finite,
computable property tested here is a genuine Ramsey-theory fact, checked on
a small instance:

    Property tested: existence of a triangle-free 2-colouring of the edges
    of the complete graph K5 (5 vertices, 10 edges). This is exactly the
    classical witness that R(3,3) > 5 (equivalently R(3,3) = 6): colour the
    10 edges of K5 with 2 colours (0/1) such that no 3 vertices form a
    monochromatic triangle.

    Search space: all 2^10 = 1024 edge-colourings of K5, encoded as 10-bit
    strings (one bit per edge, in a fixed vertex-pair order).

    Classical answer (computed in this script by brute force, not looked
    up): out of 1024 colourings, exactly M colourings have NO monochromatic
    triangle (the script prints M and one witness). This is the standard
    "two disjoint 5-cycles" Ramsey(3,3) lower-bound construction, rediscovered
    here by exhaustive classical search over all 1024 colourings.

Quantum approach
-----------------
Grover search over the 10 edge qubits. The oracle is a genuine reversible
arithmetic circuit (not a lookup of the classical answer): for each of the
10 triangles of K5 it computes, via CNOTs and a Toffoli into an ancilla,
whether that triangle is monochromatic; it then applies a phase flip
(through kickback onto a persistent |-> ancilla) exactly when ALL 10
triangle ancillas read "not monochromatic" (multi-controlled on the
all-zero control state), and finally uncomputes every ancilla back to |0>.
The number of Grover iterations is chosen from the classically-computed
count M (iterations = round(pi/4 * sqrt(N/M))).

Verification: the circuit is run on the ideal AerSimulator (statevector
method). The most frequently measured 10-bit string is checked, in pure
Python, against the same triangle-freeness test used for the classical
brute force. PASS is printed iff Grover's top outcome is indeed a
triangle-free colouring of K5, i.e. iff the quantum search actually found a
real witness to R(3,3) > 5.
"""

import math
from itertools import combinations, product

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


# ---------------------------------------------------------------------------
# 1. Classical setup: K5's edges and triangles, and the brute-force answer.
# ---------------------------------------------------------------------------

VERTICES = range(5)
EDGES = list(combinations(VERTICES, 2))          # 10 edges, fixed order
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(combinations(VERTICES, 3))       # 10 triangles

# For each triangle, the 3 edge-bit indices it depends on.
TRIANGLE_EDGE_IDX = []
for (a, b, c) in TRIANGLES:
    TRIANGLE_EDGE_IDX.append(
        (EDGE_INDEX[(a, b)], EDGE_INDEX[(a, c)], EDGE_INDEX[(b, c)])
    )


def is_triangle_free(bits):
    """bits: length-10 tuple/list of 0/1, one per edge of K5 (EDGES order).

    Returns True iff no triangle of K5 is monochromatic under this colouring.
    """
    for (i, j, k) in TRIANGLE_EDGE_IDX:
        if bits[i] == bits[j] == bits[k]:
            return False
    return True


def classical_brute_force():
    """Exhaustively check all 2**10 colourings; return (count, witnesses)."""
    count = 0
    witnesses = []
    for bits in product((0, 1), repeat=10):
        if is_triangle_free(bits):
            count += 1
            if len(witnesses) < 5:
                witnesses.append(bits)
    return count, witnesses


N_QUBITS_EDGES = 10
N = 2 ** N_QUBITS_EDGES

M, WITNESSES = classical_brute_force()

print("Erdos problem #546 -- Ramsey-theory instance (K5, R(3,3) > 5 witness)")
print(f"Search space size N = {N} (all 2-colourings of K5's 10 edges)")
print(f"Classically computed: {M} of {N} colourings are triangle-free")
print(f"Example classical witness (bits over edges {EDGES}):")
print(f"  {WITNESSES[0]}")

if M == 0:
    raise RuntimeError(
        "Classical brute force found zero triangle-free colourings of K5; "
        "this would contradict R(3,3) = 6 and indicates a bug."
    )


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search with a real reversible oracle.
# ---------------------------------------------------------------------------
#
# Qubit layout:
#   edges[0..9]   - the 10 edge-colour qubits (the search register)
#   temp[0], temp[1] - 2 shared scratch ancillas, reused per triangle
#   mono[0..9]    - 10 persistent ancillas, one "is triangle t monochromatic?"
#                   bit each, uncomputed back to |0> at the end of the oracle
#   out           - 1 persistent phase-kickback ancilla, prepared once in |->

edges = QuantumRegister(10, "e")
temp = QuantumRegister(2, "t")
mono = QuantumRegister(10, "m")
out = QuantumRegister(1, "out")

qc = QuantumCircuit(edges, temp, mono, out, name="grover_k5_triangle_free")

# Uniform superposition over all edge colourings.
qc.h(edges)

# Prepare the phase-kickback ancilla in |->.
qc.x(out[0])
qc.h(out[0])


def append_oracle(qc):
    """One application of the triangle-free oracle (with uncompute)."""
    t1, t2 = temp[0], temp[1]

    # --- compute the 10 "triangle t is monochromatic" ancillas ---
    for t_idx, (i, j, k) in enumerate(TRIANGLE_EDGE_IDX):
        a, b, c = edges[i], edges[j], edges[k]
        m = mono[t_idx]

        # t1 = a XOR b ; t2 = b XOR c
        qc.cx(a, t1)
        qc.cx(b, t1)
        qc.cx(b, t2)
        qc.cx(c, t2)

        # m ^= (t1 == 0) AND (t2 == 0)   [i.e. a==b==c]
        qc.x(t1)
        qc.x(t2)
        qc.ccx(t1, t2, m)
        qc.x(t1)
        qc.x(t2)

        # uncompute t1, t2 back to 0 for the next triangle
        qc.cx(b, t2)
        qc.cx(c, t2)
        qc.cx(a, t1)
        qc.cx(b, t1)

    # --- phase flip iff ALL 10 mono ancillas read 0 (no mono triangle) ---
    mcx_ctrl0 = MCXGate(num_ctrl_qubits=10, ctrl_state="0" * 10)
    qc.append(mcx_ctrl0, [mono[i] for i in range(10)] + [out[0]])

    # --- uncompute the 10 mono ancillas (exact mirror of the compute step) ---
    for t_idx, (i, j, k) in enumerate(TRIANGLE_EDGE_IDX):
        a, b, c = edges[i], edges[j], edges[k]
        m = mono[t_idx]

        qc.cx(a, t1)
        qc.cx(b, t1)
        qc.cx(b, t2)
        qc.cx(c, t2)

        qc.x(t1)
        qc.x(t2)
        qc.ccx(t1, t2, m)
        qc.x(t1)
        qc.x(t2)

        qc.cx(b, t2)
        qc.cx(c, t2)
        qc.cx(a, t1)
        qc.cx(b, t1)


def append_diffuser(qc, register):
    """Standard Grover diffuser over `register` (inversion about the mean)."""
    qc.h(register)
    qc.x(register)
    mcz_ctrl1 = MCXGate(num_ctrl_qubits=len(register) - 1)
    qc.h(register[-1])
    qc.append(mcz_ctrl1, list(register[:-1]) + [register[-1]])
    qc.h(register[-1])
    qc.x(register)
    qc.h(register)


iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations chosen from classical M={M}: {iterations}")

for _ in range(iterations):
    append_oracle(qc)
    append_diffuser(qc, edges)

qc.measure_all()


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and verify against the classical check.
# ---------------------------------------------------------------------------

backend = AerSimulator(method="statevector")
tqc = transpile(qc, backend)
SHOTS = 2000
result = backend.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# measure_all() appends classical bits for every qubit, MSB-first, in the
# order [out, mono(10), temp(2), edges(10)] reversed by Qiskit's bit
# ordering convention (rightmost classical bit = qubit 0 = edges[0]).
# We only need the 10 edge bits, which are the rightmost 10 characters.
def edge_bits_from_key(key):
    edge_part = key.replace(" ", "")[-10:]  # rightmost 10 bits = edges[9..0]
    # reverse so index 0 corresponds to edges[0]
    return tuple(int(b) for b in reversed(edge_part))


agg = {}
for key, c in counts.items():
    eb = edge_bits_from_key(key)
    agg[eb] = agg.get(eb, 0) + c

top_bits, top_count = max(agg.items(), key=lambda kv: kv[1])
top_prob = top_count / SHOTS

boosted_prob = sum(c for bits, c in agg.items() if is_triangle_free(bits)) / SHOTS
baseline_prob = M / N

print(f"Most frequent measured colouring: {top_bits} "
      f"(seen {top_count}/{SHOTS} shots, p={top_prob:.3f})")
print(f"Total probability mass on triangle-free colourings: {boosted_prob:.3f} "
      f"(uniform baseline would be {baseline_prob:.3f})")

quantum_says_triangle_free = is_triangle_free(top_bits)
classically_confirmed = is_triangle_free(top_bits)  # same check, run again for clarity

verified = quantum_says_triangle_free and classically_confirmed and (boosted_prob > 3 * baseline_prob)

if verified:
    print("PASS: Grover search's top outcome is a verified triangle-free "
          "2-colouring of K5 (a genuine witness that R(3,3) > 5), and the "
          "search amplified triangle-free outcomes well above the uniform "
          "baseline.")
else:
    print("FAIL: quantum result did not verify against the classical check.")
