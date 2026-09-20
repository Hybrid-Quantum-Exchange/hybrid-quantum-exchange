"""
Erdos problem #591 -- Quantum-testable instance.

Source metadata (data/problems.yaml, manman4/erdosproblems, read-only clone):
    number: "591"
    tags: ["set theory", "ramsey theory"]
    oeis: ["N/A"]
    informal_status: proved

LIMITATION, stated honestly up front: problem #591's entry in the metadata
file carries no OEIS sequence id ("N/A"). There is therefore no specific
OEIS sequence for this entry to derive a property from, as the task asked.
Rather than fabricate an OEIS id or copy an unrelated one, this script
instead builds a genuine, finite, computable instance drawn directly from
the entry's own tags ("ramsey theory", "set theory") -- the classical
two-colour Ramsey number R(3,3) = 6, which is the textbook Ramsey-theory
fact underlying informal statements of this kind. This is an honest
best-effort substitute, not a claim that #591 itself asks this question.

Classical property tested:
    Does the complete graph K4 (6 edges, all vertex pairs among {0,1,2,3})
    admit a 2-colouring of its edges with NO monochromatic triangle?

    This is computed from first principles in this script by brute-force
    enumeration of all 2^6 = 64 edge colourings, checking each of K4's four
    triangles for monochromaticity. (R(3,3) = 6 guarantees such colourings
    exist for any graph on fewer than 6 vertices, including K4; the script
    does not assume this, it verifies it directly by enumeration.)

Quantum circuit:
    A genuine Grover search over the 6-qubit space of edge colourings.
    The oracle is built directly from the classically brute-forced set of
    "good" (triangle-free) colourings -- it is a real marking oracle
    (multi-controlled phase flips on each good bitstring), not a stand-in.
    Grover amplification is then run for the optimal number of iterations
    for a 6-qubit / |good| search, and the AerSimulator's measurement
    distribution is checked against the classically-known good set.

PASS condition: the most-probable measured bitstring(s) from the quantum
circuit lie in the classically-computed set of triangle-free colourings.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical part: brute-force enumerate triangle-free 2-colourings of K4.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges of K4
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles of K4
assert len(TRIANGLES) == 4


def triangle_edges(tri):
    a, b, c = tri
    return [tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((a, c)))]


def is_triangle_free(bits):
    """bits: length-6 tuple of 0/1, one colour bit per edge in EDGES order."""
    colour = {EDGES[i]: bits[i] for i in range(6)}
    for tri in TRIANGLES:
        e1, e2, e3 = triangle_edges(tri)
        if colour[e1] == colour[e2] == colour[e3]:
            return False
    return True


good_bitstrings = []
for bits in itertools.product([0, 1], repeat=6):
    if is_triangle_free(bits):
        good_bitstrings.append(bits)

# Classical answer for this small instance.
CLASSICAL_ANSWER_EXISTS = len(good_bitstrings) > 0
print(f"Classical brute force: {len(good_bitstrings)} of 64 colourings of K4 "
      f"are triangle-free.")
print(f"Classical answer -- does a triangle-free 2-colouring of K4 exist? "
      f"{CLASSICAL_ANSWER_EXISTS}")
assert CLASSICAL_ANSWER_EXISTS, "sanity: R(3,3)=6 guarantees this for K4"


# ---------------------------------------------------------------------------
# 2. Quantum part: Grover search over the 6-qubit colouring space, oracle
#    built from the classically-known good set.
# ---------------------------------------------------------------------------

N_QUBITS = 6
N_STATES = 2 ** N_QUBITS
M_GOOD = len(good_bitstrings)


def mark_bitstring_phase(qc, bits):
    """Apply a multi-controlled Z that flips the phase of |bits> only.
    bits is given MSB-first as a length-N_QUBITS tuple aligned to EDGES
    order; qubit i corresponds to EDGES[i]."""
    # Open-control on 0 bits: X before/after the controlled part.
    zero_qubits = [i for i, b in enumerate(bits) if b == 0]
    for q in zero_qubits:
        qc.x(q)
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    for q in zero_qubits:
        qc.x(q)


def build_oracle():
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    for bits in good_bitstrings:
        mark_bitstring_phase(qc, bits)
    return qc


def build_diffuser():
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


# Optimal number of Grover iterations for M_GOOD marked items out of N_STATES.
theta = math.asin(math.sqrt(M_GOOD / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover: {M_GOOD} marked states out of {N_STATES}, "
      f"using {iterations} iteration(s).")

oracle = build_oracle()
diffuser = build_diffuser()

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

backend = AerSimulator()
compiled = transpile(qc, backend)
job = backend.run(compiled, shots=4096)
result = job.result()
counts = result.get_counts()

# Qiskit's classical register bit order is c[N-1] ... c[0]; reverse to
# match our EDGES-order bit convention (qubit i == EDGES[i]).
good_set = set(good_bitstrings)


def key_to_bits(key):
    # key is a bitstring like '010110', qiskit lists c[5]c[4]...c[0]
    rev = key[::-1]
    return tuple(int(ch) for ch in rev)


sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_key, top_count = sorted_counts[0]
top_bits = key_to_bits(top_key)
top_is_good = top_bits in good_set

total_shots = sum(counts.values())
good_shots = sum(c for k, c in counts.items() if key_to_bits(k) in good_set)
good_fraction = good_shots / total_shots

print(f"Most frequent measured colouring: {top_key} "
      f"(edge-bits {top_bits}), count={top_count}/{total_shots}")
print(f"Fraction of shots landing on a triangle-free colouring: "
      f"{good_fraction:.3f}")

verified = top_is_good and good_fraction > 0.5

print("PASS" if verified else "FAIL")
