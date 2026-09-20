"""
Erdos problem #713 (from https://github.com/manman4/erdosproblems,
data/problems.yaml entry `number: "713"`).

Problem metadata as recorded there:
  prize: $500
  informal_status: open (last_update 2025-08-31)
  oeis: ["N/A"]   <-- NO OEIS sequence id is associated with this problem
  tags: ["graph theory", "turan number"]

LIMITATION (read before trusting "verified_against_classical"):
Problem #713 has no OEIS id in the source data ("N/A"), so there is no
integer sequence to build a genuine "is n in OEIS sequence Axxxxxx" style
membership/search oracle for, as the task instructions ask for. Rather than
fabricate a fake OEIS-backed property, this script instead builds a REAL,
self-contained instance of the mathematical object named by the problem's
own tags -- a Turan-type extremal graph problem -- and tests a small,
finite, exactly-computable property of it with a genuine Grover search
circuit. This is offered as the best honest attempt for a problem whose
OEIS field is "N/A"; it is not a claim that #713 itself is resolved or
that this circuit "is" sequence data for it.

Classical property under test
------------------------------
Turan's theorem for triangle-free graphs: the maximum number of edges in a
triangle-free graph on n vertices is floor(n^2/4) (Mantel's theorem, the
n=3 case of Turan's theorem, which problem #713's tag "turan number"
refers to).

For n = 4 vertices, K4 has 6 possible edges:
  e0=(0,1) e1=(0,2) e2=(0,3) e3=(1,2) e4=(1,3) e5=(2,3)
and 4 possible triangles: {0,1,2} {0,1,3} {0,2,3} {1,2,3}.

Mantel's bound gives floor(4^2/4) = 4 as the maximum edge count of a
triangle-free graph on 4 vertices. This script:
  1. Brute-forces, classically, over all 2^6 = 64 edge-subsets of K4 to find
     every subset that is BOTH triangle-free AND has exactly 4 edges (i.e.
     every graph that actually attains the Mantel bound). This is the
     "small, finite, computable search space whose answer is a known
     extremal value" the task asks for.
  2. Builds a genuine 6-qubit Grover search circuit whose oracle marks
     exactly those bitstrings (found in step 1) via multi-controlled phase
     gates, runs the standard number of Grover iterations, and samples the
     ideal AerSimulator.
  3. Compares the bitstrings returned with high probability by the quantum
     circuit against the classically-computed solution set from step 1 and
     prints PASS/FAIL.

Dependencies: qiskit, qiskit_aer, numpy only (no pip installs).
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate


# ---------------------------------------------------------------------
# Step 1: classical computation (first principles, no OEIS lookup)
# ---------------------------------------------------------------------

N_VERTICES = 4
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 6 edges
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))  # 4 triangles


def edges_of_triangle(tri):
    a, b, c = tri
    return (EDGE_INDEX[tuple(sorted((a, b)))],
            EDGE_INDEX[tuple(sorted((b, c)))],
            EDGE_INDEX[tuple(sorted((a, c)))])


TRIANGLE_EDGE_TRIPLES = [edges_of_triangle(t) for t in TRIANGLES]


def is_triangle_free(bits):
    """bits: length-6 tuple/list of 0/1, bits[i] = whether EDGES[i] is present."""
    for (i, j, k) in TRIANGLE_EDGE_TRIPLES:
        if bits[i] and bits[j] and bits[k]:
            return False
    return True


mantel_bound = N_VERTICES * N_VERTICES // 4  # floor(n^2/4) = 4 for n=4

classical_solutions = []
for mask in range(64):
    bits = [(mask >> i) & 1 for i in range(6)]
    if sum(bits) == mantel_bound and is_triangle_free(bits):
        classical_solutions.append(mask)

# Sanity: also confirm no triangle-free graph on 4 vertices exceeds the bound.
max_triangle_free_edges = max(
    sum((mask >> i) & 1 for i in range(6))
    for mask in range(64)
    if is_triangle_free([(mask >> i) & 1 for i in range(6)])
)
assert max_triangle_free_edges == mantel_bound, (
    f"Mantel bound mismatch: brute force found max {max_triangle_free_edges}, "
    f"expected {mantel_bound}"
)
assert len(classical_solutions) > 0

print(f"Classical result: Mantel bound for n={N_VERTICES} is {mantel_bound} edges.")
print(f"Classical brute force confirms max triangle-free edge count = "
      f"{max_triangle_free_edges} (matches floor(n^2/4)).")
print(f"Number of triangle-free 4-edge graphs on K4 (bitmask solutions): "
      f"{len(classical_solutions)}")
print("Solution bitmasks (bit i = edge EDGES[i] present):",
      sorted(classical_solutions))


# ---------------------------------------------------------------------
# Step 2: Grover search circuit whose oracle marks exactly those bitmasks
# ---------------------------------------------------------------------

NUM_QUBITS = 6  # one qubit per edge of K4


def mark_bitmask_phase(qc, mask, num_qubits):
    """Apply a multi-controlled Z (phase flip) on |mask> using the standard
    trick: X-gate the 0-bits, apply an (n-1)-controlled Z targeting the last
    qubit, then undo the X-gates."""
    bits = [(mask >> i) & 1 for i in range(num_qubits)]
    zero_positions = [i for i, b in enumerate(bits) if b == 0]
    for i in zero_positions:
        qc.x(i)

    # multi-controlled Z across all num_qubits qubits (phase flip on |11..1>)
    controls = list(range(num_qubits - 1))
    target = num_qubits - 1
    mcz = MCMTGate(ZGate(), num_ctrl_qubits=num_qubits - 1, num_target_qubits=1)
    qc.append(mcz, controls + [target])

    for i in zero_positions:
        qc.x(i)


def build_oracle(num_qubits, solutions):
    qc = QuantumCircuit(num_qubits, name="oracle")
    for mask in solutions:
        mark_bitmask_phase(qc, mask, num_qubits)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    controls = list(range(num_qubits - 1))
    target = num_qubits - 1
    # MCMTGate(ZGate()) already applies a phase flip to |1...1>, so no extra
    # H-sandwich is needed here (that would turn it into a multi-controlled X).
    mcz = MCMTGate(ZGate(), num_ctrl_qubits=num_qubits - 1, num_target_qubits=1)
    qc.append(mcz, controls + [target])
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


N = 2 ** NUM_QUBITS
M = len(classical_solutions)
# standard optimal number of Grover iterations
num_iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))

oracle = build_oracle(NUM_QUBITS, classical_solutions)
diffuser = build_diffuser(NUM_QUBITS)

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(num_iterations):
    qc.append(oracle.to_instruction(), range(NUM_QUBITS))
    qc.append(diffuser.to_instruction(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

print(f"\nGrover circuit: {NUM_QUBITS} qubits, N={N}, M={M} marked states, "
      f"{num_iterations} iteration(s).")

backend = AerSimulator()
transpiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(transpiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit ordering: classical bit c0 (edge 0) is the rightmost character
# of the returned bitstring. Convert each measured bitstring back to our
# edge-index mask convention (bit i <-> EDGES[i]) for comparison.
def bitstring_to_mask(bitstring):
    # bitstring is c5 c4 c3 c2 c1 c0 (qiskit prints MSB first)
    reversed_bits = bitstring[::-1]
    mask = 0
    for i, ch in enumerate(reversed_bits):
        if ch == "1":
            mask |= (1 << i)
    return mask

measured_masks_by_count = {}
for bitstring, cnt in counts.items():
    mask = bitstring_to_mask(bitstring)
    measured_masks_by_count[mask] = measured_masks_by_count.get(mask, 0) + cnt

# Top len(classical_solutions) most-measured masks should be exactly the
# classical solution set.
top_masks = sorted(measured_masks_by_count, key=lambda m: -measured_masks_by_count[m])
top_k = set(top_masks[:len(classical_solutions)])

prob_on_solutions = sum(
    measured_masks_by_count.get(m, 0) for m in classical_solutions
) / SHOTS

print(f"Top measured masks (by count): {top_masks[:len(classical_solutions) + 2]}")
print(f"Fraction of shots landing on a classical solution mask: "
      f"{prob_on_solutions:.4f}")

verified = (top_k == set(classical_solutions)) and (prob_on_solutions > 0.9)

if verified:
    print("\nPASS: quantum Grover search recovered exactly the classically "
          "computed set of maximum (Mantel-bound) triangle-free 4-edge "
          "graphs on K4 with high probability.")
else:
    print("\nFAIL: quantum result did not match the classical computation.")

print(f"\nran_ok=True verified_against_classical={verified}")
