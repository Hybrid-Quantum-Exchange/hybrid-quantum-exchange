"""
Erdos problem #545 (erdosproblems.com), quantum-testable instance.

Metadata (from erdosproblems.com data, problem 545): tags ["graph theory",
"ramsey theory"], OEIS id A059442 -- "Array of Ramsey numbers R(n,k)
(n >= 2, k >= 2) read by antidiagonals" (T(3,3) = R(3,3) = 6 is a term of
this array; it is one of the very few off-diagonal/diagonal Ramsey numbers
known exactly).

Classical property tested (derived here, not copied from OEIS):
  R(3,3) = 6 means: (a) every 2-coloring of the edges of the complete graph
  K6 contains a monochromatic triangle, and (b) K5 does NOT force one --
  i.e. there EXISTS a 2-coloring of K5's edges with no monochromatic
  triangle. Property (b) is what makes R(3,3) > 5, hence R(3,3) = 6 rather
  than something smaller.

  To keep the quantum search space small (few qubits), this script tests
  the same existence statement one graph size down, on K4 (6 edges, 4
  triangles), which is implied by R(3,3) = 6 (if even K4, a strict
  subgraph situation, forces a mono triangle for every coloring, R(3,3)
  could not be 6). Concretely:

      PROPERTY: there exists a 2-coloring of the 6 edges of K4 such that
      none of its 4 triangles is monochromatic (all-same-color).

  The classical answer is computed from first principles in this script
  by brute-force enumeration over all 2^6 = 64 edge-colorings of K4 (no
  OEIS value is looked up or hard-coded): 18 of the 64 colorings satisfy
  the property. The full set of valid colorings is computed classically
  and used to build a Grover oracle; the same 64-state brute force is also
  used, independently, to verify the quantum output.

Quantum circuit:
  6 qubits represent the color (0/1) of each of K4's 6 edges. A Grover
  search runs over this 6-qubit register with an oracle that phase-flips
  exactly the basis states corresponding to a valid (mono-triangle-free)
  coloring -- the oracle is built directly from the classically-computed
  valid set, using the standard X...MCZ...X marking construction per
  target state. The number of Grover iterations is chosen from the known
  count of solutions (18 out of 64). After running on the ideal
  AerSimulator (statevector-backed sampler), the most frequently measured
  bitstring is checked against the classical valid-coloring predicate
  (not just membership in the target list) to confirm it is genuinely a
  mono-triangle-free coloring of K4.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical setup: K4's edges and triangles, and the property predicate.
# ---------------------------------------------------------------------------

NUM_VERTICES = 4
EDGES = list(itertools.combinations(range(NUM_VERTICES), 2))   # 6 edges
NUM_EDGES = len(EDGES)
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(range(NUM_VERTICES), 3))  # 4 triangles
TRIANGLE_EDGE_TRIPLES = [
    (EDGE_INDEX[(a, b)], EDGE_INDEX[(a, c)], EDGE_INDEX[(b, c)])
    for (a, b, c) in TRIANGLES
]

assert NUM_EDGES == 6 and len(TRIANGLES) == 4


def bits_of(mask, width):
    return [(mask >> i) & 1 for i in range(width)]


def is_mono_triangle_free(mask):
    """True iff the edge-coloring `mask` (bit i = color of EDGES[i]) has no
    monochromatic triangle among K4's 4 triangles."""
    bits = bits_of(mask, NUM_EDGES)
    for (i, j, k) in TRIANGLE_EDGE_TRIPLES:
        if bits[i] == bits[j] == bits[k]:
            return False
    return True


# ---------------------------------------------------------------------------
# 2. Classical brute force: compute the full valid set (from first principles).
# ---------------------------------------------------------------------------

VALID_COLORINGS = [m for m in range(2 ** NUM_EDGES) if is_mono_triangle_free(m)]
NUM_SOLUTIONS = len(VALID_COLORINGS)

print(f"K4 has {NUM_EDGES} edges and {len(TRIANGLES)} triangles.")
print(f"Classical brute force over {2 ** NUM_EDGES} colorings found "
      f"{NUM_SOLUTIONS} mono-triangle-free colorings (property: EXISTS such "
      f"a coloring -> classical answer is TRUE, since {NUM_SOLUTIONS} > 0).")
assert NUM_SOLUTIONS > 0, "classical answer must be TRUE for this to be a valid Grover instance"


# ---------------------------------------------------------------------------
# 3. Quantum circuit: Grover search over the 6-qubit coloring register.
# ---------------------------------------------------------------------------

def mark_state(qc, mask, n):
    """Phase-flip the single computational basis state `mask` (n qubits),
    via X-sandwiched multi-controlled-Z (an MCX with phase target trick)."""
    bits = bits_of(mask, n)
    zero_positions = [q for q in range(n) if bits[q] == 0]

    for q in zero_positions:
        qc.x(q)

    if n == 1:
        qc.z(0)
    else:
        # multi-controlled Z on all n qubits: controls = first n-1, target = last,
        # with target put into |-> convention via H-MCX-H.
        qc.h(n - 1)
        qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
        qc.h(n - 1)

    for q in zero_positions:
        qc.x(q)


def build_oracle(n, targets):
    qc = QuantumCircuit(n, name="oracle")
    for mask in targets:
        mark_state(qc, mask, n)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


n = NUM_EDGES
oracle = build_oracle(n, VALID_COLORINGS)
diffuser = build_diffuser(n)

# Optimal number of Grover iterations for NUM_SOLUTIONS out of 2**n states.
theta = math.asin(math.sqrt(NUM_SOLUTIONS / 2 ** n))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations: {iterations} (solutions={NUM_SOLUTIONS}, space=2^{n})")

qc = QuantumCircuit(n, n)
qc.h(range(n))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(n))
    qc.append(diffuser.to_instruction(), range(n))
qc.measure(range(n), range(n))
qc = qc.decompose().decompose().decompose()


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
result = backend.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-bit string is written with qubit (n-1) first; convert
# back to our mask convention (bit i = qubit i, i.e. bit 0 is the rightmost
# character, which is Qiskit's own convention for a single classical register).
best_bitstring = max(counts, key=counts.get)
measured_mask = int(best_bitstring, 2)
measured_probability = counts[best_bitstring] / shots

print(f"Most frequent measured state: {best_bitstring} "
      f"(mask={measured_mask}), probability={measured_probability:.3f} "
      f"over {shots} shots")

# Sanity: total probability mass landing on ANY valid coloring should be
# strongly amplified above the uniform baseline (NUM_SOLUTIONS / 2**n).
valid_mass = sum(c for bs, c in counts.items() if int(bs, 2) in VALID_COLORINGS) / shots
baseline = NUM_SOLUTIONS / 2 ** n
print(f"Total measured probability on valid colorings: {valid_mass:.3f} "
      f"(uniform baseline would be {baseline:.3f})")


# ---------------------------------------------------------------------------
# 5. Verify quantum result against the classical answer.
# ---------------------------------------------------------------------------

quantum_found_valid = is_mono_triangle_free(measured_mask)
amplification_ok = valid_mass > 3 * baseline

verified = quantum_found_valid and amplification_ok

print()
if verified:
    print("PASS: Grover search on the ideal AerSimulator returned a "
          "mono-triangle-free 2-coloring of K4's edges, confirmed by "
          "independent classical re-checking, and search probability was "
          "amplified well above the uniform baseline -- matching the "
          "classically brute-forced answer (EXISTS such a coloring).")
else:
    print("FAIL: quantum result did not match the classical answer.")
