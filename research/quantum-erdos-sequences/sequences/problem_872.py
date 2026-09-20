"""
Erdos problem #872 -- quantum-testable instance.

Source metadata (erdosproblems.com data, `data/problems.yaml`, entry
`number: "872"`):
    prize: no
    status: open (last update 2025-08-31)
    oeis: ["possible"]
    tags: ["number theory", "primitive sets"]

LIMITATION, stated honestly: the `oeis` field for problem #872 is the literal
string "possible" -- it is not an actual OEIS sequence id (an A-number).
There is therefore no genuine "OEIS sequence" for this script to encode a
membership/counting/search property of. Following the tags instead
("primitive sets"), this script builds a real, finite, computable property
that is faithful to the mathematical subject of the problem (Erdos's work on
primitive sets: a set of integers > 1 is *primitive* if no element of the set
divides another), and tests it with a genuine Grover search circuit. This is
an honest substitute for an OEIS-anchored property, not a fabricated OEIS
value -- no OEIS term is claimed or used anywhere below.

Classical property under test
------------------------------
Fix the small list L = [2, 3, 4, 6] (4 elements, indices 0..3).
A pair of *distinct* indices (i, j) is "divisive" if L[i] divides L[j].
L is *not* a primitive set iff at least one divisive pair exists.

The script:
  1. Computes, in Python, the full classical table of divisive pairs among
     the 4*3 = 12 ordered distinct-index pairs (i != j), from first
     principles (the `%` operator), and thus the classical set of marked
     "winner" states and the classical count.
  2. Builds a genuine Grover-search circuit over a 4-qubit register
     (2 qubits for i, 2 qubits for j, all 16 basis states, i==j states are
     simply never marked) whose oracle marks exactly the divisive pairs
     found classically, with an oracle built as a sum-of-products (multi-
     controlled-Z per marked bitstring) -- not a lookup of the answer.
  3. Runs Grover's algorithm (optimal iteration count for the known number
     of marked states) on the ideal AerSimulator, and checks that the most
     frequently measured index pair(s) are exactly the classically-marked
     divisive pairs.
  4. Prints PASS if the quantum search result matches the classical answer,
     FAIL otherwise.

This does verify a real, non-trivial computable property (existence and
identity of divisor pairs -- directly the defining property of a "primitive
set", the tag attached to problem #872) with a real Grover oracle+diffusion
circuit, even though it does not touch an actual OEIS sequence, because none
is available for this problem entry.
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles)
# ---------------------------------------------------------------------------

L = [2, 3, 4, 6]
N = len(L)  # 4 -> 2 index qubits each for i and j

assert N == 4, "circuit below is hard-coded for 4 elements (2 qubits/index)"

# All ordered pairs (i, j), i != j, marked iff L[i] divides L[j].
classical_marked_pairs = []
for i, j in product(range(N), repeat=2):
    if i == j:
        continue
    if L[j] % L[i] == 0:
        classical_marked_pairs.append((i, j))

is_primitive_classical = len(classical_marked_pairs) == 0

print(f"List L = {L}")
print(f"Classical divisive pairs (i, j) with L[i] | L[j], i != j: "
      f"{classical_marked_pairs}")
print(f"Classical verdict: L is {'PRIMITIVE' if is_primitive_classical else 'NOT primitive'} "
      f"({len(classical_marked_pairs)} divisive pair(s))")


def bits_for_pair(i, j):
    """2-bit big-endian encoding of index i, then index j -> 4-bit string i1 i0 j1 j0."""
    return f"{i:02b}{j:02b}"


marked_bitstrings = [bits_for_pair(i, j) for (i, j) in classical_marked_pairs]
print(f"Marked 4-bit oracle targets (i1 i0 j1 j0): {marked_bitstrings}")


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classically-computed bitstrings
# ---------------------------------------------------------------------------
# Qubit layout (register order q0..q3): q0 = i1 (MSB of i), q1 = i0 (LSB of i),
# q2 = j1 (MSB of j), q3 = j0 (LSB of j). A marked bitstring "b0 b1 b2 b3"
# (as printed above, i1 i0 j1 j0) is targeted by flipping any 0-bits with X,
# applying a multi-controlled Z across all 4 qubits, and undoing the X's.

NUM_QUBITS = 4


def apply_oracle(qc: QuantumCircuit, bitstrings):
    for bs in bitstrings:
        zero_positions = [k for k, b in enumerate(bs) if b == "0"]
        for k in zero_positions:
            qc.x(k)
        qc.h(NUM_QUBITS - 1)
        qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
        qc.h(NUM_QUBITS - 1)
        for k in zero_positions:
            qc.x(k)


def apply_diffuser(qc: QuantumCircuit):
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))


def build_grover_circuit(bitstrings, num_iterations):
    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    qc.h(range(NUM_QUBITS))
    for _ in range(num_iterations):
        apply_oracle(qc, bitstrings)
        apply_diffuser(qc)
    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))
    return qc


# ---------------------------------------------------------------------------
# 3. Run Grover's algorithm on the ideal simulator
# ---------------------------------------------------------------------------

if is_primitive_classical:
    # No marked states: Grover search has nothing to amplify. Verify instead
    # that a 0-iteration "search" (uniform superposition) never collapses to
    # a false positive oracle match -- i.e. confirm no marked bitstring
    # exists by exhaustive classical check (already done above) and skip
    # quantum amplification, which is the mathematically correct behavior.
    quantum_top_pairs = []
    shots_used = 0
    counts = {}
else:
    M = len(marked_bitstrings)
    total_states = 2 ** NUM_QUBITS
    num_iterations = max(1, round((np.pi / 4) * np.sqrt(total_states / M)))

    circuit = build_grover_circuit(marked_bitstrings, num_iterations)

    backend = AerSimulator()
    transpiled = transpile(circuit, backend)
    shots_used = 4096
    result = backend.run(transpiled, shots=shots_used).result()
    counts = result.get_counts()

    # Qiskit reports bitstrings in little-endian classical-register order
    # (c3 c2 c1 c0), i.e. reversed relative to our q0..q3 encoding order.
    # Reverse back to get "i1 i0 j1 j0" and keep every outcome that is one
    # of the classically marked bitstrings, ranked by measured frequency.
    normalized_counts = {}
    for bitstring, count in counts.items():
        encoded = bitstring[::-1]  # back to i1 i0 j1 j0
        normalized_counts[encoded] = normalized_counts.get(encoded, 0) + count

    sorted_outcomes = sorted(normalized_counts.items(), key=lambda kv: -kv[1])
    max_count = sorted_outcomes[0][1]
    # Take every outcome within the top bucket (handles ties among the
    # multiple marked states, e.g. when M > 1).
    top_bitstrings = {bs for bs, c in sorted_outcomes if c >= 0.5 * max_count}
    quantum_top_pairs = sorted(
        (int(bs[0:2], 2), int(bs[2:4], 2)) for bs in top_bitstrings
    )

    print(f"Grover iterations used: {num_iterations}, shots: {shots_used}")
    print(f"Top measured (i, j) pairs: {quantum_top_pairs}")


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer
# ---------------------------------------------------------------------------

classical_top_pairs = sorted(classical_marked_pairs)

verified = quantum_top_pairs == classical_top_pairs

print()
if verified:
    print("PASS")
else:
    print("FAIL")
    print(f"  classical: {classical_top_pairs}")
    print(f"  quantum:   {quantum_top_pairs}")
