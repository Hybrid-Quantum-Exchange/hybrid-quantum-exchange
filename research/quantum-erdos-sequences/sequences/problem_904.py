"""
Erdos problem #904 -- quantum-testable instance.

Source metadata (from /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: \"904\"", verified 2026-09-19):
    prize: no
    status: proved (Lean)
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION, stated honestly up front: problem 904 has no associated OEIS
sequence id ("N/A" in the source data). There is therefore no OEIS integer
sequence to test membership/term-computation against for this problem, as
the general recipe in this task calls for. Rather than fabricate an OEIS
id or copy a value with no real content, this script instead builds a
genuine, non-trivial quantum computation rooted in the one real piece of
metadata problem 904 does carry: its tag "graph theory". It is offered as
the best honest attempt for a problem that does not supply the intended
raw material (an OEIS id), not as a literal test of Erdos problem 904's
mathematical content.

Classical property under test
------------------------------
Graph: the path graph P4 on vertices {0,1,2,3} with edges
    (0,1), (1,2), (2,3)
(a minimal, concrete, finite graph-theory object, in the spirit of the
problem's "graph theory" tag).

Property: the set of INDEPENDENT SETS OF SIZE EXACTLY 2 in this graph --
i.e. all 2-element vertex subsets {u, v} such that (u, v) is NOT an edge
of the graph. This is computed here from first principles by brute-force
enumeration over all size-2 subsets of {0,1,2,3}, checked against the
explicit edge list -- no external data, no OEIS lookup.

For P4 the classical brute-force answer (computed below, not asserted) is
the 3 subsets: {0,2}, {0,3}, {1,3}.

Quantum circuit
----------------
A genuine Grover search over the N = 2^4 = 16 computational basis states
of 4 qubits, each qubit x_i meaning "vertex i is in the subset". The
oracle phase-flips exactly the basis states that were classically found
above to be independent sets of size 2, and diffusion (inversion about
the mean) amplifies them. This is run on the ideal AerSimulator (real
Grover circuit: Hadamards, a multi-target phase oracle built from the
classically-verified target list, the standard diffuser, repeated the
optimal number of Grover iterations for |targets|=3 out of N=16).

Pass condition: measuring the circuit many times, the set of bitstrings
whose measured probability mass dominates (i.e. the most frequent
outcomes) must equal exactly the classically-computed target set.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


# ---------------------------------------------------------------------------
# 1. Classical computation, from first principles.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = [(0, 1), (1, 2), (2, 3)]  # path graph P4
EDGE_SET = set(frozenset(e) for e in EDGES)


def is_independent_set(subset):
    """True iff no edge of the graph has both endpoints in `subset`."""
    for u, v in itertools.combinations(subset, 2):
        if frozenset((u, v)) in EDGE_SET:
            return False
    return True


def classical_size2_independent_sets():
    targets = []
    for subset in itertools.combinations(VERTICES, 2):
        if is_independent_set(subset):
            targets.append(subset)
    return targets


CLASSICAL_TARGETS = classical_size2_independent_sets()
print("Classical brute-force size-2 independent sets of P4:", CLASSICAL_TARGETS)
assert CLASSICAL_TARGETS == [(0, 2), (0, 3), (1, 3)], (
    "Sanity check on the brute-force enumeration failed"
)

N_QUBITS = len(VERTICES)  # 4
N = 2 ** N_QUBITS  # 16


def subset_to_bitstring(subset):
    """Map a vertex subset to the little-endian 4-bit string with those
    qubits set to 1 (qubit i <-> vertex i)."""
    bits = ["0"] * N_QUBITS
    for v in subset:
        bits[v] = "1"
    # Qiskit reports bitstrings with qubit 0 as the rightmost character.
    return "".join(reversed(bits))


TARGET_BITSTRINGS = sorted(subset_to_bitstring(s) for s in CLASSICAL_TARGETS)
print("Target bitstrings (qubit0=vertex0, ..., rightmost bit = qubit0):", TARGET_BITSTRINGS)


# ---------------------------------------------------------------------------
# 2. Grover search circuit.
# ---------------------------------------------------------------------------

def build_oracle(n_qubits, target_bitstrings):
    """Phase-flip exactly the given target computational basis states."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bitstring in target_bitstrings:
        # bitstring[0] is qubit n-1 ... bitstring[-1] is qubit 0 (Qiskit order)
        zero_qubits = [q for q in range(n_qubits) if bitstring[n_qubits - 1 - q] == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        if n_qubits == 1:
            qc.z(0)
        else:
            mcz_via_mcx(qc, list(range(n_qubits)))
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def mcz_via_mcx(qc, qubits):
    """Multi-controlled Z on `qubits` (phase-flip |11...1>), built from an
    MCX with the last qubit as target, sandwiched in H gates."""
    target = qubits[-1]
    controls = qubits[:-1]
    qc.h(target)
    qc.append(MCXGate(len(controls)), controls + [target])
    qc.h(target)


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    mcz_via_mcx(qc, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits, target_bitstrings, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, target_bitstrings)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


M = len(TARGET_BITSTRINGS)
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"N={N}, M={M} marked states, optimal Grover iterations = {optimal_iterations}")

circuit = build_grover_circuit(N_QUBITS, TARGET_BITSTRINGS, optimal_iterations)

simulator = AerSimulator()
compiled = transpile(circuit, simulator)
SHOTS = 20000
result = simulator.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
print("Top measured outcomes:", sorted_counts[:6])

# The `M` most frequent measured outcomes should be exactly the classical
# target set, each with a large probability boost over the 1/N baseline.
top_m_outcomes = set(bs for bs, _ in sorted_counts[:M])
quantum_answer = sorted(top_m_outcomes)

baseline_prob = 1.0 / N
marked_total = sum(counts.get(bs, 0) for bs in TARGET_BITSTRINGS) / SHOTS
print(f"Total measured probability on the {M} marked states: {marked_total:.4f} "
      f"(uniform baseline would be {M * baseline_prob:.4f})")

matches_targets = quantum_answer == TARGET_BITSTRINGS
amplified = marked_total > 3 * (M * baseline_prob)  # clearly boosted vs. uniform

if matches_targets and amplified:
    print("PASS: Grover search on the ideal AerSimulator recovered exactly the "
          "classically brute-forced size-2 independent sets of P4, with "
          "probability mass strongly concentrated on the correct answers.")
else:
    print("FAIL: quantum result did not match the classical brute-force answer.")
    print("  classical:", TARGET_BITSTRINGS)
    print("  quantum top-M:", quantum_answer)
