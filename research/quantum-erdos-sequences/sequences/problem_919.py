"""
Erdos problem #919 -- quantum-testable lane.

Source metadata (erdosproblems.com data, data/problems.yaml, entry
"number: 919"):
    prize: no
    status: open
    tags: ["graph theory", "chromatic number"]
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #919 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no "sequence" from
this problem to build a quantum-testable membership/search property from, as
the other lanes in this library do. Rather than fabricate an OEIS id or copy
a value that doesn't exist, this script instead builds a REAL, correctly
verified quantum circuit for the one piece of genuine, finite, computable
mathematical content the problem's own tags name: a *chromatic-number*
decision question from graph theory -- "is this small graph 2-colorable
(equivalently, bipartite / has chromatic number <= 2)?" -- answered by a
Grover search over all 2-colorings of a fixed small graph.

Concretely:
    - Graph: the 4-cycle C4 with vertices {0,1,2,3} and edges
      (0,1), (1,2), (2,3), (3,0).
    - Property under test: a proper 2-coloring of C4, i.e. an assignment of
      one bit (color) to each of the 4 vertices such that every edge joins
      two different colors.
    - Classical ground truth (computed here in the script, by brute-force
      enumeration of all 2^4 = 16 colorings -- not copied from anywhere):
      exactly 2 of the 16 colorings are proper 2-colorings of C4:
      0101 and 1010 (bit i = color of vertex i), confirming C4 is bipartite
      and has chromatic number exactly 2.
    - Quantum circuit: a genuine Grover search circuit over the 4 "color"
      qubits. A reversible oracle computes, on 4 ancilla qubits, the XOR of
      the two endpoint colors for each of the 4 edges (so an ancilla reads 1
      exactly when that edge is properly colored), phase-flips the state
      when all 4 ancillas are 1 (all edges proper), then uncomputes the
      ancillas. This oracle is wrapped in the standard Grover diffusion
      operator, run for the optimal number of iterations for N=16, M=2
      marked states (round(pi/4 * sqrt(N/M)) = 2 iterations), and the result
      is measured on the ideal AerSimulator.
    - PASS/FAIL: the script passes if the two most probable measured
      4-bit strings on the color register are exactly the 2 classically
      verified proper 2-colorings (0101 and 1010), each with amplified
      probability well above the 1/16 baseline of uniform random guessing.

This is a real amplitude-amplification computation (not a lookup table) over
a real, correctly brute-force-verified finite instance of the chromatic
number question the problem's tags describe -- but it should be read as an
honest substitute for a missing OEIS-sequence property, not as a circuit
that tests problem #919 itself (which remains open and untouched by this).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # the 4-cycle C4


def is_proper_2_coloring(bits: tuple[int, ...]) -> bool:
    """bits[i] is the color (0/1) of vertex i; proper iff every edge differs."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


def classical_valid_colorings() -> list[str]:
    """Brute-force all 2^4 colorings of C4 and return the proper ones.

    Returned as bitstrings in Qiskit's little-endian convention (qubit 0 is
    the rightmost character), so they can be compared directly against
    measurement outcome keys.
    """
    valid = []
    for bits in itertools.product((0, 1), repeat=N_VERTICES):
        if is_proper_2_coloring(bits):
            # bits[0] -> qubit 0, ... ; Qiskit prints qubit (n-1)...qubit 0.
            bitstring = "".join(str(bits[i]) for i in reversed(range(N_VERTICES)))
            valid.append(bitstring)
    return sorted(valid)


VALID_COLORINGS = classical_valid_colorings()
assert VALID_COLORINGS == ["0101", "1010"], (
    "brute-force chromatic check disagrees with the known result that C4 "
    f"has exactly two proper 2-colorings; got {VALID_COLORINGS}"
)

N_STATES = 2 ** N_VERTICES  # 16
M_MARKED = len(VALID_COLORINGS)  # 2

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for proper 2-colorings of C4.
# ---------------------------------------------------------------------------

# Register layout:
#   qubits 0..3   : color qubits, one per vertex (the search register)
#   qubits 4..7   : ancillas, ancilla k = XOR of the colors on EDGES[k]
#   qubit 8       : phase-kickback target for the multi-controlled oracle
COLOR = list(range(N_VERTICES))
ANC = list(range(N_VERTICES, N_VERTICES + len(EDGES)))
TARGET = N_VERTICES + len(EDGES)
N_QUBITS = TARGET + 1


def build_oracle() -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    # Compute edge-parity ancillas: ancilla k = color[u] XOR color[v].
    for k, (u, v) in enumerate(EDGES):
        qc.cx(COLOR[u], ANC[k])
        qc.cx(COLOR[v], ANC[k])
    # Phase-flip |...> when all 4 ancillas are 1, via phase kickback on
    # TARGET prepared in |-> : an X on TARGET controlled on all ancillas
    # being 1 applies a -1 phase to exactly those computational states.
    qc.append(MCXGate(len(ANC)), ANC + [TARGET])
    # Uncompute the ancillas so they return to |0> for the next iteration.
    for k, (u, v) in enumerate(EDGES):
        qc.cx(COLOR[v], ANC[k])
        qc.cx(COLOR[u], ANC[k])
    return qc


def build_diffuser_correct() -> QuantumCircuit:
    """Standard Grover diffuser: reflection about the uniform superposition."""
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(COLOR)
    qc.x(COLOR)
    qc.h(COLOR[-1])
    qc.append(MCXGate(N_VERTICES - 1), COLOR[:-1] + [COLOR[-1]])
    qc.h(COLOR[-1])
    qc.x(COLOR)
    qc.h(COLOR)
    return qc


def build_grover_circuit(iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, N_VERTICES, name="grover_c4_2coloring")
    # Search register in uniform superposition.
    qc.h(COLOR)
    # Phase-kickback target qubit prepared in |->.
    qc.x(TARGET)
    qc.h(TARGET)

    oracle = build_oracle()
    diffuser = build_diffuser_correct()
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(N_QUBITS))
        qc.append(diffuser.to_instruction(), range(N_QUBITS))

    # Undo the |-> prep on the target so it's clean (not strictly required
    # for correctness of the color-register measurement, but tidy).
    qc.h(TARGET)
    qc.x(TARGET)

    qc.measure(COLOR, list(range(N_VERTICES)))
    return qc


def optimal_iterations(n_states: int, n_marked: int) -> int:
    theta = math.asin(math.sqrt(n_marked / n_states))
    return max(1, round((math.pi / (4 * theta)) - 0.5))


ITERATIONS = optimal_iterations(N_STATES, M_MARKED)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------


def main() -> bool:
    circuit = build_grover_circuit(ITERATIONS)
    backend = AerSimulator()
    transpiled = transpile(circuit, basis_gates=["u", "cx"])
    shots = 4096
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_strings = {bitstring for bitstring, _ in ranked[:M_MARKED]}
    top_prob = sum(counts.get(s, 0) for s in top_strings) / total
    baseline_prob = M_MARKED / N_STATES  # what uniform random guessing gives

    print("Erdos problem #919 -- no OEIS id available (oeis: ['N/A']).")
    print("Substitute property tested: proper 2-colorings of the 4-cycle C4")
    print(f"Classical brute-force valid colorings: {VALID_COLORINGS}")
    print(f"Grover iterations used: {ITERATIONS} (N={N_STATES}, M={M_MARKED})")
    print(f"Measurement counts (top {2 * M_MARKED}):")
    for bitstring, count in ranked[: 2 * M_MARKED]:
        print(f"  {bitstring}: {count} ({count / total:.3f})")
    print(f"Top-{M_MARKED} measured strings: {sorted(top_strings)}")
    print(f"Top-{M_MARKED} probability mass: {top_prob:.3f} "
          f"(uniform baseline would be {baseline_prob:.3f})")

    matches_classical = top_strings == set(VALID_COLORINGS)
    amplified = top_prob > 3 * baseline_prob  # Grover should amplify strongly

    passed = matches_classical and amplified
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
