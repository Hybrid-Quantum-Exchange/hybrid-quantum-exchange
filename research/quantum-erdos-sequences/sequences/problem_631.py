"""
Erdos problem #631 (erdosproblems.com), quantum-testable instance.

Source metadata (from manman4/erdosproblems data/problems.yaml, entry
`number: "631"`): status "proved", tags ["graph theory", "chromatic number"],
oeis: ["N/A"] -- this problem has NO associated OEIS sequence. There is
therefore no OEIS term to look up or verify against; this is documented
honestly rather than fabricating an OEIS id.

Because problem #631 is a chromatic-number / graph-coloring statement with
no attached sequence, the closest genuine, finite, computable property that
still lives in the same mathematical territory (graph coloring / chromatic
number) and that a small quantum circuit can actually search is:

    PROPERTY TESTED: "Is a given small graph G properly 2-colorable?"
    (equivalently: does G admit a partition of its vertices into two classes
    such that every edge has endpoints in different classes -- i.e. is G
    bipartite / is its chromatic number <= 2?)

Concretely we fix G = the 4-cycle C4 on vertices {0,1,2,3} with edges
    (0,1), (1,2), (2,3), (3,0)
and search, with Grover's algorithm, the space of all 2^4 = 16 assignments
of one bit (color) per vertex for an assignment under which every edge is
properly colored (its two endpoints differ).

Classical ground truth (computed in this script by brute force over all 16
assignments, independent of any quantum step): C4 is bipartite, so there are
exactly 2 valid proper 2-colorings -- the two class assignments {0,2}/{1,3}
and its complement, i.e. bitstrings 0101 and 1010 (little/big-endian handled
explicitly below). Grover's algorithm with 2 marked states out of 16 should
amplify exactly those two computational basis states.

CIRCUIT:
  - 4 "vertex" qubits q0..q3 (one bit = one color per vertex).
  - 4 ancilla "edge" qubits, one per edge, each computed as
    q_u XOR q_v via two CNOTs (so the ancilla is 1 iff that edge's endpoints
    differ, i.e. the edge is properly colored).
  - A multi-controlled Z (phase flip) that fires only when all 4 edge
    ancillas are 1 (all edges properly colored) -- this is the Grover oracle,
    implemented reversibly (edge ancillas are uncomputed after the phase
    flip so they return to |0> and can be reused across iterations).
  - Standard Grover diffusion operator on the 4 vertex qubits.
  - ceil(pi/4 * sqrt(N/M)) = 2 Grover iterations for N=16, M=2 marked states.

The script runs this on Qiskit's ideal AerSimulator, measures the vertex
qubits, and checks that essentially all sampled shots land on the two
classically-valid colorings, printing PASS/FAIL accordingly.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force all proper 2-colorings of C4.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
N_VERTICES = 4


def is_proper_2_coloring(bits):
    """bits: tuple of 4 ints (0/1), bits[i] = color of vertex i."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


def classical_valid_colorings():
    valid = []
    for bits in itertools.product([0, 1], repeat=N_VERTICES):
        if is_proper_2_coloring(bits):
            valid.append(bits)
    return valid


CLASSICAL_VALID = classical_valid_colorings()
assert CLASSICAL_VALID == [(0, 1, 0, 1), (1, 0, 1, 0)], (
    "C4 must have exactly the two alternating 2-colorings; got "
    f"{CLASSICAL_VALID}"
)

# Bitstring form matching Qiskit's little-endian qubit-to-bit ordering when
# we later read `counts` keys (Qiskit prints c[n-1] ... c[0], i.e. the
# classical-register bit string has qubit 0 as the *rightmost* character).
CLASSICAL_VALID_BITSTRINGS = {
    "".join(str(b) for b in reversed(bits)) for bits in CLASSICAL_VALID
}


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for proper 2-colorings of C4.
# ---------------------------------------------------------------------------

def build_oracle(qc, vqs, aqs):
    """Phase-flip the marked states (all 4 edges properly colored).

    vqs: list of 4 vertex qubits (register objects/indices)
    aqs: list of 4 ancilla qubits, one per edge, used as scratch (XOR of
         endpoints) and restored to |0> at the end (uncomputed).
    """
    # Compute edge ancillas: ancilla_e = q_u XOR q_v
    for i, (u, v) in enumerate(EDGES):
        qc.cx(vqs[u], aqs[i])
        qc.cx(vqs[v], aqs[i])

    # Multi-controlled Z on the 4 ancillas: flip phase iff all ancillas = 1,
    # i.e. iff every edge is properly colored.
    qc.h(aqs[3])
    qc.mcx(aqs[0:3], aqs[3])
    qc.h(aqs[3])

    # Uncompute the ancillas so they return to |0>.
    for i, (u, v) in enumerate(EDGES):
        qc.cx(vqs[v], aqs[i])
        qc.cx(vqs[u], aqs[i])


def build_diffuser(qc, vqs):
    qc.h(vqs)
    qc.x(vqs)
    qc.h(vqs[-1])
    qc.mcx(vqs[0:-1], vqs[-1])
    qc.h(vqs[-1])
    qc.x(vqs)
    qc.h(vqs)


def build_grover_circuit(n_iterations):
    v = QuantumRegister(N_VERTICES, "v")
    a = QuantumRegister(len(EDGES), "a")
    c = ClassicalRegister(N_VERTICES, "c")
    qc = QuantumCircuit(v, a, c)

    # Uniform superposition over all 2^4 vertex-colorings.
    qc.h(v)

    for _ in range(n_iterations):
        build_oracle(qc, list(v), list(a))
        build_diffuser(qc, list(v))

    qc.measure(v, c)
    return qc


def main():
    N = 2 ** N_VERTICES          # search space size = 16
    M = len(CLASSICAL_VALID)     # number of marked (valid) states = 2

    n_iter = max(1, round(np.pi / 4 * np.sqrt(N / M)))
    qc = build_grover_circuit(n_iter)

    sim = AerSimulator()
    shots = 4096
    job = sim.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Fraction of shots landing on one of the two classically-valid colorings.
    hits = sum(cnt for bitstring, cnt in counts.items()
               if bitstring in CLASSICAL_VALID_BITSTRINGS)
    hit_fraction = hits / shots

    print("Erdos problem #631 -- quantum test")
    print(f"  tags: graph theory, chromatic number; oeis: N/A (documented, not fabricated)")
    print(f"  property tested: C4 proper 2-colorability via Grover search")
    print(f"  classical valid colorings (vertex order q0 q1 q2 q3): {CLASSICAL_VALID}")
    print(f"  Grover iterations used: {n_iter} (N={N}, M={M})")
    print(f"  measured counts: {counts}")
    print(f"  fraction of shots on a classically-valid coloring: {hit_fraction:.4f}")

    # With N=16, M=2 and the standard optimal iteration count, ideal Grover
    # amplifies the marked-state probability close to 1; we require a high
    # majority of shots to land on a valid coloring as the pass criterion.
    passed = hit_fraction > 0.90

    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
