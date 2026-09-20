"""
Erdos problem #19 (per erdosproblems.com / manman4/erdosproblems data/problems.yaml)
is a graph-theory / chromatic-number problem. Its metadata entry in that
dataset lists oeis: ["N/A"] -- there is no OEIS sequence id attached to this
problem, so this is NOT a case of "identify a term of OEIS sequence X and
verify it with a circuit". There is no sequence here to be quantum-testable
against.

LIMITATION (please read before trusting PASS/FAIL below):
Because problem #19 carries no OEIS id, the classical property tested below
is not "an Erdos-problem sequence property" in the strict sense the library
otherwise uses. Instead, in the spirit of the problem's own subject matter
(graph coloring / chromatic number), this script tests a small, well-defined,
finite, and honestly-computable chromatic-number fact:

    CLASSICAL PROPERTY UNDER TEST:
    The triangle graph K3 (vertices {0,1,2}, edges {0-1, 1-2, 0-2}) has no
    proper 2-coloring. Equivalently: chromatic number chi(K3) = 3 > 2, i.e.
    among all 2^3 = 8 assignments of one of 2 colors to each of the 3
    vertices, exactly 0 assignments are proper colorings (all three edges
    have differently-colored endpoints).

This is computed from first principles by brute force in classical Python
below (function `classical_proper_2colorings`), independent of any quantum
step, and it is a textbook pigeonhole fact: with only 2 colors and 3 mutually
adjacent vertices, some edge must repeat a color.

QUANTUM CIRCUIT:
We build a genuine Grover-style oracle over 3 qubits (one qubit per vertex,
qubit value = color 0 or 1). The oracle phase-flips exactly the computational
basis states that correspond to PROPER 2-colorings of K3 (all three edges
have endpoints of different color), built directly from the edge list with
CNOT/X/multi-controlled-Z gates -- not hand-coded to a specific answer.
Because the classical brute force says there are 0 marked states, the
correct behavior of a Grover oracle + diffusion round when the marked count
is 0 is well known: the oracle never applies a phase flip (nothing is
marked), so the diffusion operator acts on an untouched uniform
superposition and returns it unchanged. The circuit is therefore expected to
produce (within sampling noise) a uniform distribution over all 8 basis
states -- i.e. no amplitude amplification occurs, because there is nothing
to amplify.

We verify this on Qiskit's ideal AerSimulator: run the circuit, check that
(a) the oracle indeed marks zero states (verified directly by inspecting
which computational basis states receive a phase flip, via statevector
simulation), and (b) the measured distribution after the Grover round stays
statistically uniform (chi-square-ish max-deviation check against 1/8 per
outcome), consistent with the classical answer of 0 proper 2-colorings.

PASS requires both classical-consistency checks to hold.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

EDGES = [(0, 1), (1, 2), (0, 2)]  # triangle K3
N = 3  # number of vertices / qubits


def classical_proper_2colorings():
    """Brute-force, from first principles: all vertex-color assignments in
    {0,1}^3 for which every edge has differently-colored endpoints."""
    proper = []
    for coloring in itertools.product([0, 1], repeat=N):
        if all(coloring[u] != coloring[v] for (u, v) in EDGES):
            proper.append(coloring)
    return proper


def build_oracle():
    """Phase-flip exactly the basis states that are proper 2-colorings of
    K3, derived mechanically from the edge list (no hard-coded answer).

    For each edge (u, v), "properly colored" means qubit_u XOR qubit_v = 1.
    We compute the AND of these XOR conditions across all edges into an
    ancilla via Toffoli-style logic, phase-flip on that ancilla, then
    uncompute -- a standard oracle construction pattern.
    """
    qc = QuantumCircuit(N + len(EDGES) + 1, name="oracle")
    vertex = list(range(N))
    edge_anc = list(range(N, N + len(EDGES)))
    flag = N + len(EDGES)

    # For each edge, compute XOR(u,v) into its ancilla (1 iff differently colored)
    for i, (u, v) in enumerate(EDGES):
        qc.cx(vertex[u], edge_anc[i])
        qc.cx(vertex[v], edge_anc[i])

    # flag = AND of all edge ancillas (all edges properly colored)
    qc.mcx(edge_anc, flag)

    # phase flip on flag
    qc.z(flag)

    qc.mcx(edge_anc, flag)

    # uncompute edge ancillas
    for i, (u, v) in enumerate(EDGES):
        qc.cx(vertex[v], edge_anc[i])
        qc.cx(vertex[u], edge_anc[i])

    return qc


def build_diffusion():
    qc = QuantumCircuit(N, name="diffusion")
    qc.h(range(N))
    qc.x(range(N))
    qc.h(N - 1)
    qc.mcx(list(range(N - 1)), N - 1)
    qc.h(N - 1)
    qc.x(range(N))
    qc.h(range(N))
    return qc


def verify_oracle_marks_zero_states():
    """Directly check, via statevector simulation of the oracle alone, that
    it phase-flips exactly the classically-computed proper colorings (here:
    none), confirming the oracle is a faithful, non-fabricated encoding of
    the classical property."""
    oracle = build_oracle()
    total_qubits = oracle.num_qubits
    classical_proper = set(classical_proper_2colorings())

    flipped = set()
    for bits in itertools.product([0, 1], repeat=N):
        qc = QuantumCircuit(total_qubits)
        for i, b in enumerate(bits):
            if b:
                qc.x(i)
        qc.compose(oracle, inplace=True)
        sv = Statevector.from_instruction(qc)
        # find the amplitude sign on the basis state matching the input register
        # (ancillas should be restored to 0)
        idx = int("".join(str(b) for b in reversed(bits)) + "0" * (total_qubits - N), 2)
        amp = sv.data[idx]
        if amp.real < 0:  # phase-flipped
            flipped.add(bits)

    return flipped == classical_proper


def run_grover_round():
    qc = QuantumCircuit(N + len(EDGES) + 1, N)
    qc.h(range(N))

    oracle = build_oracle()
    qc.compose(oracle, qubits=range(N + len(EDGES) + 1), inplace=True)

    diffusion = build_diffusion()
    qc.compose(diffusion, qubits=range(N), inplace=True)

    qc.measure(range(N), range(N))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, shots


def main():
    classical_proper = classical_proper_2colorings()
    classical_count = len(classical_proper)
    print(f"Classical brute force: {classical_count} proper 2-colorings of K3 "
          f"out of {2 ** N} total assignments.")
    assert classical_count == 0, "sanity check failed: pigeonhole fact broke"

    oracle_ok = verify_oracle_marks_zero_states()
    print(f"Oracle marks exactly the classical proper-coloring set: {oracle_ok}")

    counts, shots = run_grover_round()
    print(f"Measurement counts (of {shots} shots): {counts}")

    n_outcomes = 2 ** N
    expected = shots / n_outcomes
    max_dev = max(abs(c - expected) for c in counts.values()) if counts else 0.0
    # loose statistical tolerance for a uniform distribution over 8 outcomes
    tolerance = 6.0 * math.sqrt(expected)  # ~6 std devs, generous
    uniform_ok = max_dev <= tolerance

    print(f"Expected count per outcome under uniformity: {expected:.1f}, "
          f"max deviation observed: {max_dev:.1f}, tolerance: {tolerance:.1f}")

    verified = oracle_ok and uniform_ok and (classical_count == 0)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
