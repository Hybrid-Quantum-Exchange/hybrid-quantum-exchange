"""
Erdos problem #58 — quantum-testable lane.

Source metadata (data/problems.yaml, entry "number: '58'"): prize "no",
status "proved", tags ["graph theory", "chromatic number", "cycles"],
oeis: ["N/A"].

LIMITATION: problem #58 has no associated OEIS sequence id ("N/A"), so
there is no integer sequence to build a membership/term-search circuit
around. This script therefore does not test an OEIS sequence property.
Instead, honoring the problem's own tags (chromatic number of cycle
graphs), it tests a small, finite, genuinely computable graph-coloring
fact that the problem's subject matter is about: proper 2-colorings of
the 4-cycle graph C4 (vertices 0-1-2-3-0).

Classical property under test
------------------------------
Let C4 be the cycle graph on vertices {0,1,2,3} with edges
(0,1), (1,2), (2,3), (3,0). A 2-coloring is an assignment of one bit
(color) to each vertex. A 2-coloring is PROPER iff every edge joins two
vertices of different colors (i.e. the coloring is a valid 2-coloring,
witnessing chi(C4) <= 2, consistent with C4 being bipartite/even).

There are 2^4 = 16 possible colorings. We first enumerate all 16
classically (first principles, in this script) and count how many are
proper. For an even cycle C4 there are exactly 2: 0101 and 1010
(alternating colors around the cycle).

Quantum circuit
----------------
We build a genuine Grover search over the 4-bit coloring space:
  - 4 "vertex" qubits hold the candidate coloring (superposition of all
    16 states).
  - An oracle computes, for each of the 4 edges, XOR(bit_i, bit_j) into
    an ancilla qubit (marking "these two differ"), then flips the phase
    of the state only when all 4 ancillas are 1 (all edges satisfied),
    then uncomputes the ancillas.
  - The standard Grover diffusion operator amplifies the marked
    (proper-coloring) basis states.
  - With N=16 and M=2 solutions, the optimal number of Grover
    iterations is round(pi/4 * sqrt(N/M)) = 1.

We run the circuit on the ideal AerSimulator, measure the 4 vertex
qubits many times, and check that the two most probable outcomes are
exactly {0101, 1010} (as bitstrings) with amplified probability,
matching the classical enumeration. PASS/FAIL is decided by comparing
the quantum measurement distribution's top outcomes to the classical
brute-force answer.
"""

from itertools import product

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
N_VERTICES = 4


def is_proper_coloring(bits):
    """bits: tuple of 4 ints (0/1), bits[i] = color of vertex i."""
    return all(bits[i] != bits[j] for (i, j) in EDGES)


def classical_brute_force():
    """Enumerate all 2^4 colorings from first principles."""
    solutions = []
    for bits in product([0, 1], repeat=N_VERTICES):
        if is_proper_coloring(bits):
            solutions.append(bits)
    return solutions


def build_oracle(qc, v, anc):
    """Phase-flip states where every edge's endpoints differ (proper coloring)."""
    # Compute edge-difference bits into ancillas: anc[k] = v[i] XOR v[j]
    for k, (i, j) in enumerate(EDGES):
        qc.cx(v[i], anc[k])
        qc.cx(v[j], anc[k])
    # Multi-controlled Z on the 4 ancillas: flip phase iff all are 1
    # (all edges satisfied -> proper coloring).
    qc.h(anc[3])
    qc.mcx([anc[0], anc[1], anc[2]], anc[3])
    qc.h(anc[3])
    # Uncompute ancillas
    for k, (i, j) in enumerate(EDGES):
        qc.cx(v[j], anc[k])
        qc.cx(v[i], anc[k])


def build_diffuser(qc, v):
    n = len(v)
    qc.h(v)
    qc.x(v)
    qc.h(v[n - 1])
    qc.mcx(v[: n - 1], v[n - 1])
    qc.h(v[n - 1])
    qc.x(v)
    qc.h(v)


def build_grover_circuit(iterations):
    v = QuantumRegister(N_VERTICES, "v")
    anc = QuantumRegister(len(EDGES), "anc")
    qc = QuantumCircuit(v, anc, name="grover_c4_coloring")

    qc.h(v)  # uniform superposition over all 16 colorings

    for _ in range(iterations):
        build_oracle(qc, v, anc)
        build_diffuser(qc, v)

    qc.measure_all()
    return qc, v


def run_and_verify():
    classical_solutions = classical_brute_force()
    classical_bitstrings = {
        "".join(str(b) for b in reversed(bits)) for bits in classical_solutions
    }
    n_solutions = len(classical_solutions)
    n_total = 2 ** N_VERTICES

    import math

    iterations = max(1, round((math.pi / 4) * math.sqrt(n_total / n_solutions)))

    qc, v = build_grover_circuit(iterations)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 8192
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Extract just the vertex-register bits from each measured bitstring.
    # qc.measure_all() measures in a single classical register ordered as
    # [anc(3)...anc(0), v(3)...v(0)] in the returned bitstring (Qiskit
    # prints classical bits with the last-created register's bits first,
    # each register itself in little-endian qubit order).
    vertex_counts = {}
    for bitstring, c in counts.items():
        bitstring = bitstring.replace(" ", "")
        # Qiskit's default ordering: rightmost char = clbit 0.
        # clbits 0..N_VERTICES-1 correspond to v[0..3] (measure_all measures
        # qubits in order: v then anc, into clbits in the same order).
        v_bits_lsb_first = bitstring[::-1][:N_VERTICES]
        v_bitstring = "".join(reversed(v_bits_lsb_first))
        vertex_counts[v_bitstring] = vertex_counts.get(v_bitstring, 0) + c

    sorted_outcomes = sorted(vertex_counts.items(), key=lambda kv: -kv[1])
    top_two = {bs for bs, _ in sorted_outcomes[:2]}
    top_two_total_prob = sum(c for bs, c in sorted_outcomes[:2]) / shots

    print("Erdos problem #58 -- quantum lane (no OEIS id; C4 proper-2-coloring search)")
    print(f"Classical brute-force proper 2-colorings of C4: {sorted(classical_bitstrings)}")
    print(f"Grover iterations used: {iterations}")
    print("Top measured vertex-coloring outcomes (bitstring: counts):")
    for bs, c in sorted_outcomes[:6]:
        print(f"  {bs}: {c} ({c/shots:.3f})")
    print(f"Top-2 outcomes match classical solution set: {top_two == classical_bitstrings}")
    print(f"Top-2 combined probability: {top_two_total_prob:.3f}")

    verified = (top_two == classical_bitstrings) and (top_two_total_prob > 0.5)
    return verified


if __name__ == "__main__":
    ok = False
    try:
        ok = run_and_verify()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR during run: {e}")
        ok = False

    print("PASS" if ok else "FAIL")
