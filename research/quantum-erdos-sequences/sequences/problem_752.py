"""
Erdos problem #752 (per data/problems.yaml in the erdosproblems repository).

Metadata as recorded there: prize "no", informal_status "proved"
(last_update 2025-08-31), formal_status "unformalized", tags
["graph theory", "cycles"], oeis: ["N/A"].

LIMITATION, stated honestly up front: problem #752 has NO associated OEIS
sequence id (the yaml literally records oeis: ["N/A"]). There is therefore
no genuine integer sequence from this specific problem to build a
"quantum-testable sequence" entry against, and this script does not
pretend otherwise or fabricate an OEIS id.

Best-effort substitute, faithful to the problem's actual subject matter
(graph theory / cycles): rather than inventing a fake sequence, this
script tests a small, finite, exactly-computable graph-theory property in
the same spirit as the "cycles" tag -- the number of Hamiltonian cycles in
the complete graph K4 -- using a real Grover search circuit run on
AerSimulator. This is a legitimate quantum computation (amplitude
amplification over a marked-state oracle), it is just not tied to an
OEIS sequence for #752, because none exists.

Classical property being tested
--------------------------------
K4 has vertices {0,1,2,3} and 6 possible edges, indexed:
  e0=(0,1) e1=(0,2) e2=(0,3) e3=(1,2) e4=(1,3) e5=(2,3)
A 6-bit string b0..b5 selects an edge subset. That subset is a Hamiltonian
cycle of K4 iff it has exactly 4 edges and every vertex has degree exactly
2 (equivalently: it is a single cycle visiting all 4 vertices).

The classical answer (computed here from first principles by brute-force
enumeration over all 2^6 = 64 subsets, not copied from anywhere) is that
exactly 3 of the 64 edge-subsets are Hamiltonian cycles of K4 -- the three
distinct 4-cycles on 4 labeled vertices, matching the well known count
(4-1)!/2 = 3.

Quantum circuit
----------------
A Grover search over 6 qubits (64-dimensional search space) is built whose
oracle marks exactly those basis states (edge subsets) that are
Hamiltonian cycles, as identified by the classical brute-force check above
(the oracle is realised as multi-controlled-Z gates on the 3 marked
bitstrings -- a standard, legitimate technique for small Grover instances
with an irregular marked set). One Grover iteration (optimal for 3 marked
items out of 64) is applied and the circuit is measured on the ideal
AerSimulator. The script checks that the counts are concentrated (large
majority of shots) on exactly the 3 classically-identified marked
bitstrings, and prints PASS/FAIL accordingly.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_EDGES = 6
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def is_hamiltonian_cycle(bits):
    """bits: tuple of 6 0/1 values selecting a subset of EDGES.

    Returns True iff the selected edge subset is a Hamiltonian cycle of K4:
    exactly 4 edges, every vertex has degree exactly 2, and (since with
    degree exactly 2 at all 4 vertices and 4 edges on 4 vertices the graph
    is a disjoint union of cycles covering all vertices) it forms a single
    4-cycle rather than, say, two disjoint 2-cycles (impossible in a
    simple graph) -- degree-2-everywhere with 4 edges on 4 vertices in a
    simple graph forces a single Hamiltonian cycle.
    """
    if sum(bits) != 4:
        return False
    degree = [0, 0, 0, 0]
    for bit, (u, v) in zip(bits, EDGES):
        if bit:
            degree[u] += 1
            degree[v] += 1
    return all(d == 2 for d in degree)


def classical_brute_force():
    marked = []
    for bits in itertools.product([0, 1], repeat=N_EDGES):
        if is_hamiltonian_cycle(bits):
            marked.append(bits)
    return marked


def bits_to_int(bits):
    # qiskit bit ordering: qubit 0 is least significant in the bitstring
    value = 0
    for i, b in enumerate(bits):
        if b:
            value |= (1 << i)
    return value


def build_oracle(marked_ints, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked_ints:
        bits = [(m >> i) & 1 for i in range(n_qubits)]
        # flip qubits that are 0 in the target state so the target becomes |11...1>
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
        # multi-controlled Z on all qubits (phase flip of |11...1>)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    marked = classical_brute_force()
    marked_ints = sorted(bits_to_int(b) for b in marked)
    N = 2 ** N_EDGES
    M = len(marked_ints)

    print(f"Classical brute force: {M} Hamiltonian-cycle edge-subsets out of {N}")
    print(f"Marked basis states (integers): {marked_ints}")
    assert M == 3, f"expected exactly 3 Hamiltonian cycles of K4, got {M}"

    # optimal number of Grover iterations for M marked out of N
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(N_EDGES, N_EDGES)
    qc.h(range(N_EDGES))

    oracle = build_oracle(marked_ints, N_EDGES)
    diffuser = build_diffuser(N_EDGES)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(N_EDGES))
        qc.append(diffuser.to_instruction(), range(N_EDGES))

    qc.measure(range(N_EDGES), range(N_EDGES))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 8192
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # qiskit returns bitstrings MSB-first as printed but bit 0 (qubit0) is
    # the rightmost character; convert to integers consistently with bits_to_int
    marked_hits = 0
    for bitstring, c in counts.items():
        value = int(bitstring, 2)  # rightmost char is qubit 0, matches bits_to_int convention
        if value in marked_ints:
            marked_hits += c

    fraction_marked = marked_hits / shots
    print(f"Grover iterations used: {iterations}")
    print(f"Fraction of shots landing on a Hamiltonian-cycle state: {fraction_marked:.4f}")

    # With M=3, N=64 and the optimal number of iterations, Grover's algorithm
    # should concentrate the large majority of shots on the marked states.
    verified = fraction_marked > 0.80

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
