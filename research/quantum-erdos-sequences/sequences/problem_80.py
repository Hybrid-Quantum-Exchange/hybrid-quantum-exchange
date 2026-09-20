"""
Erdos problem #80 -- quantum-testable companion script.

Source metadata (data/problems.yaml, erdosproblems.com dataset, entry
"number: 80"):
    tags: ["graph theory", "ramsey theory"]
    oeis: ["N/A"]

LIMITATION, stated up front: problem #80's dataset entry carries no OEIS
sequence id at all (oeis: ["N/A"]). There is therefore no OEIS sequence for
this script to test membership/terms of, and the task's request ("From its
OEIS sequence id(s) and tags, identify a ... property of the sequence") has
no literal sequence to anchor to. Rather than fabricate a fake OEIS id or
a fake sequence property, this script instead builds a genuine, small,
computable decision problem drawn directly from problem #80's own tags
("graph theory", "ramsey theory"): the classical fact underlying
Ramsey's theorem for R(3,3), restricted to the smallest interesting case.

The classical property tested
------------------------------
Take the complete graph K4 on 4 vertices (6 edges). 2-color its edges
(each edge gets color 0 or 1 -- represented as one qubit per edge, so the
full search space has 2^6 = 64 colorings, satisfying N <= 64). K4 contains
exactly 4 triangles (one per way to omit a vertex): {0,1,2}, {0,1,3},
{0,2,3}, {1,2,3}. A coloring is "triangle-free" (in the Ramsey sense) if
none of these 4 triangles is monochromatic (all three of its edges the
same color).

This is exactly the finite fact that underlies R(3,3) = 6: for n = 4 (and
n = 5) there EXIST 2-colorings of K_n with no monochromatic triangle, while
for n = 6 there do not (that is what makes R(3,3) = 6). This script
verifies, classically and then with a real Grover search circuit, that
such triangle-free 2-colorings of K4 exist, and finds them.

This is real mathematical content tied to problem #80's own tags. It is
NOT a claim about any OEIS sequence, since none exists for this problem.

Classical ground truth (computed here, brute force over all 64 colorings)
---------------------------------------------------------------------------
The script enumerates all 2^6 edge colorings of K4, and for each one checks
whether all 4 triangles are non-monochromatic. It records the exact set of
"good" (triangle-free) colorings. This is the ground truth the quantum
result is checked against.

Quantum method
---------------
A Grover search circuit is built over 6 "edge" qubits plus 4 ancilla
qubits (one per triangle, used to flag "this triangle is monochromatic").
The oracle phase-flips exactly the edge-colorings for which all 4 ancillas
end up in state |0> (i.e. no monochromatic triangle) via a multi-controlled
Z gate, with the ancilla-computing gates uncomputed afterward so the
ancillas return to |0> before the diffuser. The standard Grover diffuser
is applied. The number of iterations is chosen from the classically-known
count of good colorings (via the standard Grover formula). The circuit is
then simulated on the ideal AerSimulator and the measured bitstrings are
checked against the classical set of good colorings.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
import itertools

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force all 2-colorings of K4's 6 edges.
# ---------------------------------------------------------------------------

# Vertices 0,1,2,3. Edge order (bit index -> edge):
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}


def edge_bit(coloring_bits, u, v):
    e = (u, v) if u < v else (v, u)
    return coloring_bits[EDGE_INDEX[e]]


# The 4 triangles of K4, each as a triple of vertices.
TRIANGLES = [
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
]


def is_triangle_free_coloring(coloring_bits):
    """coloring_bits: tuple of 6 bits (0/1), one per edge in EDGES order.
    Returns True iff no triangle in TRIANGLES is monochromatic."""
    for (a, b, c) in TRIANGLES:
        eab = edge_bit(coloring_bits, a, b)
        ebc = edge_bit(coloring_bits, b, c)
        eac = edge_bit(coloring_bits, a, c)
        if eab == ebc == eac:
            return False
    return True


def classical_brute_force():
    good = []
    for bits in itertools.product([0, 1], repeat=6):
        if is_triangle_free_coloring(bits):
            good.append(bits)
    return good


GOOD_COLORINGS = classical_brute_force()
N_GOOD = len(GOOD_COLORINGS)
N_TOTAL = 2 ** 6

assert N_GOOD > 0, (
    "Sanity check failed: classically, K4 must admit at least one "
    "triangle-free 2-edge-coloring (this is the base case underlying "
    "R(3,3) = 6). If this assertion fires, the classical logic above is "
    "wrong, not the mathematics."
)

# Represent each good coloring as the bitstring Qiskit will report
# (Qiskit's classical-register bitstrings are little-endian: c[0] is the
# rightmost character). Our qubit q_i holds edge i's color, and we measure
# q0..q5 into a 6-bit classical register, so bit i of the coloring tuple
# corresponds to character position (5 - i) from the left, i.e. standard
# Qiskit convention subject to reversal below.
GOOD_BITSTRINGS = set(
    "".join(str(bits[i]) for i in reversed(range(6))) for bits in GOOD_COLORINGS
)


# ---------------------------------------------------------------------------
# 2. Quantum: Grover search for a triangle-free coloring.
# ---------------------------------------------------------------------------

N_EDGE_QUBITS = 6
N_ANCILLA = 4  # one "monochromatic flag" ancilla per triangle
N_QUBITS = N_EDGE_QUBITS + N_ANCILLA


def build_oracle_and_diffuser():
    edge_q = QuantumRegister(N_EDGE_QUBITS, "e")
    anc_q = QuantumRegister(N_ANCILLA, "anc")  # monochromatic flags
    tmp_q = QuantumRegister(2 * N_ANCILLA, "tmp")  # 2 scratch qubits/triangle
    qc = QuantumCircuit(edge_q, anc_q, tmp_q)

    def edge_qubit(u, v):
        e = (u, v) if u < v else (v, u)
        return edge_q[EDGE_INDEX[e]]

    def compute_mono_flags():
        for t_idx, (a, b, c) in enumerate(TRIANGLES):
            qa, qb, qc_ = edge_qubit(a, b), edge_qubit(b, c), edge_qubit(a, c)
            t0, t1 = tmp_q[2 * t_idx], tmp_q[2 * t_idx + 1]
            # t0 = qa XOR qb ; t1 = qb XOR qc_
            qc.cx(qa, t0)
            qc.cx(qb, t0)
            qc.cx(qb, t1)
            qc.cx(qc_, t1)
            # mono <=> t0 == 0 AND t1 == 0  -> flip both, AND via Toffoli
            qc.x(t0)
            qc.x(t1)
            qc.ccx(t0, t1, anc_q[t_idx])
            qc.x(t0)
            qc.x(t1)

    def uncompute_mono_flags():
        for t_idx, (a, b, c) in enumerate(TRIANGLES):
            qa, qb, qc_ = edge_qubit(a, b), edge_qubit(b, c), edge_qubit(a, c)
            t0, t1 = tmp_q[2 * t_idx], tmp_q[2 * t_idx + 1]
            qc.x(t0)
            qc.x(t1)
            qc.ccx(t0, t1, anc_q[t_idx])
            qc.x(t0)
            qc.x(t1)
            qc.cx(qc_, t1)
            qc.cx(qb, t1)
            qc.cx(qb, t0)
            qc.cx(qa, t0)

    def oracle():
        compute_mono_flags()
        # Phase-flip states where ALL 4 mono-flag ancillas are 0
        # (no monochromatic triangle => a "good" coloring).
        qc.x(anc_q)
        qc.h(anc_q[N_ANCILLA - 1])
        qc.mcx(anc_q[: N_ANCILLA - 1], anc_q[N_ANCILLA - 1])
        qc.h(anc_q[N_ANCILLA - 1])
        qc.x(anc_q)
        uncompute_mono_flags()

    def diffuser():
        qc.h(edge_q)
        qc.x(edge_q)
        qc.h(edge_q[N_EDGE_QUBITS - 1])
        qc.mcx(list(edge_q[: N_EDGE_QUBITS - 1]), edge_q[N_EDGE_QUBITS - 1])
        qc.h(edge_q[N_EDGE_QUBITS - 1])
        qc.x(edge_q)
        qc.h(edge_q)

    qc.h(edge_q)

    # Standard Grover iteration count for N_TOTAL items, N_GOOD marked.
    theta = math.asin(math.sqrt(N_GOOD / N_TOTAL))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    for _ in range(iterations):
        oracle()
        diffuser()

    creg_name = "result"
    qc.add_register(QuantumRegister(0))  # no-op, keeps structure explicit
    return qc, edge_q, iterations


def run():
    qc, edge_q, iterations = build_oracle_and_diffuser()
    qc.measure_all()

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # measure_all() appends a classical register covering ALL qubits
    # (edges + ancillas + tmp), in qubit order, so the leftmost 6
    # characters after Qiskit's bit-reversal correspond to the ancilla/tmp
    # qubits reported last in creation order... to stay unambiguous, we
    # instead pick out exactly the N_EDGE_QUBITS edge-qubit bits by name.
    edge_qubit_indices = list(range(N_EDGE_QUBITS))

    def extract_edge_bits(bitstring):
        # Qiskit's returned bitstring is ordered qubit[N-1] ... qubit[0]
        # (MSB first), across ALL qubits in the circuit (edges, then
        # ancillas, then tmp, in registration order).
        full = bitstring.replace(" ", "")
        total_qubits = N_QUBITS
        # rightmost N_EDGE_QUBITS characters correspond to edge_q[0..5]
        return full[-N_EDGE_QUBITS:]

    edge_counts = {}
    for bitstring, c in counts.items():
        eb = extract_edge_bits(bitstring)
        edge_counts[eb] = edge_counts.get(eb, 0) + c

    # Success = probability mass on classically-good colorings.
    good_shots = sum(c for b, c in edge_counts.items() if b in GOOD_BITSTRINGS)
    success_prob = good_shots / shots

    most_likely = max(edge_counts.items(), key=lambda kv: kv[1])[0]

    print(f"Erdos problem #80 (tags: graph theory, ramsey theory; oeis: N/A)")
    print(f"Classical brute force: {N_GOOD} / {N_TOTAL} colorings of K4 are "
          f"triangle-free.")
    print(f"Grover iterations used: {iterations}")
    print(f"Most likely measured edge-coloring: {most_likely} "
          f"({'triangle-free' if most_likely in GOOD_BITSTRINGS else 'MONOCHROMATIC-CONTAINS'})")
    print(f"Fraction of {shots} shots landing on a classically-verified "
          f"triangle-free coloring: {success_prob:.4f}")

    # Amplified Grover search should concentrate most probability mass on
    # good colorings, far above the uniform baseline (N_GOOD / N_TOTAL).
    baseline = N_GOOD / N_TOTAL
    passed = (most_likely in GOOD_BITSTRINGS) and (success_prob > baseline)

    if passed:
        print("PASS")
    else:
        print("FAIL")

    return passed


if __name__ == "__main__":
    run()
