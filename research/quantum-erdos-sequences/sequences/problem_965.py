"""
Erdos problem #965 (data/problems.yaml, erdosproblems.com dataset, number: "965")
-----------------------------------------------------------------------------
Status in the source dataset: informal_status = disproved (Lean-formalized
2026-08-23). Tags: ["ramsey theory"]. oeis: ["N/A"] -- the dataset records NO
OEIS sequence id for this problem. This is a documented limitation: the task
asked to identify a property from the problem's OEIS id(s) and tags, but
there is no OEIS id to derive one from here.

Honest fallback (best-effort attempt, not a literal instance of problem 965):
Since the problem's tag is "ramsey theory" and the classical anchor result of
small Ramsey theory is R(3,3) = 6 (the fact that every 2-coloring of the
edges of K_6 contains a monochromatic triangle, while K_5 has a coloring that
avoids one), we test the smallest non-trivial building block of that fact on
a real quantum circuit: whether a 2-coloring of the 3 edges of a SINGLE
triangle (K_3) is monochromatic (all 3 edges the same color).

Classical property being tested
--------------------------------
Let x = (e0, e1, e2) in {0,1}^3 be a 2-coloring of the 3 edges of a triangle
(0 = red, 1 = blue). x is "monochromatic" iff e0 == e1 == e2, i.e. x is 000
or 111. Among the 8 possible colorings, exactly 2 are monochromatic. This is
computed classically from first principles below (brute force over all 8
colorings, no shortcuts), and is the well known base fact underlying Ramsey
number R(3,3): a monochromatic triangle is precisely a triangle whose 3
edges got assigned the same color by a 2-coloring.

Quantum approach
-----------------
Grover's search algorithm on 3 qubits (search space N = 8) with an oracle
that marks the 2 monochromatic states |000> and |111>. With M = 2 marked
states out of N = 8, the optimal number of Grover iterations is
floor(pi/4 * sqrt(N/M)) = floor(pi/4 * 2) = 1, so a single Grover iteration
should amplify the marked states close to certainty. We build the oracle,
diffuser, and full Grover circuit in Qiskit, run it on the ideal AerSimulator,
and check that the two most frequent measured bitstrings across many shots
are exactly the classically-computed monochromatic colorings {000, 111}.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_monochromatic_colorings(n_edges: int = 3):
    """Brute-force, from first principles, all 2-colorings of n_edges edges
    of a triangle that are monochromatic (all edges same color)."""
    mono = []
    for bits in product([0, 1], repeat=n_edges):
        if len(set(bits)) == 1:
            mono.append(bits)
    return mono


def bits_to_bitstring(bits):
    # Qiskit bit ordering: qubit 0 is the rightmost character of the
    # classical register string.
    return "".join(str(b) for b in reversed(bits))


def build_oracle(marked_bitstrings, n_qubits):
    """Phase-flip oracle marking each given bitstring (qubit-0-rightmost
    convention) using multi-controlled Z gates, with X-gate sandwiching for
    zero-bits."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bitstring in marked_bitstrings:
        # bitstring[i] corresponds to qubit (n_qubits - 1 - i)
        zero_qubits = [n_qubits - 1 - i for i, c in enumerate(bitstring) if c == "0"]
        for q in zero_qubits:
            qc.x(q)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
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


def build_grover_circuit(marked_bitstrings, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_bitstrings, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    n_qubits = 3
    n_states = 2 ** n_qubits

    # --- classical ground truth, computed here, no OEIS lookup ---
    mono_bits = classical_monochromatic_colorings(n_qubits)
    marked_bitstrings = sorted(bits_to_bitstring(b) for b in mono_bits)
    n_marked = len(marked_bitstrings)
    assert marked_bitstrings == ["000", "111"], marked_bitstrings

    # optimal Grover iteration count for N=8, M=2
    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(n_states / n_marked))))

    qc = build_grover_circuit(marked_bitstrings, n_qubits, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    top_states = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top2 = sorted(state for state, _ in top_states[:2])

    marked_shots = sum(counts.get(s, 0) for s in marked_bitstrings)
    marked_fraction = marked_shots / shots

    print(f"Erdos problem #965: oeis=['N/A'] (no OEIS id in source data), tags=['ramsey theory']")
    print(f"Classical monochromatic triangle colorings (from first principles): {marked_bitstrings}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")
    print(f"Top-2 measured bitstrings: {top2}")
    print(f"Fraction of shots landing on a marked (monochromatic) state: {marked_fraction:.4f}")

    verified = (top2 == marked_bitstrings) and (marked_fraction > 0.8)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
