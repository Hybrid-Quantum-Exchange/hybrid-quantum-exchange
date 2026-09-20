"""
Erdos problem #804 -- quantum-testable sequence entry (honest best-effort / limitation noted)
================================================================================================

Source record checked: /home/user/manman4/erdosproblems/data/problems.yaml,
block "number: \"804\"":

    number: "804"
    prize: "no"
    informal_status: {state: "disproved", last_update: "2025-08-31"}
    formal_status:   {state: "unformalized"}
    status:          {state: "disproved", last_update: "2025-08-31"}
    oeis: ["possible"]
    tags: ["graph theory"]

LIMITATION (read before trusting the "PASS" below as evidence about problem 804
specifically): the `oeis` field for problem #804 is the literal string
"possible", not an actual OEIS A-number. There is no real integer sequence
attached to this problem in the source data, so there is nothing sequence-
specific to derive a classical property from or to build a genuine oracle
around. Per the task instructions ("If after reasonable effort no genuine
quantum circuit can be constructed for this problem's sequence ... write the
script anyway with your best honest attempt, note the limitation clearly").

Best honest attempt: the only real content problem #804 carries is its tag,
"graph theory", and its status ("disproved"). To still deliver a *genuine*,
non-fabricated, classically-checkable small computation with a real quantum
circuit behind it, this script falls back to a canonical finite graph-theory
decision problem that is at least in the right mathematical family: 3-vertex
graph triangle detection by exhaustive/Grover search over all 8 possible
edge-subsets of the 3 possible edges of K3 (vertices {0,1,2}, edges
{(0,1),(0,2),(1,2)}), searching for the unique edge-subset that forms a
triangle (all three edges present). This is NOT derived from problem 804's
actual mathematical content (there isn't any usable sequence to derive from);
it is a stand-in chosen to still exercise a real oracle + Grover diffusion
circuit rather than faking a number.

Classical property tested
--------------------------
Search space: 3 bits (q0,q1,q2) encode presence/absence of the 3 edges of the
triangle K3 on vertices {0,1,2}: q0 = edge(0,1), q1 = edge(0,2), q2 = edge(1,2).
Marked state: the unique subset where ALL three edges are present, i.e. the
bitstring "111" (integer 7), which is exactly the condition "this edge-subset
forms a triangle". This is computed classically from first principles in
`classical_answer()` below by brute-force enumeration of all 8 subsets.

Circuit
-------
A 3-qubit Grover search (oracle = multi-controlled-Z on |111>, diffuser =
standard Grover diffusion operator) run on the ideal AerSimulator. With N=8
and 1 marked state, the optimal number of Grover iterations is 2
(floor(pi/4 * sqrt(8)) = 2), which should amplify |111> to near-certainty.

Pass condition
--------------
The most frequently measured bitstring across 2048 shots must equal the
classical answer "111" (index 7).

Honesty note reported back: ran_ok reflects whether the script executed
without error; verified_against_classical reflects whether the quantum
result matched the classical brute-force answer for THIS fallback instance.
It does NOT certify anything about Erdos problem #804's actual (nonexistent
in source) OEIS sequence, which is the limitation stated above.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_answer():
    """Brute-force, from first principles: which 3-bit edge-subsets of K3
    form a triangle (all 3 edges present)? Returns the marked index/indices."""
    marked = []
    for state in range(8):
        # bit i (LSB) = presence of edge i, for i in {0,1,2}
        edges_present = [(state >> i) & 1 for i in range(3)]
        is_triangle = all(edges_present)  # all three edges must be present
        if is_triangle:
            marked.append(state)
    assert marked == [7], f"expected exactly the all-edges subset (7), got {marked}"
    return marked[0]


def build_oracle():
    """Phase oracle marking |111> (state 7) with a -1 phase, via multi-controlled Z."""
    qc = QuantumCircuit(3, name="oracle")
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    return qc


def build_diffuser():
    """Standard 3-qubit Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(3, name="diffuser")
    qc.h([0, 1, 2])
    qc.x([0, 1, 2])
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x([0, 1, 2])
    qc.h([0, 1, 2])
    return qc


def build_grover_circuit(iterations):
    qc = QuantumCircuit(3, 3)
    qc.h([0, 1, 2])
    oracle = build_oracle()
    diffuser = build_diffuser()
    for _ in range(iterations):
        qc.append(oracle.to_gate(), [0, 1, 2])
        qc.append(diffuser.to_gate(), [0, 1, 2])
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def main():
    target = classical_answer()
    target_bits = format(target, "03b")

    n = 8
    optimal_iterations = max(1, round((np.pi / 4) * np.sqrt(n)))  # -> 2 for N=8,M=1

    qc = build_grover_circuit(optimal_iterations)

    sim = AerSimulator()
    shots = 2048
    qc = qc.decompose()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit returns bitstrings as 'c2 c1 c0' (classical reg order, MSB..LSB
    # over reversed qubit order by default); build the measured bitstring
    # back into our q0,q1,q2 convention (LSB-first "q0 q1 q2" -> string).
    # Qiskit's default count key is big-endian over qubit index i.e.
    # key[0] corresponds to qubit 2, key[-1] to qubit 0. Convert to our
    # integer convention (bit i = qubit i) for a clean comparison.
    def key_to_int(key):
        # key is a string of length 3, key[0]=q2, key[1]=q1, key[2]=q0
        q2, q1, q0 = key[0], key[1], key[2]
        return int(q0) | (int(q1) << 1) | (int(q2) << 2)

    tally = {}
    for key, c in counts.items():
        val = key_to_int(key)
        tally[val] = tally.get(val, 0) + c

    most_common_state = max(tally, key=tally.get)
    most_common_prob = tally[most_common_state] / shots

    print(f"Classical answer (brute-force triangle search over K3 edge-subsets): "
          f"state={target} (bits={target_bits})")
    print(f"Grover iterations used: {optimal_iterations}")
    print(f"Raw counts (converted to q0,q1,q2 integer encoding): {tally}")
    print(f"Most frequent measured state: {most_common_state} "
          f"(bits={format(most_common_state, '03b')}), "
          f"probability={most_common_prob:.4f}")

    verified = (most_common_state == target) and (most_common_prob > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
