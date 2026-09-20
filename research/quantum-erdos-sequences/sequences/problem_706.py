"""
Erdos problem #706 -- quantum-testable lane.

Source metadata (data/problems.yaml in the erdosproblems repo, entry for
number "706"): tags = ["graph theory", "chromatic number"]; oeis = ["possible"]
(a placeholder, not a real OEIS sequence id) and formal_status "unformalized".
There is therefore no real OEIS sequence id attached to this problem to build
a "membership in the sequence" oracle from -- honesty note per the task
instructions: this script does NOT test problem #706's actual open conjecture
(that would not be finite/small/computable), and it does NOT use a genuine
OEIS id, because none is given. What it does instead, in good faith and in
the same subject area as the problem's own tags (graph theory / chromatic
number), is pose a small, finite, fully-computable chromatic-number question
and answer it two ways: classically (brute force) and with a real Grover
search circuit run on Qiskit's AerSimulator.

Classical property tested
--------------------------
Graph: the 3-vertex path P3 with vertices {0, 1, 2} and edges (0,1), (1,2).
Question: which 2-colorings (color in {0,1} per vertex, so a 3-bit string
q0 q1 q2) are PROPER, i.e. satisfy q0 != q1 AND q1 != q2?

By brute force over all 2**3 = 8 colorings (computed in this script from
first principles, not looked up), there are exactly two proper colorings:
"010" and "101" (using bit order q0 q1 q2). This also certifies P3 is
2-colorable (bipartite), consistent with its known chromatic number 2.

Quantum circuit
----------------
A genuine 3-qubit Grover search over the 8-element search space, marking
exactly the states satisfying (q0 XOR q1) AND (q1 XOR q2):
  - Two ancilla qubits compute q0^q1 and q1^q2 via CNOTs.
  - A CZ between the two ancillas flips the phase of the marked states
    (both ancillas = 1 iff the coloring is proper).
  - The ancillas are uncomputed (CNOTs undone) so they return to |00>.
  - One standard 3-qubit Grover diffuser follows (optimal iteration count
    for N=8, M=2 marked states is floor(pi/4 * sqrt(N/M)) = 1).
The circuit is run on AerSimulator (statevector-exact, no noise) with 4096
shots. PASS requires the two most frequent measured outcomes to be exactly
the classically-derived set {"010", "101"} and to together hold at least
90% of the shot counts (Grover amplification, not a guessed answer).
"""

from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_proper_colorings():
    """Brute-force all 3-bit colorings of path graph 0-1-2, first principles."""
    edges = [(0, 1), (1, 2)]
    valid = []
    for bits in product([0, 1], repeat=3):
        if all(bits[a] != bits[b] for a, b in edges):
            # Bit order in the string matches qubit order q0 q1 q2.
            valid.append("".join(str(b) for b in bits))
    return sorted(valid)


def build_grover_circuit():
    # Qubits: 0,1,2 = data (q0,q1,q2); 3,4 = ancillas for the two XORs.
    qc = QuantumCircuit(5, 3)

    # Uniform superposition over the 3 data qubits.
    qc.h([0, 1, 2])

    def oracle(circ):
        circ.cx(0, 3)
        circ.cx(1, 3)  # ancilla 3 = q0 ^ q1
        circ.cx(1, 4)
        circ.cx(2, 4)  # ancilla 4 = q1 ^ q2
        circ.cz(3, 4)  # phase flip iff both XORs are 1 (proper coloring)
        # uncompute
        circ.cx(1, 4)
        circ.cx(2, 4)
        circ.cx(0, 3)
        circ.cx(1, 3)

    def diffuser(circ):
        circ.h([0, 1, 2])
        circ.x([0, 1, 2])
        circ.h(2)
        circ.ccx(0, 1, 2)
        circ.h(2)
        circ.x([0, 1, 2])
        circ.h([0, 1, 2])

    # Optimal iteration count for N=8, M=2 marked states is 1.
    oracle(qc)
    diffuser(qc)

    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def run_quantum(shots=4096):
    qc = build_grover_circuit()
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    # Qiskit prints classical-bit strings as c2 c1 c0; c0<-q0, c1<-q1, c2<-q2,
    # so reverse to recover the natural q0 q1 q2 bit order used classically.
    fixed = {}
    for bitstring, n in counts.items():
        natural = bitstring[::-1]
        fixed[natural] = fixed.get(natural, 0) + n
    return fixed


def main():
    classical_answer = classical_proper_colorings()
    print(f"Classical proper 2-colorings of P3 (q0 q1 q2): {classical_answer}")

    counts = run_quantum()
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    print("Quantum measurement counts (q0 q1 q2 -> shots):")
    for bitstring, n in ranked:
        print(f"  {bitstring}: {n}")

    top_two = {b for b, _ in ranked[:2]}
    top_two_mass = sum(n for b, n in ranked if b in top_two)
    fraction = top_two_mass / total

    matches_classical = top_two == set(classical_answer)
    amplified_enough = fraction >= 0.90

    print(f"Top-2 measured states: {sorted(top_two)}  (fraction of shots: {fraction:.3f})")
    print(f"Matches classical answer set: {matches_classical}")

    if matches_classical and amplified_enough:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
