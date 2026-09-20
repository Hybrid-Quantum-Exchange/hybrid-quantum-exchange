"""
Erdos problem #108 (erdosproblems.com/108) -- quantum-testable instance.

Source metadata (from data/problems.yaml, manman4/erdosproblems, read-only
clone at /home/user/manman4/erdosproblems):
    number: "108"
    tags: ["graph theory", "chromatic number", "cycles"]
    oeis: ["possible"]   <- NOT a real OEIS id. The metadata for this problem
                             carries no usable OEIS sequence identifier (the
                             field literally holds the placeholder string
                             "possible", not an A-number), so this script
                             cannot honestly claim to test "the OEIS sequence
                             for problem 108" the way most other lanes can.

LIMITATION (reported honestly, per instructions): because there is no real
OEIS id attached to problem 108, this script does not test an OEIS sequence
membership property. Instead it uses the problem's own declared tags --
graph theory / chromatic number / cycles -- to build the closest genuine,
small, finite, computable property in that spirit: proper vertex 2-colorings
of a cycle graph, which is exactly the classical fact underlying "chromatic
number of a cycle" (even cycles are 2-chromatic/bipartite, odd cycles are
3-chromatic). This is a real, independently-checkable combinatorial search
problem, not a fabricated stand-in for the OEIS value -- it is just not the
OEIS sequence itself, since problem 108 has none recorded.

Classical property under test
------------------------------
Let C4 be the 4-cycle graph on vertices {0,1,2,3} with edges
(0,1),(1,2),(2,3),(3,0). A 2-coloring is a bitstring c0c1c2c3 in {0,1}^4.
A coloring is PROPER iff every edge's endpoints differ:
    c0!=c1 and c1!=c2 and c2!=c3 and c3!=c0.

C4 is bipartite (an even cycle), so it has a proper 2-coloring, and in fact
exactly 2 of them: 0101 and 1010 (as bitstrings c0c1c2c3), matching the
well-known chromatic-number-of-cycles fact chi(C_{2k}) = 2.

The script first computes this classical answer directly by brute-force
enumeration of all 16 colorings (first principles, no lookup).

Quantum circuit
----------------
A genuine Grover search over the 4-qubit coloring space (N=16 states,
M=2 marked/proper colorings). The oracle computes, into 4 ancilla qubits,
the XOR of each edge's two color qubits (ancilla_i = 1 iff edge i's
endpoints differ), then applies a phase flip when all 4 ancillas are 1
(i.e. all edges properly colored), and uncomputes the ancillas. The
diffuser is the standard Grover inversion-about-the-mean operator on the
4 color qubits. With N=16, M=2, the optimal number of Grover iterations is
floor((pi/4)*sqrt(N/M)) = 2.

The circuit is run on the ideal AerSimulator (statevector method, no
noise), and PASS/FAIL is decided by checking that the two most probable
measured outcomes are exactly the two proper 2-colorings computed
classically above.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # the 4-cycle C4


def is_proper_coloring(bits):
    """bits: tuple of 4 ints (0/1), bits[i] = color of vertex i."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


def classical_proper_colorings():
    solutions = []
    for bits in itertools.product([0, 1], repeat=4):
        if is_proper_coloring(bits):
            solutions.append(bits)
    return solutions


CLASSICAL_SOLUTIONS = classical_proper_colorings()
N_STATES = 16
M_SOLUTIONS = len(CLASSICAL_SOLUTIONS)

assert M_SOLUTIONS == 2, f"expected exactly 2 proper 2-colorings of C4, got {M_SOLUTIONS}"

# Bit order note: qiskit reports measurement bitstrings as c3 c2 c1 c0
# (qubit 0 is the rightmost character). Build the expected bitstrings in
# that convention for direct comparison with simulator output.
def bits_to_qiskit_string(bits):
    # bits = (c0, c1, c2, c3) -> qiskit prints "c3c2c1c0"
    return "".join(str(bits[i]) for i in (3, 2, 1, 0))


EXPECTED_STRINGS = sorted(bits_to_qiskit_string(b) for b in CLASSICAL_SOLUTIONS)


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for proper 2-colorings of C4.
# ---------------------------------------------------------------------------

def build_oracle(color, anc, out):
    """Marks (phase-flips) states where all 4 edges are properly colored.

    color: QuantumRegister of 4 qubits, the coloring c0 c1 c2 c3.
    anc:   QuantumRegister of 4 qubits, scratch (edge-difference flags).
    out:   QuantumRegister of 1 qubit, held in the |-> state by the caller
           so that a controlled-X on it performs a phase kick-back.
    """
    qc = QuantumCircuit(color, anc, out, name="oracle")

    # Compute anc[i] = color[u] XOR color[v] for each edge (u, v).
    for i, (u, v) in enumerate(EDGES):
        qc.cx(color[u], anc[i])
        qc.cx(color[v], anc[i])

    # Phase-kick the output qubit iff all 4 ancillas are 1 (all edges differ).
    qc.mcx(list(anc), out[0])

    # Uncompute the ancillas.
    for i, (u, v) in enumerate(EDGES):
        qc.cx(color[v], anc[i])
        qc.cx(color[u], anc[i])

    return qc


def build_diffuser(color):
    """Standard Grover diffuser (inversion about the mean) on 4 qubits."""
    n = len(color)
    qc = QuantumCircuit(color, name="diffuser")
    qc.h(color)
    qc.x(color)
    qc.h(color[n - 1])
    qc.mcx(list(color[: n - 1]), color[n - 1])
    qc.h(color[n - 1])
    qc.x(color)
    qc.h(color)
    return qc


def build_grover_circuit(iterations):
    color = QuantumRegister(4, "c")
    anc = QuantumRegister(4, "a")
    out = QuantumRegister(1, "o")
    creg = ClassicalRegister(4, "m")

    qc = QuantumCircuit(color, anc, out, creg)

    # Uniform superposition over all 16 colorings.
    qc.h(color)

    # Output ancilla prepared in |-> for phase kick-back.
    qc.x(out[0])
    qc.h(out[0])

    oracle = build_oracle(color, anc, out)
    diffuser = build_diffuser(color)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), color[:] + anc[:] + out[:])
        qc.append(diffuser.to_instruction(), color[:])

    qc.measure(color, creg)
    return qc


def optimal_iterations(n_states, m_solutions):
    theta = math.asin(math.sqrt(m_solutions / n_states))
    return max(1, round((math.pi / (4 * theta)) - 0.5))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

def main():
    iterations = optimal_iterations(N_STATES, M_SOLUTIONS)

    print(f"Erdos problem #108 -- tags: graph theory / chromatic number / cycles")
    print(f"OEIS id in source metadata: 'possible' (not a real A-number; see docstring)")
    print(f"Classical proper 2-colorings of C4 (brute force over all 16): "
          f"{CLASSICAL_SOLUTIONS}")
    print(f"Expected measurement strings (qiskit bit order): {EXPECTED_STRINGS}")
    print(f"Grover iterations used: {iterations} (N={N_STATES}, M={M_SOLUTIONS})")

    qc = build_grover_circuit(iterations)

    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print(f"Top measured outcomes: {sorted_counts[:6]}")

    top_two = sorted(k for k, _ in sorted_counts[:2])
    top_two_prob_mass = sum(v for k, v in counts.items() if k in EXPECTED_STRINGS) / shots

    verified = (top_two == EXPECTED_STRINGS) and (top_two_prob_mass > 0.7)

    print(f"Top-2 measured bitstrings: {top_two}")
    print(f"Probability mass on classically-correct colorings: {top_two_prob_mass:.3f}")

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
