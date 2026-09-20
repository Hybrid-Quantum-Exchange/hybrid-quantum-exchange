"""
Erdos problem #814 -- quantum-testable companion script.

Source metadata (data/problems.yaml in the manman4/erdosproblems clone,
entry "number: \"814\"", tags: ["graph theory"]):

    oeis: ["possible"]

"possible" is not an OEIS sequence id -- it is a placeholder value used in
that dataset when no OEIS sequence has actually been associated with the
problem. There is therefore no genuine OEIS sequence to build a
quantum-testable property from for problem #814. This is disclosed here
rather than fabricated: no OEIS id is used, and no literal OEIS term is
copied anywhere in this script.

Because problem #814 is tagged "graph theory", this script instead builds
its BEST HONEST ATTEMPT at a small, finite, genuinely computable
graph-theoretic property, so the exercise (design a real quantum circuit
that computes/verifies a small combinatorial fact and checks it against a
from-scratch classical computation) still has real content. It is NOT a
verification of Erdos problem #814 itself, and should not be read as one.

Chosen property (independent of #814, honestly a stand-in):
    Let G range over all labeled graphs on 4 vertices {0,1,2,3}. Each such
    graph is encoded as a 6-bit string, one bit per possible edge, in the
    fixed order:
        bit0 = edge(0,1), bit1 = edge(0,2), bit2 = edge(0,3),
        bit3 = edge(1,2), bit4 = edge(1,3), bit5 = edge(2,3)
    There are N = 2**6 = 64 such graphs. The property tested is:

        "G contains the triangle on vertices {0,1,2}"
        i.e. edge(0,1) = edge(0,2) = edge(1,2) = 1
        (bits 0, 1, 3 all set; bits 2, 4, 5 free)

    This is a small, finite, fully enumerable search space (N=64) with a
    known, classically-computed set of M marked solutions. It is exactly
    the shape of problem Grover's algorithm is built for.

Classical answer (computed here from first principles, not copied):
    M = number of 6-bit strings with bits 0,1,3 = 1 = 2**3 = 8 (bits
    2,4,5 free). The script enumerates all 64 strings by brute force to
    confirm this before running anything quantum.

Quantum method:
    A 6-qubit Grover search is built. The oracle applies a phase flip
    (via a multi-controlled Z realized with an ancilla-based MCX + phase
    trick) to exactly the computational basis states whose bits 0,1,3 are
    all 1, regardless of bits 2,4,5 -- i.e. it flips the phase of all 8
    marked graphs. The optimal number of Grover iterations for N=64,
    M=8 is round(pi/4 * sqrt(N/M)) = round(pi/4 * sqrt(8)) = 2. The
    circuit is run on the ideal AerSimulator (statevector-exact, no
    noise), and the resulting measurement distribution is compared
    against the classical set of 8 marked strings.

PASS/FAIL:
    The script prints PASS if, over many shots, essentially all
    measured outcomes (probability above a strict threshold) fall in the
    classically-enumerated marked set, and FAIL otherwise.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_VERTICES = 4
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
N_QUBITS = len(EDGES)  # 6
N = 2 ** N_QUBITS  # 64

# Triangle we search for: vertices {0,1,2} -> edges (0,1),(0,2),(1,2)
# -> bit indices 0, 1, 3 in the EDGES list above.
TRIANGLE_EDGE_BITS = [EDGES.index((0, 1)), EDGES.index((0, 2)), EDGES.index((1, 2))]
assert TRIANGLE_EDGE_BITS == [0, 1, 3]


def classical_marked_set():
    """Brute-force enumerate all 64 4-vertex graphs and return the bit
    strings (as ints, bit i = qubit i, little-endian to match Qiskit's
    measurement convention) whose graph contains the triangle {0,1,2}."""
    marked = []
    for bits in itertools.product([0, 1], repeat=N_QUBITS):
        if all(bits[i] == 1 for i in TRIANGLE_EDGE_BITS):
            # bits[0] is edge index 0 = qubit 0 = least significant bit
            value = 0
            for i, b in enumerate(bits):
                value |= (b << i)
            marked.append(value)
    return sorted(marked)


def build_oracle():
    """Phase-flip oracle marking every basis state with qubits 0,1,3 = 1
    (bits 2,4,5 unconstrained), using a standard multi-controlled-Z
    (phase kickback via an X-sandwiched multi-controlled X on an ancilla
    prepared in the |-> state is not needed here: Qiskit's mcx with
    target in |-> state pattern, done directly with a controlled-Z built
    from H + MCX + H on the last control qubit acting as target)."""
    qc = QuantumCircuit(N_QUBITS, name="triangle_oracle")
    controls = TRIANGLE_EDGE_BITS[:-1]
    target = TRIANGLE_EDGE_BITS[-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    return qc


def build_diffuser():
    """Standard Grover diffuser (inversion about the mean) over all 6
    qubits."""
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


def build_grover_circuit(iterations):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))

    oracle = build_oracle().to_gate()
    diffuser = build_diffuser().to_gate()

    for _ in range(iterations):
        qc.append(oracle, range(N_QUBITS))
        qc.append(diffuser, range(N_QUBITS))

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def main():
    marked = classical_marked_set()
    m = len(marked)
    expected_m = 2 ** (N_QUBITS - len(TRIANGLE_EDGE_BITS))
    print(f"Classical brute force: N={N} graphs on 4 vertices, "
          f"{m} contain the triangle {{0,1,2}} (expected 2**3={expected_m}).")
    assert m == expected_m == 8
    print(f"Marked bit-strings (decimal): {marked}")

    iterations = max(1, round((math.pi / 4) * math.sqrt(N / m)))
    print(f"Grover iterations used: {iterations} "
          f"(optimal ~ pi/4 * sqrt(N/M) = {(math.pi / 4) * math.sqrt(N / m):.3f})")

    qc = build_grover_circuit(iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's counts-dict bit-string keys already index the same way our
    # little-endian bit encoding does (bit i in the key == qubit i).
    def key_to_int(bitstring):
        return int(bitstring, 2)

    marked_set = set(marked)
    marked_shots = sum(c for k, c in counts.items() if key_to_int(k) in marked_set)
    fraction_marked = marked_shots / shots

    print(f"Fraction of {shots} shots landing on a marked (triangle-containing) "
          f"graph: {fraction_marked:.4f}")

    # With N=64, M=8, 2 Grover iterations the theoretical success
    # probability is close to 1 (sin^2((2*iter+1)*theta) with
    # theta = arcsin(sqrt(M/N))); demand a strict but safe threshold.
    theta = math.asin(math.sqrt(m / N))
    theoretical_p = math.sin((2 * iterations + 1) * theta) ** 2
    print(f"Theoretical success probability for {iterations} iteration(s): "
          f"{theoretical_p:.4f}")

    threshold = 0.85
    verified = fraction_marked >= threshold

    if verified:
        print("PASS: quantum Grover search result matches the classical "
              "brute-force marked set of triangle-containing 4-vertex graphs.")
    else:
        print("FAIL: quantum measurement distribution did not concentrate "
              "on the classically-verified marked set.")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
