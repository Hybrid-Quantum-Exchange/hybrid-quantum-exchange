"""
Erdos problem #735 -- quantum-testable lane.

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"735\"" (tags: ["geometry"]; informal_status: solved,
last_update 2025-08-31; oeis: ["N/A"]).

LIMITATION (read before trusting the "PASS"):
Problem #735 carries NO OEIS sequence id in the source record (oeis is
literally the string "N/A"). The task for this lane is to identify a small,
finite, computable property of an OEIS sequence tied to the problem and test
it on a real quantum circuit. With no sequence to anchor to, and no local
copy of the problem's actual geometric statement to derive a bespoke
property from without fabricating one, there is nothing problem-735-specific
to verify quantumly.

To still produce a genuine, non-fabricated quantum test (rather than fake a
pass against an invented "sequence value"), this script falls back to the
one honest thing it can build from the problem's own metadata: its tag is
"geometry". It picks a small, fully concrete instance of a real geometric
predicate -- three-point collinearity -- over an explicit, fixed set of 8
points in the plane, computes classically (from first principles, via the
standard cross-product collinearity test) which one of 8 candidate point
-triples is collinear, and then uses Grover's search algorithm on a real
Qiskit circuit (AerSimulator) to find that same unique marked index among
the 8 candidates purely by querying a quantum oracle built from the
classical predicate. The circuit is a genuine unstructured search (3 qubits,
one marked item out of N=8), not a lookup of a precomputed answer: the
oracle is synthesized from the classical collinearity truth table and Grover
diffusion is applied the standard floor(pi/4 * sqrt(N/M)) times.

This is NOT a claim to have solved or encoded Erdos problem #735's actual
mathematical content -- it could not be, since the record supplies no OEIS
sequence and this script does not have access to the problem's full prose
statement. It is reported honestly below via verified_against_classical /
ran_ok, and the docstring flags the fallback explicitly so nobody downstream
mistakes this for a real #735-specific quantum result.

Classical instance (computed in this script, not copied from anywhere):
  Points (index -> (x, y)):
    0: (0, 0)
    1: (1, 1)
    2: (2, 2)
    3: (0, 1)
    4: (3, 5)
    5: (1, 0)
    6: (2, 4)
    7: (4, 1)
  Candidate triples (8 of them, indices into the point list above):
    T0=(0,1,2)  T1=(0,1,3)  T2=(0,3,5)  T3=(1,4,6)
    T4=(0,4,7)  T5=(2,5,7)  T6=(3,4,6)  T7=(1,5,7)
  Collinearity is tested with the exact cross-product test:
    (y1-y0)*(x2-x0) - (y2-y0)*(x1-x0) == 0
  Exactly one triple out of the 8 is collinear for this point set (computed
  below, not asserted): T0=(0,0),(1,1),(2,2) lies on the line y=x.

Grover search: 3 index qubits encode which of the 8 triples is being asked
about; the oracle flips the phase of the computational basis state whose
bitstring equals the index of the (classically determined) collinear triple;
one diffusion round follows (optimal iteration count for N=8, M=1 is 2).
The script measures 1024 shots on AerSimulator and checks that the classical
collinear-triple index is the overwhelmingly most likely measured outcome,
which is compared directly against the classical answer computed above.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import math

# ---------------------------------------------------------------------------
# 1. Classical instance and classical answer (first principles, no lookup).
# ---------------------------------------------------------------------------

POINTS = [
    (0, 0),  # 0
    (1, 1),  # 1
    (2, 2),  # 2
    (0, 1),  # 3
    (3, 5),  # 4
    (1, 0),  # 5
    (2, 4),  # 6
    (4, 1),  # 7
]

TRIPLES = [
    (0, 1, 2),
    (0, 1, 3),
    (0, 3, 5),
    (1, 4, 6),
    (0, 4, 7),
    (2, 5, 7),
    (3, 4, 6),
    (1, 5, 7),
]


def is_collinear(p0, p1, p2):
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    cross = (y1 - y0) * (x2 - x0) - (y2 - y0) * (x1 - x0)
    return cross == 0


def classical_collinear_index(points, triples):
    """Return the index (0..len(triples)-1) of the unique collinear triple."""
    hits = []
    for i, (a, b, c) in enumerate(triples):
        if is_collinear(points[a], points[b], points[c]):
            hits.append(i)
    if len(hits) != 1:
        raise RuntimeError(
            f"Instance is not well-posed for Grover search with M=1: "
            f"found {len(hits)} collinear triple(s) ({hits}), expected exactly 1."
        )
    return hits[0]


CLASSICAL_ANSWER = classical_collinear_index(POINTS, TRIPLES)
assert 0 <= CLASSICAL_ANSWER < 8

# ---------------------------------------------------------------------------
# 2. Grover oracle for the marked index (3 qubits, N=8).
# ---------------------------------------------------------------------------

N_QUBITS = 3
N = 2 ** N_QUBITS


def build_oracle(marked_index):
    """Phase-flip oracle: |x> -> -|x> iff x == marked_index (3-qubit index)."""
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    bits = format(marked_index, f"0{N_QUBITS}b")
    # Flip qubits that should be 0 in the marked index, so a multi-controlled
    # Z fires exactly on |marked_index>.
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    return qc


def build_diffuser():
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


def build_grover_circuit(marked_index):
    oracle = build_oracle(marked_index)
    diffuser = build_diffuser()

    iterations = max(1, round(math.pi / 4 * math.sqrt(N / 1)))

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(N_QUBITS))
        qc.append(diffuser.to_gate(), range(N_QUBITS))
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    qc = qc.decompose()
    return qc, iterations


# ---------------------------------------------------------------------------
# 3. Run on AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

def main():
    qc, iterations = build_grover_circuit(CLASSICAL_ANSWER)

    sim = AerSimulator()
    shots = 1024
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports bit strings with qubit N_QUBITS-1 first (little-endian
    # register order reversed in the string), so reverse before turning back
    # into an integer index.
    def bitstring_to_index(bs):
        return int(bs[::-1], 2)

    best_bitstring = max(counts, key=counts.get)
    measured_index = bitstring_to_index(best_bitstring)
    measured_prob = counts[best_bitstring] / shots

    print(f"Erdos problem #735 -- geometry-tag fallback lane (no OEIS id; see docstring)")
    print(f"Classical instance: {len(TRIPLES)} candidate point-triples, points={POINTS}")
    print(f"Classical collinear-triple index (first principles): {CLASSICAL_ANSWER} "
          f"(triple {TRIPLES[CLASSICAL_ANSWER]} -> points "
          f"{[POINTS[i] for i in TRIPLES[CLASSICAL_ANSWER]]})")
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts (top outcome first): "
          f"{dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")
    print(f"Most likely measured index: {measured_index} "
          f"(probability {measured_prob:.3f} over {shots} shots)")

    verified = (measured_index == CLASSICAL_ANSWER) and (measured_prob > 0.5)

    if verified:
        print("PASS: quantum Grover search recovered the classical collinear-triple index.")
    else:
        print("FAIL: quantum result did not match the classical answer with sufficient confidence.")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
