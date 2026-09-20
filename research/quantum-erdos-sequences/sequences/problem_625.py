"""
Erdos problem #625 (source: erdosproblems.com data, problems.yaml entry
`number: "625"`) — tags: ["graph theory", "chromatic number"], prize $1000,
status "solved" (2025-08-31), `oeis: ["N/A"]`.

LIMITATION: this problem's metadata record carries no OEIS sequence id at
all (the yaml field is the literal string "N/A"), so there is no integer
sequence for this entry to build a "sequence membership" style quantum test
around, and none was fabricated. Per the task's fallback instructions, this
script instead builds a genuine, small, finite, classically-checkable
instance of the property the problem's own tags name: graph chromatic
number, specifically "does this graph admit a proper 2-coloring (a legal
assignment of 2 colors to vertices such that no edge joins two same-colored
vertices)?" for the 4-cycle graph C4 (vertices 0,1,2,3; edges (0,1) (1,2)
(2,3) (3,0)).

Classical fact checked in this script (computed here, not looked up): C4 is
bipartite, so it is 2-colorable, and brute force over all 2^4 = 16 colorings
confirms there are exactly 2 proper 2-colorings: 0101 and 1010 (i.e. the two
alternating assignments). This is computed by exhaustive classical search
in `classical_valid_colorings()` below, independent of the quantum part.

QUANTUM PART: a Grover search circuit over the 4-qubit space of all 2^4
vertex colorings. A phase oracle is built from first principles: for each
edge (u, v) an ancilla qubit computes q_u XOR q_v via two CNOTs; the oracle
flips the phase of the state only when every ancilla is 1 (i.e. every edge's
endpoints differ, i.e. the coloring is proper), then uncomputes the
ancillas. The standard Grover diffusion operator is applied for the optimal
number of iterations for this problem size (N=16, M=2 valid colorings). The
circuit is run on the ideal AerSimulator and the measured output
distribution is checked against the classical answer: the two bitstrings
that classical brute force found valid must be the (near-)exclusive
high-probability outcomes.

PASS/FAIL: the script prints PASS if the two classically-valid colorings
account for the (overwhelming) majority of the measured shots and no
classically-invalid coloring appears among the top-2 measured outcomes;
otherwise FAIL.
"""

from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, transpile
from qiskit_aer import AerSimulator
import numpy as np
import itertools

# ---------------------------------------------------------------------------
# Graph definition: C4 (4-cycle)
# ---------------------------------------------------------------------------
NUM_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def classical_valid_colorings():
    """Brute-force, from first principles, every proper 2-coloring of C4."""
    valid = []
    for bits in itertools.product([0, 1], repeat=NUM_VERTICES):
        if all(bits[u] != bits[v] for (u, v) in EDGES):
            valid.append(bits)
    return valid


CLASSICAL_VALID = classical_valid_colorings()
# bit order for the circuit: qubit i <-> vertex i, and Qiskit reports
# bitstrings with qubit 0 as the rightmost character.
CLASSICAL_VALID_BITSTRINGS = {
    "".join(str(b) for b in reversed(bits)) for bits in CLASSICAL_VALID
}


def build_oracle(data, anc):
    """Phase oracle: flip sign of states where every edge's endpoints differ."""
    qc = QuantumCircuit(data, anc, name="oracle")
    # compute a_i = q_u XOR q_v for each edge into its own ancilla
    for i, (u, v) in enumerate(EDGES):
        qc.cx(data[u], anc[i])
        qc.cx(data[v], anc[i])
    # phase-flip when all ancillas are 1 (multi-controlled Z via H-MCX-H
    # on the last ancilla, controlled by the rest)
    controls = anc[:-1]
    target = anc[-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    # uncompute ancillas
    for i, (u, v) in reversed(list(enumerate(EDGES))):
        qc.cx(data[v], anc[i])
        qc.cx(data[u], anc[i])
    return qc


def build_diffuser(data):
    n = len(data)
    qc = QuantumCircuit(data, name="diffuser")
    qc.h(data)
    qc.x(data)
    qc.h(data[-1])
    qc.mcx(data[:-1], data[-1])
    qc.h(data[-1])
    qc.x(data)
    qc.h(data)
    return qc


def build_grover_circuit(iterations):
    data = QuantumRegister(NUM_VERTICES, "q")
    anc = AncillaRegister(len(EDGES), "a")
    qc = QuantumCircuit(data, anc)

    qc.h(data)  # uniform superposition over all colorings

    oracle = build_oracle(data, anc)
    diffuser = build_diffuser(data)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), list(data) + list(anc))
        qc.append(diffuser.to_instruction(), list(data))

    qc.measure_all()
    return qc


def optimal_iterations(n_total, n_marked):
    return max(1, round((np.pi / 4) * np.sqrt(n_total / n_marked)))


def main():
    n_total = 2 ** NUM_VERTICES
    n_marked = len(CLASSICAL_VALID)
    iterations = optimal_iterations(n_total, n_marked)

    print(f"Classical brute force: {n_marked} proper 2-colorings of C4 out of "
          f"{n_total} total assignments: {sorted(CLASSICAL_VALID_BITSTRINGS)}")
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(iterations)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 8192
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # strip the ancilla bits (they should be back to 0) and keep only the
    # 4 data-qubit bits, which are the leftmost 4 characters of the
    # space-separated "data ancilla" clbit string Qiskit's measure_all
    # produces when registers are measured together; instead, isolate by
    # register size explicitly.
    def data_bits(bitstring):
        # measure_all concatenates classical bits with qubit 0 rightmost
        # across the whole circuit (data then ancilla registers), so the
        # last NUM_VERTICES characters correspond to the data register.
        return bitstring[-NUM_VERTICES:]

    data_counts = {}
    for bitstring, count in counts.items():
        clean = bitstring.replace(" ", "")
        db = data_bits(clean)
        data_counts[db] = data_counts.get(db, 0) + count

    ranked = sorted(data_counts.items(), key=lambda kv: -kv[1])
    print("Top measured data-qubit outcomes (bitstring: count):")
    for bs, cnt in ranked[:6]:
        print(f"  {bs}: {cnt} ({100.0 * cnt / shots:.1f}%)")

    top2 = {bs for bs, _ in ranked[:2]}
    top2_mass = sum(cnt for bs, cnt in ranked[:2]) / shots

    verified = (
        top2 == CLASSICAL_VALID_BITSTRINGS
        and top2_mass > 0.85
    )

    print()
    print(f"Classically valid colorings (bitstrings): {sorted(CLASSICAL_VALID_BITSTRINGS)}")
    print(f"Top-2 quantum-measured outcomes:          {sorted(top2)}")
    print(f"Probability mass on the classically valid pair: {100.0 * top2_mass:.1f}%")

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
