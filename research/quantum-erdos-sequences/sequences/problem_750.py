"""
Erdos problem #750 (per erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: '750'", tags ["graph theory", "chromatic number"]).

LIMITATION: problem #750's entry lists oeis: ["N/A"] -- there is no OEIS sequence
attached to this problem in the source data, so this script cannot build a circuit
that tests membership in, or a term of, an actual OEIS sequence tied to #750. This
is therefore a best-honest-attempt fallback: it stays faithful to the problem's own
tag ("chromatic number" in graph theory) by testing a small, finite, genuinely
computable graph-coloring property with a real Grover search circuit, rather than
fabricating or mis-citing an OEIS id that isn't there.

Classical property tested
--------------------------
Graph G = the 4-cycle C4 on vertices {0,1,2,3} with edges (0,1),(1,2),(2,3),(3,0).
Property: G is 2-colorable (its chromatic number is 2, i.e. it is bipartite), and
the set of proper 2-colorings (using colors {0,1}, one bit per vertex) is exactly
{0101, 1010} in (q3 q2 q1 q0) order -- out of all 16 possible colorings.

This is computed from first principles classically in `classical_valid_colorings()`
below by brute-force enumeration over all 2^4 colorings, checking every edge has
differently-colored endpoints. The result (exactly 2 valid colorings) is then the
ground truth the quantum circuit is checked against.

Quantum approach
-----------------
A Grover search circuit over the 4 vertex-color qubits. The oracle computes, into
ancilla qubits, the XOR of the two endpoint colors for each of the 4 edges (XOR=1
means "properly colored" for that edge), phase-flips the marked state when all 4
edge-ancillas are 1 (via a multi-controlled Z against a phase-kickback qubit), and
then uncomputes the ancillas. The standard 4-qubit Grover diffuser follows. With
N=16 states and M=2 marked solutions, the optimal iteration count is
floor(pi/4 * sqrt(N/M)) = 2.

After running on the ideal AerSimulator, the two most frequent measured bitstrings
must be exactly the classical valid-coloring set, which is what PASS/FAIL checks.
"""

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
N_VERTICES = 4


def classical_valid_colorings():
    """Brute-force, from first principles: all proper 2-colorings of C4."""
    valid = []
    for assignment in range(2 ** N_VERTICES):
        colors = [(assignment >> v) & 1 for v in range(N_VERTICES)]
        if all(colors[u] != colors[v] for (u, v) in EDGES):
            # bitstring in Qiskit convention q_{n-1}...q_0
            bitstring = "".join(str(colors[v]) for v in reversed(range(N_VERTICES)))
            valid.append(bitstring)
    return sorted(valid)


def build_oracle(q, anc, out):
    qc = QuantumCircuit(q, anc, out, name="oracle")
    # compute edge-XOR into ancillas
    for i, (u, v) in enumerate(EDGES):
        qc.cx(q[u], anc[i])
        qc.cx(q[v], anc[i])
    # phase-flip when all 4 ancillas are 1, via multi-controlled X on the
    # phase-kickback qubit `out` (prepared in |-> before the oracle runs)
    qc.mcx(list(anc), out[0])
    # uncompute
    for i, (u, v) in reversed(list(enumerate(EDGES))):
        qc.cx(q[v], anc[i])
        qc.cx(q[u], anc[i])
    return qc


def build_diffuser(q):
    n = len(q)
    qc = QuantumCircuit(q, name="diffuser")
    qc.h(q)
    qc.x(q)
    qc.h(q[n - 1])
    qc.mcx(list(q[0 : n - 1]), q[n - 1])
    qc.h(q[n - 1])
    qc.x(q)
    qc.h(q)
    return qc


def build_grover_circuit(iterations):
    q = QuantumRegister(N_VERTICES, "q")
    anc = QuantumRegister(len(EDGES), "anc")
    out = QuantumRegister(1, "out")
    qc = QuantumCircuit(q, anc, out)

    qc.h(q)
    qc.x(out)
    qc.h(out)

    oracle = build_oracle(q, anc, out)
    diffuser = build_diffuser(q)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), list(q) + list(anc) + list(out))
        qc.append(diffuser.to_instruction(), list(q))

    qc.measure_all()
    return qc, q


def main():
    classical = classical_valid_colorings()
    n = 2 ** N_VERTICES
    m = len(classical)
    import math

    iterations = max(1, round((math.pi / 4) * math.sqrt(n / m)))

    qc, q = build_grover_circuit(iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Extract just the q3q2q1q0 vertex-color bits (measure_all appends a
    # classical bit per qubit in circuit order: q0..q3, anc0..anc3, out;
    # Qiskit count keys are reversed, most-significant first, so the
    # rightmost 4 characters, after also dropping ancilla/out bits, need
    # correct slicing -- do it explicitly via bit positions.
    n_q, n_anc, n_out = N_VERTICES, len(EDGES), 1
    total_bits = n_q + n_anc + n_out

    def extract_vertex_bits(bitstring):
        # bitstring is MSB..LSB over all qubits in registration order
        # q0..q3, anc0..anc3, out0 -> reversed for printing
        full = bitstring.replace(" ", "")
        assert len(full) == total_bits
        # rightmost n_q characters correspond to q0..q3 with q0 as the very
        # last character (Qiskit's little-endian convention)
        vertex_part = full[-n_q:]
        return vertex_part  # already in q3 q2 q1 q0 order (MSB..LSB of that slice)

    vertex_counts = {}
    for bitstring, freq in counts.items():
        vbits = extract_vertex_bits(bitstring)
        vertex_counts[vbits] = vertex_counts.get(vbits, 0) + freq

    top2 = sorted(vertex_counts.items(), key=lambda kv: -kv[1])[:2]
    top2_bitstrings = sorted(b for b, _ in top2)

    print("Classical valid 2-colorings of C4:", classical)
    print("Grover iterations used:", iterations)
    print("Top measured vertex-coloring bitstrings (by frequency):", top2)

    verified = top2_bitstrings == classical
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
