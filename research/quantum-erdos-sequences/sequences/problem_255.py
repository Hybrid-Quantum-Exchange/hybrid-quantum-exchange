"""
Erdos problem #255 (per manman4/erdosproblems data/problems.yaml, entry
"number: '255'") is tagged ["discrepancy"], status "proved (Lean)", and its
`oeis` field is literally `["N/A"]` -- there is no OEIS sequence attached to
this problem. That means the "identify a property from its OEIS id(s)" step
in the assignment has no id to work from.

LIMITATION (stated honestly, per instructions): this script does NOT test any
OEIS sequence for problem 255, because none exists in the source data. What
follows is the closest genuine, small, computable property in the same
mathematical area (discrepancy of +-1 sequences), used as the best-effort
substitute the task instructions call for when no OEIS id is available. It is
a real instance of the Erdos Discrepancy Problem (a different, famous Erdos
problem on {-1,+1} sequences and discrepancy over homogeneous arithmetic
progressions), *not* a derivation from problem 255's own OEIS data.

Classical property tested (computed from first principles below, not copied
from any table):

    For sequences x_1..x_4 in {-1,+1}, consider all homogeneous arithmetic
    progressions x_d, x_2d, x_3d, ... for d = 1..4 and all partial sums along
    each. The discrepancy of x is the maximum absolute partial sum over all
    such progressions. We ask: which of the 16 possible +-1 sequences of
    length 4 have discrepancy <= 1?

    Brute-force classical search (done in this script) finds exactly 2 of the
    16 sequences satisfy this: (+1,-1,-1,+1) and (-1,+1,+1,-1).

Quantum approach: Grover's algorithm on 4 qubits (one per sequence position,
0 -> +1, 1 -> -1) with an oracle that marks exactly the classically-verified
low-discrepancy bitstrings ("0110" and "1001"), amplifying their probability
so that measurement returns one of them with high probability on the ideal
AerSimulator.

The script:
  1. Classically brute-forces the true set of valid ("low discrepancy")
     bitstrings for n=4 from first principles.
  2. Builds a Grover oracle + diffuser for exactly that marked set.
  3. Runs the circuit on AerSimulator (statevector-based, ideal, noiseless).
  4. Compares the most-probable measured bitstring(s) against the classical
     answer and prints PASS/FAIL.
"""

import itertools
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_discrepancy_solutions(n=4):
    """Brute force all +-1 sequences of length n with discrepancy <= 1
    over every homogeneous arithmetic progression x_d, x_2d, x_3d, ...
    Returns the set of bitstrings (0 -> +1, 1 -> -1), in qubit order
    q0 q1 q2 q3 == x1 x2 x3 x4 (q0 is the least-significant / first qubit,
    printed as the leftmost character to match how the oracle below marks
    them)."""
    valid_bits = []
    for signs in itertools.product([1, -1], repeat=n):
        ok = True
        for d in range(1, n + 1):
            partial = 0
            for k in range(d, n + 1, d):
                partial += signs[k - 1]
                if abs(partial) > 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            bits = "".join("0" if s == 1 else "1" for s in signs)
            valid_bits.append(bits)
    return valid_bits


def build_oracle(marked_bitstrings, n):
    """Phase oracle flipping the sign of exactly the marked computational
    basis states (given as n-bit strings, index 0 = qubit 0)."""
    qc = QuantumCircuit(n, name="oracle")
    for bits in marked_bitstrings:
        # bits[i] corresponds to qubit i (bits[0] -> q0, ... bits[n-1] -> q(n-1))
        zero_qubits = [i for i in range(n) if bits[i] == "0"]
        for q in zero_qubits:
            qc.x(q)
        # multi-controlled Z across all n qubits
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def main():
    n = 4
    marked = classical_discrepancy_solutions(n)
    print(f"Classical brute force (n={n}): {len(marked)} valid sequences: {marked}")
    assert len(marked) == 2, "unexpected classical result, aborting"

    oracle = build_oracle(marked, n)
    diffuser = build_diffuser(n)

    import math
    N = 2 ** n
    M = len(marked)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n), range(n))

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit prints classical register bits with qubit (n-1) leftmost;
    # our marked strings use qubit 0 leftmost, so reverse before comparing.
    def to_our_order(qiskit_bitstring):
        return qiskit_bitstring[::-1]

    top_measured = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
    top_measured_ours = [to_our_order(b) for b, _ in top_measured]

    total_marked_shots = sum(c for b, c in counts.items() if to_our_order(b) in marked)
    marked_fraction = total_marked_shots / shots

    print(f"Grover iterations used: {iterations}")
    print(f"Top measured outcomes (our bit order, qubit0 leftmost): {top_measured_ours}")
    print(f"Fraction of shots landing on a classically-valid sequence: {marked_fraction:.3f}")

    verified = marked_fraction > 0.90 and set(top_measured_ours).issubset(set(marked))

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
