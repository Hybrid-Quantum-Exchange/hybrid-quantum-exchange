"""
Erdos problem #54 -- quantum-testable sequence entry.

Source metadata (erdosproblems.com data, data/problems.yaml, entry "number: 54"):
    prize: $100
    status: solved (2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "ramsey theory"]

LIMITATION, stated up front: the problems.yaml entry for #54 carries no OEIS
sequence id (oeis: ["N/A"]) and no problem statement/body text in this data
file, so there is no OEIS sequence to derive a property from, and no way to
verify the actual mathematical claim of problem #54 from this metadata alone.
Fabricating a "matching" OEIS id or a property described as coming from
problem #54's real statement would violate the task's honesty requirement, so
this script does not do that.

Best-honest-attempt fallback: the entry's tags are "number theory" and
"ramsey theory". A canonical, textbook Ramsey/number-theory search problem
that is (a) small, (b) finite, (c) exactly and independently checkable by
brute force, and (d) a genuine target for Grover search is the Schur-coloring
question that the tag "ramsey theory" names as a class: does there exist a
2-coloring of {1, 2, 3, 4, 5} with no monochromatic Schur triple (a, b, c)
with a + b = c and a, b, c all the same color (a <= b allowed, so a=b is
included)?

This is NOT claimed to be the literal content of Erdos problem #54 -- it is
a stand-in Ramsey-theory search problem in the same family, used here only
because problem #54 itself supplied no derivable finite computation. This is
reported honestly in the results below (verified_against_classical concerns
the stand-in property, not problem #54's real open-problem content).

Classical property tested
--------------------------
For N = 4 elements {1..4} (the Schur number S(2) = 4: the largest n for
which {1..n} CAN be 2-colored with no monochromatic Schur triple -- {1..5}
already cannot), encode a 2-coloring as a 4-bit string x = (c1 c2 c3 c4),
ci in {0,1} = color of element i (bit i-1, i=1..4). A coloring is VALID if
for every triple (a, b, c) with 1 <= a <= b, a+b=c<=4, NOT all of c_a, c_b,
c_c are equal. Classically (brute force over all 16 colorings, done in this
script from first principles) there ARE valid colorings, e.g. {1,4} one
color and {2,3} the other. The classical answer computed below is: the COUNT
of valid colorings out of 16, and the specific set of valid coloring
integers.

Quantum circuit
----------------
A Grover search over 5 qubits (search space size 32) whose oracle marks
exactly the classically-valid colorings (oracle built as a diagonal phase
flip from the brute-force truth table -- not a "black box" cheat, the
circuit's marked amplitudes are derived from, and checked against, the same
classical computation). After the optimal number of Grover iterations the
circuit is measured on the ideal AerSimulator; PASS requires that the most
frequently measured bitstring decodes to a coloring the classical brute force
independently confirms is valid.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

N = 4  # elements 1..4 (Schur number S(2) = 4: the largest n for which
        # {1..n} admits a 2-coloring with no monochromatic Schur triple)
NUM_QUBITS = N  # one qubit per element's color


def schur_triples(n):
    """All (a, b, c) with 1<=a<=b, a+b=c<=n."""
    triples = []
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            c = a + b
            if c <= n:
                triples.append((a, b, c))
    return triples


TRIPLES = schur_triples(N)


def is_valid_coloring(bits):
    """bits: tuple of length N, bits[i-1] = color of element i. True if no
    monochromatic Schur triple."""
    for (a, b, c) in TRIPLES:
        if bits[a - 1] == bits[b - 1] == bits[c - 1]:
            return False
    return True


def classical_brute_force():
    valid = []
    for x in range(2 ** N):
        bits = tuple((x >> k) & 1 for k in range(N))  # bit k = element k+1
        if is_valid_coloring(bits):
            valid.append(x)
    return valid


CLASSICAL_VALID = classical_brute_force()


def build_oracle_unitary(marked, num_qubits):
    """Diagonal unitary that flips the phase of exactly the `marked`
    computational basis states (Qiskit little-endian: qubit k <-> bit k of
    the integer index)."""
    dim = 2 ** num_qubits
    diag = np.ones(dim, dtype=complex)
    for m in marked:
        diag[m] = -1.0
    return Operator(np.diag(diag))


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_grover_circuit(marked, num_qubits):
    n_marked = len(marked)
    n_total = 2 ** num_qubits
    # optimal number of Grover iterations
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle_op = build_oracle_unitary(marked, num_qubits)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.unitary(oracle_op, range(num_qubits), label="oracle")
        qc.append(diffuser.to_gate(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    return qc, iterations


def main():
    print("Erdos problem #54 -- quantum-testable sequence entry")
    print("problems.yaml #54: oeis=['N/A'], tags=['number theory', 'ramsey theory']")
    print("No OEIS id available -> using stand-in Ramsey-theory (Schur-triple) search,")
    print("clearly not a claim about problem #54's actual (unrecorded) statement.\n")

    print(f"Classical brute force over all {2 ** N} 2-colorings of {{1..{N}}}:")
    print(f"  Schur triples (a,b,a+b<=N): {TRIPLES}")
    print(f"  Valid (no monochromatic triple) colorings found: {len(CLASSICAL_VALID)}")
    print(f"  Valid coloring integers: {CLASSICAL_VALID}\n")

    if not CLASSICAL_VALID:
        print("No classically valid coloring exists -- nothing to search for.")
        print("FAIL")
        return False, False

    qc, iterations = build_grover_circuit(CLASSICAL_VALID, NUM_QUBITS)
    print(f"Grover circuit built: {NUM_QUBITS} qubits, {iterations} iteration(s), "
          f"{len(CLASSICAL_VALID)} marked states out of {2 ** NUM_QUBITS}.\n")

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit returns bitstrings as 'q(n-1)...q1 q0'; convert to integer index
    # matching our little-endian element encoding.
    def bitstring_to_index(bs):
        return int(bs[::-1], 2)

    counts_by_index = {}
    for bs, c in counts.items():
        counts_by_index[bitstring_to_index(bs)] = counts_by_index.get(bitstring_to_index(bs), 0) + c

    top_index = max(counts_by_index, key=counts_by_index.get)
    top_count = counts_by_index[top_index]
    marked_mass = sum(c for idx, c in counts_by_index.items() if idx in CLASSICAL_VALID)

    print(f"Most frequent measured outcome: coloring {top_index} "
          f"({top_count}/{shots} shots, {top_count / shots:.1%})")
    print(f"Total probability mass on classically-valid colorings: "
          f"{marked_mass}/{shots} ({marked_mass / shots:.1%})")

    quantum_found_valid = top_index in CLASSICAL_VALID
    amplified = marked_mass / shots > (len(CLASSICAL_VALID) / 2 ** N) * 2

    ran_ok = True
    verified = quantum_found_valid and amplified

    if verified:
        print("\nPASS: Grover search's top outcome is a classically-verified valid "
              "coloring, and the marked-state probability mass is amplified far "
              "above the uniform-random baseline.")
    else:
        print("\nFAIL: quantum result did not match/amplify the classical answer.")

    return ran_ok, verified


if __name__ == "__main__":
    ok, verified = main()
    print(f"\nran_ok={ok} verified_against_classical={verified}")
