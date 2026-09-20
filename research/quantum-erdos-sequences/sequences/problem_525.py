"""
Erdos problem #525 (source: manman4/erdosproblems data/problems.yaml, entry
"number: \"525\"", tags: analysis / probability / polynomials).

LIMITATION, stated up front: problem #525's YAML entry does not carry a real
OEIS id. Its `oeis` field is the literal string "possible" (not an A-number),
which is not a usable sequence identifier, and the problem's substance
(a probability/analysis statement about random polynomials) is not itself a
finite decidable property that a small quantum circuit can search or verify.
So this script cannot build a circuit whose correctness is tied to problem
#525's actual mathematical content -- there is no OEIS-derived small property
to test.

Rather than fabricate a fake connection, this script honestly substitutes the
nearest real, finite, computable task suggested by the problem's own tags
(polynomials + a search over small integers): find an integer root of a
concrete quadratic polynomial modulo N by exhaustive/Grover search. This is a
textbook instance of Grover's algorithm (unstructured search with a quantum
speedup over an oracle), built and run for real on Qiskit's AerSimulator, and
its classical correctness is derived from first principles in this script
(brute force over all N candidates), not copied from any table.

Concretely:
  - N = 8  (3 qubits, indices x = 0..7)
  - polynomial: f(x) = x^2 - 4  (mod 8)
  - property under test: which x in {0,...,7} satisfy f(x) == 0 (mod 8),
    i.e. x^2 == 4 (mod 8). Classically this is computed by brute force below.
  - the oracle marks exactly those x; Grover's algorithm is run to amplify
    them, then the top measurement outcomes are compared against the
    classically computed root set.

Honesty note for downstream aggregation: verified_against_classical here
means "the Grover circuit correctly finds the roots of the chosen small
polynomial, checked against brute force" -- it does NOT mean this verifies
any specific term of an OEIS sequence tied to Erdos problem #525, because no
such usable OEIS id exists in the source data for this problem.
"""

import sys
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np

ERDOS_PROBLEM = 525
OEIS_IDS_FOUND = []  # none usable; source field was the literal string "possible"

N_QUBITS = 3
N = 2 ** N_QUBITS  # 8


def f(x: int) -> int:
    """f(x) = x^2 - 4 (mod 8)."""
    return (x * x - 4) % N


def classical_roots():
    """Brute-force, first-principles computation of {x : f(x) == 0 mod N}."""
    return sorted(x for x in range(N) if f(x) == 0)


def build_oracle(marked):
    """Phase-flip oracle marking each x in `marked` (a list of ints < N)."""
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    for x in marked:
        bits = format(x, f"0{N_QUBITS}b")
        # flip qubits where bit is 0, so the target becomes |111>, then
        # apply a multi-controlled Z via H-MCX-H, then flip back
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(i)
        qc.h(N_QUBITS - 1)
        qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
        qc.h(N_QUBITS - 1)
        for i, b in enumerate(reversed(bits)):
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


def run_grover(marked, shots=2000):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))

    oracle = build_oracle(marked)
    diffuser = build_diffuser()

    # optimal number of Grover iterations for M marked items out of N
    M = len(marked)
    if M == 0 or M == N:
        iterations = 0
    else:
        theta = np.arcsin(np.sqrt(M / N))
        iterations = max(1, round((np.pi / 4) / theta - 0.5))

    for _ in range(iterations):
        qc.compose(oracle, range(N_QUBITS), inplace=True)
        qc.compose(diffuser, range(N_QUBITS), inplace=True)

    qc.measure(range(N_QUBITS), range(N_QUBITS))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    print(f"Erdos problem #{ERDOS_PROBLEM} -- quantum-testable lane")
    print(f"OEIS ids found in source data: {OEIS_IDS_FOUND!r} (none usable)")
    print("Substituted task: Grover search for integer roots of")
    print("f(x) = x^2 - 4 (mod 8) over x in {0,...,7}")

    roots = classical_roots()
    print(f"Classical (brute-force) roots: {roots}")
    assert roots == [2, 6], f"unexpected classical roots: {roots}"

    counts, iterations = run_grover(roots)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    total_shots = sum(counts.values())
    marked_bitstrings = {format(x, f"0{N_QUBITS}b") for x in roots}
    hits = sum(c for bs, c in counts.items() if bs in marked_bitstrings)
    hit_fraction = hits / total_shots

    print(f"Fraction of shots landing on a true root: {hit_fraction:.3f}")

    # A correctly amplified Grover search on N=8 with 2 marked items should
    # concentrate the large majority of shots on the marked states (ideal
    # simulator, no noise), well above the 2/8 = 0.25 baseline of random
    # guessing.
    passed = hit_fraction > 0.8

    top_outcome = max(counts, key=counts.get)
    top_x = int(top_outcome, 2)
    top_in_roots = top_x in roots
    passed = passed and top_in_roots

    print(f"Top measured outcome: {top_outcome} -> x={top_x}, is a root: {top_in_roots}")

    if passed:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
