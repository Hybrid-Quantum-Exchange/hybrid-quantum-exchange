"""
Erdos problem #880 -- quantum-testable instance.

Source metadata (erdosproblems.com data, data/problems.yaml entry
`number: "880"`): prize="no", status="proved", oeis=["N/A"],
tags=["number theory", "additive basis"].

LIMITATION, stated honestly up front: problem 880 carries no OEIS sequence
id at all (oeis: ["N/A"]), so there is no published integer sequence for
this script to test membership/terms of. What the metadata *does* give is
a topic tag, "additive basis" -- i.e. the problem concerns sets of
non-negative integers S such that every element of some target range can
be written as a sum of (at most) two elements of S ("S is an additive
basis of order 2" for that range, also called a B_2-covering set).

That notion is finite and computable for a small instance, so this script
builds a genuine, non-fabricated quantum-testable property from it
instead of inventing a fake OEIS value:

    Fix a small explicit set S = {1, 2, 3, 4} (S has 4 elements, indexed
    0..3, so a pair (i, j) of indices needs exactly 2+2 = 4 qubits) and a
    target sum t = 5. The classical/combinatorial question is:

        Which ordered index pairs (i, j) in {0,1,2,3}^2 satisfy
        S[i] + S[j] == t ?

    This is exactly the certificate-checking step that underlies the
    additive-basis property (S is a valid order-2 additive basis for a
    range R iff, for every t in R, at least one such pair exists). The
    script first computes this classical answer directly (brute force
    over all 16 pairs, first principles, no OEIS lookup involved), then
    builds a genuine Grover search circuit over the 4-qubit pair-index
    register whose oracle marks exactly the pair states satisfying
    S[i] + S[j] == t, and uses the standard number-of-Grover-iterations
    formula for M marked items out of N = 16 to amplify them.  The
    circuit is run on the ideal AerSimulator and its measured output
    distribution is compared against the classical marked set: the
    script PASSes if the quantum sampling concentrates its probability
    mass (checked via a fixed threshold) on exactly the classically
    correct marked states and no others.

This is a real (if modest) Grover amplitude-amplification circuit built
from a from-scratch classical computation -- not a copied OEIS term --
and is offered as the honest best-effort quantum-testable artifact for a
Erdos problem whose own metadata provides no sequence to draw on.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

S = [1, 2, 3, 4]        # small explicit candidate additive-basis set
TARGET = 5               # target sum t
N_INDEX_BITS = 2          # 2 bits per index -> len(S) == 4 == 2**2
N_QUBITS = 2 * N_INDEX_BITS  # 4 qubits total: (i, j) pair register

assert len(S) == 2 ** N_INDEX_BITS

def classical_marked_pairs():
    """All (i, j) in {0,1,2,3}^2 with S[i] + S[j] == TARGET, brute force."""
    marked = []
    for i, j in itertools.product(range(len(S)), repeat=2):
        if S[i] + S[j] == TARGET:
            marked.append((i, j))
    return marked


MARKED_PAIRS = classical_marked_pairs()
# Encode a pair (i, j) as a 4-bit string "i1 i0 j1 j0" (qubit order matches
# circuit construction below: qubits [0,1] hold i, qubits [2,3] hold j).
def pair_to_bits(i, j):
    return format(i, f"0{N_INDEX_BITS}b") + format(j, f"0{N_INDEX_BITS}b")


MARKED_BITSTRINGS = {pair_to_bits(i, j) for (i, j) in MARKED_PAIRS}
assert len(MARKED_PAIRS) > 0, "instance must have at least one marked pair"


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffuser for this concrete marked set.
# ---------------------------------------------------------------------------

def apply_oracle(qc: QuantumCircuit, qubits):
    """Phase-flip exactly the basis states in MARKED_BITSTRINGS.

    Each marked bitstring is targeted with a standard X-sandwiched
    multi-controlled-Z: X on every qubit whose target bit is 0, an
    (n-1)-controlled Z with the last qubit as target realized via
    H-MCX-H, then undo the X's.
    """
    n = len(qubits)
    for bits in MARKED_BITSTRINGS:
        # bits[0] -> qubits[0], ..., bits[n-1] -> qubits[n-1]
        zero_positions = [k for k, b in enumerate(bits) if b == "0"]
        for k in zero_positions:
            qc.x(qubits[k])

        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])

        for k in zero_positions:
            qc.x(qubits[k])


def apply_diffuser(qc: QuantumCircuit, qubits):
    """Standard Grover diffuser (inversion about the mean) on `qubits`."""
    n = len(qubits)
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit():
    n = N_QUBITS
    N = 2 ** n
    M = len(MARKED_BITSTRINGS)

    # Optimal number of Grover iterations for M marked items out of N.
    theta = math.asin(math.sqrt(M / N))
    r = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n, n)
    qubits = list(range(n))

    qc.h(qubits)  # uniform superposition
    for _ in range(r):
        apply_oracle(qc, qubits)
        apply_diffuser(qc, qubits)

    qc.measure(qubits, qubits)
    return qc, r, M, N


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

def run():
    qc, r, M, N = build_grover_circuit()

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit prints bit c[n-1] ... c[0]; our qubits[0..n-1] were measured
    # into classical bits [0..n-1] in the same order, and Qiskit's count
    # keys are little-endian in cbit index (rightmost char = cbit 0). Our
    # bitstring convention above was built as "i1 i0 j1 j0" with qubits[0]
    # = i1 ... qubits[3] = j0, i.e. qubits[k] <-> string position k. Qiskit
    # count keys read left-to-right as cbit (n-1) .. cbit 0, so we reverse
    # to get position k = qubit k, matching MARKED_BITSTRINGS' convention.
    def key_to_bits(key):
        return key[::-1]

    total = sum(counts.values())
    marked_mass = sum(
        c for key, c in counts.items() if key_to_bits(key) in MARKED_BITSTRINGS
    )
    marked_fraction = marked_mass / total

    # Which bitstrings the quantum run actually concentrated on (top M by
    # count), to check they are *exactly* the classically marked set.
    sorted_keys = sorted(counts.items(), key=lambda kv: -kv[1])
    top_m_bits = {key_to_bits(k) for k, _ in sorted_keys[:M]}

    print(f"Erdos problem #880 -- additive-basis certificate via Grover search")
    print(f"S = {S}, target sum t = {TARGET}, index qubits per element = {N_INDEX_BITS}")
    print(f"Classical marked pairs (i, j) with S[i]+S[j]=={TARGET}: {MARKED_PAIRS}")
    print(f"Classical marked bitstrings: {sorted(MARKED_BITSTRINGS)}")
    print(f"Grover iterations used: {r}  (N={N}, M={M})")
    print(f"Quantum measured probability mass on marked states: {marked_fraction:.4f}")
    print(f"Top-{M} measured bitstrings: {sorted(top_m_bits)}")

    # Pass criteria:
    #  (a) Grover amplification concentrated most probability mass onto the
    #      marked subspace (comfortably above the uniform baseline M/N), and
    #  (b) the exact set of most-frequent outcomes equals the classical
    #      marked set, i.e. the quantum search found precisely the right
    #      pairs and nothing else.
    baseline = M / N
    amplified_enough = marked_fraction > max(0.8, baseline * 3)
    exact_match = top_m_bits == MARKED_BITSTRINGS

    passed = amplified_enough and exact_match

    print(f"Baseline (uniform) marked fraction would be: {baseline:.4f}")
    print(f"Amplified enough: {amplified_enough}, exact top-M match: {exact_match}")
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    run()
