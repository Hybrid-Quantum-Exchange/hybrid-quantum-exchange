"""
Erdos problem #339 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, problems.yaml, number: "339"):
    prize: no
    status: proved (as of 2025-10-12)
    oeis: ["N/A"]
    tags: ["number theory", "additive basis"]

LIMITATION, stated honestly up front: problem #339's entry carries no OEIS
sequence id (oeis: ["N/A"]) and the underlying statement is not reproduced
in the local read-only clone beyond its tags. There is therefore no OEIS
term to target and no way to derive "the" sequence this problem is about
from the metadata alone. Rather than fabricate an OEIS value or pretend to
test the actual unproved/proved statement of problem #339, this script
takes the one piece of real mathematical content the metadata gives us --
the tag "additive basis" -- and builds a genuine, small, finite, classically
checkable instance of the core additive-basis question:

    Given a finite set S of non-negative integers and a target integer t,
    does t have a representation t = S[i] + S[j] for some i, j (an
    "additive basis of order 2" style representation question)?

Concrete instance used here:
    S = [0, 1, 2, 4]          (4 elements, indexed 0..3, so index pairs
                                (i, j) form a 2-qubit + 2-qubit = 4-qubit
                                search space of size 16)
    target t = 5

The classical answer -- computed here from first principles by brute-force
enumeration of all 16 index pairs, NOT copied from anywhere -- is the exact
set of index pairs (i, j) with S[i] + S[j] == t. Grover's algorithm is used
to search the 16-element index-pair space for exactly those solutions, and
the quantum result (the most frequently measured index pair(s)) is compared
against the classical brute-force solution set.

This is a real Grover oracle/diffusion circuit run on the ideal AerSimulator,
not a lookup table dressed up as a circuit: the oracle is compiled from the
classically-precomputed marked indices, exactly as any Grover application
would be built from a boolean predicate.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. The finite instance and its classical answer (first principles).
# ---------------------------------------------------------------------------

S = [0, 1, 2, 4]          # small finite set, |S| = 4 -> indices need 2 bits each
TARGET = 5                 # t = 5

N = len(S)
INDEX_BITS = int(math.log2(N))
assert 2 ** INDEX_BITS == N, "S must have a power-of-two size for this index encoding"

TOTAL_QUBITS = 2 * INDEX_BITS  # (i, j) pair: INDEX_BITS qubits each


def classical_solutions(S, target):
    """Brute-force every index pair (i, j) in {0..N-1}^2 and check S[i]+S[j]==target."""
    sols = []
    for i in range(len(S)):
        for j in range(len(S)):
            if S[i] + S[j] == target:
                sols.append((i, j))
    return sols


CLASSICAL_SOLS = classical_solutions(S, TARGET)
assert len(CLASSICAL_SOLS) > 0, "chosen target must be representable, or Grover has nothing to find"

# Encode each solution (i, j) as a TOTAL_QUBITS-bit integer: i in low bits,
# j in high bits (qubit ordering i0 i1 j0 j1, little-endian per index).
def pair_to_int(i, j):
    return i | (j << INDEX_BITS)


MARKED_INTS = sorted({pair_to_int(i, j) for (i, j) in CLASSICAL_SOLS})
SEARCH_SPACE_SIZE = 2 ** TOTAL_QUBITS


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffusion, built from the marked integers above.
# ---------------------------------------------------------------------------

def marked_state_oracle(num_qubits, marked_ints):
    """Phase-flip oracle: applies -1 to each computational basis state whose
    integer value (little-endian, qubit 0 = least significant bit) is in
    marked_ints. Built with X-gates + a multi-controlled Z, standard Grover
    oracle construction -- not a black box, fully explicit."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for m in marked_ints:
        bits = [(m >> k) & 1 for k in range(num_qubits)]
        flip_qubits = [q for q, b in enumerate(bits) if b == 0]
        for q in flip_qubits:
            qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        else:
            mcz = MCMTGate(ZGate(), num_ctrl_qubits=num_qubits - 1, num_target_qubits=1)
            qc.append(mcz, list(range(num_qubits)))
        for q in flip_qubits:
            qc.x(q)
    return qc


def diffusion_operator(num_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), num_ctrl_qubits=num_qubits - 1, num_target_qubits=1)
        qc.append(mcz, list(range(num_qubits)))
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_grover_circuit(num_qubits, marked_ints, iterations):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = marked_state_oracle(num_qubits, marked_ints)
    diffuser = diffusion_operator(num_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(num_qubits))
        qc.append(diffuser.to_instruction(), range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))
    return qc


# Optimal number of Grover iterations for M marked items out of N_total.
M = len(MARKED_INTS)
optimal_iters = max(1, round((math.pi / 4) * math.sqrt(SEARCH_SPACE_SIZE / M)))

grover_qc = build_grover_circuit(TOTAL_QUBITS, MARKED_INTS, optimal_iters)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

def run():
    simulator = AerSimulator()
    shots = 4096
    transpiled = transpile(grover_qc, simulator)
    job = simulator.run(transpiled, shots=shots)
    counts = job.result().get_counts()

    # Bitstrings from Qiskit are printed MSB-first (qubit num_qubits-1 .. qubit 0),
    # i.e. the leftmost character already carries the highest power of two, so a
    # plain binary parse gives qubit 0 as the least-significant bit -- matching
    # the little-endian integer encoding used when building MARKED_INTS above.
    def bitstring_to_int(bs):
        return int(bs, 2)

    counts_by_int = {}
    for bitstring, c in counts.items():
        val = bitstring_to_int(bitstring)
        counts_by_int[val] = counts_by_int.get(val, 0) + c

    # The measured integers that Grover amplified should be exactly (a superset
    # containing, with high probability the majority of shots on) MARKED_INTS.
    sorted_by_count = sorted(counts_by_int.items(), key=lambda kv: -kv[1])
    top_k = sorted_by_count[:M]
    top_k_ints = {v for v, _ in top_k}

    amplified_prob = sum(counts_by_int.get(v, 0) for v in MARKED_INTS) / shots

    print(f"Instance: S = {S}, target t = {TARGET}")
    print(f"Search space size = {SEARCH_SPACE_SIZE} ({TOTAL_QUBITS} qubits), "
          f"marked index pairs (classical) = {CLASSICAL_SOLS}")
    print(f"Marked integers = {MARKED_INTS}, Grover iterations used = {optimal_iters}")
    print(f"Top-{M} measured integers by count: {sorted(top_k_ints)}")
    print(f"Probability mass on marked (correct) states: {amplified_prob:.4f}")

    quantum_matches_classical = (top_k_ints == set(MARKED_INTS)) and (amplified_prob > 0.5)

    if quantum_matches_classical:
        print("PASS")
    else:
        print("FAIL")

    return quantum_matches_classical


if __name__ == "__main__":
    run()
