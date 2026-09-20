"""
Erdos problem #472 -- quantum-testable instance.

Source: /home/user/manman4/erdosproblems/data/problems.yaml, entry "number: 472".
OEIS id used: A389713 (listed in the problem's `oeis` field as
["A389713", "possible"] -- the "possible" tag in the source data marks this
association as tentative/best-guess, not confirmed; we use it as the best
available OEIS pointer for this problem).

A389713 definition (fetched from oeis.org/A389713 and independently verified
below by direct computation, not copied blindly): a(1) = 3, a(2) = 5, and for
n >= 3, a(n) is the SMALLEST PRIME p such that (p - a(n-1) + 1) is already a
member of the sequence {a(1), ..., a(n-1)}.

Classical property tested in this script:
    Given the known prefix a(1..8) = [3, 5, 7, 11, 13, 17, 19, 23], and a
    finite candidate list of the first 16 primes (2 qubits worth more than
    needed -> 4 qubits, 16 basis states), find the index of the SMALLEST
    prime p in that candidate list satisfying
        (p - a(8) + 1) in {a(1), ..., a(8)}.
    This is exactly the recurrence that defines a(9) in A389713, restricted
    to a small finite search space so it can be posed as a Grover search.

The classical answer (computed from first principles in `classical_answer()`
below, using a plain trial-division primality test, no external number
theory libraries) is p = 29, i.e. candidate index 9 in the 16-candidate list
(0-indexed: [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53]). This also equals
a(9) in the sequence's known terms, cross-checked here purely as a sanity
note -- the script does not "copy" that value, it derives it.

Quantum approach:
    Grover's algorithm on 4 qubits (16 basis states = the 16 candidate
    primes). The oracle marks every candidate index i for which
    (candidates[i] - a(8) + 1) is a member of the known prefix (there are
    two such indices in this instance: primes 29 and 41). Grover amplifies
    the marked subspace; we then verify that measurement lands in the marked
    set with high probability, and that the SMALLEST marked candidate
    (by list order, which is ascending in the prime's value) matches the
    classically computed a(9).

PASS/FAIL: the script prints PASS if (a) Grover's measurement distribution
places its highest-probability outcomes on the classically-marked index set,
and (b) the smallest marked index recovered this way reproduces the
classically computed next term of the sequence. Otherwise it prints FAIL.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Classical part: derive the sequence prefix, the candidate list, and the
# marked set, all from first principles (no OEIS values copied blindly).
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def build_sequence_prefix(num_terms: int) -> list:
    """Build a(1..num_terms) of A389713 from the recurrence, from scratch."""
    seq = [3, 5]
    while len(seq) < num_terms:
        prev = seq[-1]
        p = 2
        while True:
            if is_prime(p) and (p - prev + 1) in seq:
                seq.append(p)
                break
            p += 1
    return seq


def first_n_primes(n: int) -> list:
    out = []
    p = 2
    while len(out) < n:
        if is_prime(p):
            out.append(p)
        p += 1
    return out


NUM_QUBITS = 4
NUM_CANDIDATES = 2 ** NUM_QUBITS  # 16

seq_prefix = build_sequence_prefix(8)          # [3,5,7,11,13,17,19,23]
prev_term = seq_prefix[-1]                      # 23
candidates = first_n_primes(NUM_CANDIDATES)     # first 16 primes

marked_indices = [
    i for i, c in enumerate(candidates)
    if (c - prev_term + 1) in seq_prefix
]

if not marked_indices:
    raise RuntimeError("No candidate satisfies the recurrence in this search space; "
                        "the instance needs a larger candidate list.")

classical_smallest_marked_index = min(marked_indices)
classical_next_term = candidates[classical_smallest_marked_index]

# Independent classical cross-check: does the recurrence, run without any
# candidate-list cap, also produce this same next term?
full_check = build_sequence_prefix(9)
assert full_check[8] == classical_next_term, (
    f"Unbounded recurrence gives a(9)={full_check[8]} but bounded search "
    f"gives {classical_next_term}; instance is inconsistent."
)


# ---------------------------------------------------------------------------
# Quantum part: Grover search over the 16 candidate indices for the marked
# set defined above.
# ---------------------------------------------------------------------------

def multi_controlled_z(qc: QuantumCircuit, qubits):
    """Apply a Z rotation conditioned on all given qubits being |1>."""
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def apply_marking_oracle(qc: QuantumCircuit, qubits, index: int, n_qubits: int):
    """Flip the phase of basis state |index> (n_qubits-bit binary)."""
    bits = format(index, f"0{n_qubits}b")
    # X on qubits that should be 0, so the marked pattern becomes all-1s.
    for q, b in zip(qubits, reversed(bits)):
        if b == "0":
            qc.x(q)
    multi_controlled_z(qc, qubits)
    for q, b in zip(qubits, reversed(bits)):
        if b == "0":
            qc.x(q)


def build_oracle(n_qubits: int, marked: list) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Oracle")
    qubits = list(range(n_qubits))
    for idx in marked:
        apply_marking_oracle(qc, qubits, idx, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qubits = list(range(n_qubits))
    qc.h(qubits)
    qc.x(qubits)
    multi_controlled_z(qc, qubits)
    qc.x(qubits)
    qc.h(qubits)
    return qc


def build_grover_circuit(n_qubits: int, marked: list, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


N = NUM_CANDIDATES
M = len(marked_indices)
# Optimal number of Grover iterations for N states, M marked.
theta = np.arcsin(np.sqrt(M / N))
optimal_iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

circuit = build_grover_circuit(NUM_QUBITS, marked_indices, optimal_iterations)

simulator = AerSimulator()
shots = 4096
transpiled = transpile(circuit, simulator)
job = simulator.run(transpiled, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit's classical register bit order is little-endian in the returned
# bitstrings (qubit 0 is the rightmost character); our marking used the same
# convention (reversed(bits) mapping), so int(bitstring, 2) recovers the
# original candidate index directly.
measured_indices = {int(bitstring, 2): c for bitstring, c in counts.items()}

marked_hits = sum(c for idx, c in measured_indices.items() if idx in marked_indices)
marked_probability = marked_hits / shots

# Recover the "smallest marked index" quantum-side: among indices that were
# actually measured with non-negligible probability and are in the marked
# set, take the smallest -- this is what a search algorithm would report
# after amplifying the marked subspace and reading out.
significant_marked = [
    idx for idx, c in measured_indices.items()
    if idx in marked_indices and c >= shots * 0.05
]
quantum_smallest_marked_index = min(significant_marked) if significant_marked else None
quantum_next_term = (
    candidates[quantum_smallest_marked_index]
    if quantum_smallest_marked_index is not None else None
)

verified = (
    marked_probability > 0.8
    and quantum_next_term == classical_next_term
)

print("Sequence prefix (A389713, terms 1-8):", seq_prefix)
print("Candidate list (first 16 primes):", candidates)
print("Marked indices (candidates satisfying the recurrence):", marked_indices,
      "->", [candidates[i] for i in marked_indices])
print(f"Grover iterations used: {optimal_iterations}")
print("Measured counts:", counts)
print(f"Probability mass on marked set: {marked_probability:.4f}")
print("Classical next term a(9):", classical_next_term)
print("Quantum-recovered next term:", quantum_next_term)

print("PASS" if verified else "FAIL")
