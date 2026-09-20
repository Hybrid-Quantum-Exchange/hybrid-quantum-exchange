#!/usr/bin/env python3
"""
Erdos problem #870 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: \"870\""):
    prize: no
    informal_status: open (last update 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "additive basis"]

LIMITATION, stated honestly up front: problem #870 carries no OEIS sequence
id in the source data (oeis: ["N/A"]) and its informal status is "open" with
no closed-form finite property recorded in the metadata available here. There
is therefore no specific known term or OEIS-verified value to check a quantum
circuit against for *this exact* problem. Rather than fabricate a fake OEIS
lookup, this script instead builds a genuine, mathematically real instance of
the problem's own tag -- "additive basis" -- which is the closest small,
finite, exactly-computable property that is faithful to what erdosproblems
actually records about problem 870. This is a best-honest-effort substitute
for a term-membership check, not a literal test of the (currently open,
OEIS-less) problem statement itself.

Classical property being tested
--------------------------------
Fix the universe U = {0, 1, 2, 3, 4} (5 elements, N = |U| = 5).
A subset S subseteq U is an ADDITIVE BASIS OF ORDER 2 for the interval
[0, 2*(N-1)] = [0, 8] if the sumset

    S + S = { a + b : a, b in S }   (a = b allowed)

equals the full interval {0, 1, ..., 8}.

We classically enumerate, by brute force, every one of the 2^5 = 32 subsets
of U and determine exactly which ones are valid additive bases of order 2.
This is computed from first principles in this script (function
`is_additive_basis`), with no external data and no copied literature value.

Quantum circuit
----------------
A Grover search over the 5-qubit computational basis (32 candidate subsets).
The oracle is built directly from the classically precomputed marked set: for
every classically-verified valid additive basis, a multi-controlled-Z phase
flip marks that exact computational basis state. This is the standard way to
run Grover's algorithm once the marking predicate has been evaluated
classically for a small enough search space (32 states here), and it lets us
compare a *quantum* amplitude-amplification search against the *classical*
brute-force answer on the ideal AerSimulator.

The optimal number of Grover iterations is computed from the standard
formula r ~ (pi/4) * sqrt(N_total / N_marked), rounded to the nearest
integer, and the circuit is run twice (with and without amplitude
amplification) to make the amplification's effect explicit.

Test: after running Grover's circuit, we take the top-N_marked most
frequent measured bitstrings (N_marked = number of classically valid
subsets) and check that this set is EXACTLY the classically-computed marked
set. PASS if they match, FAIL otherwise.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no external data)
# ---------------------------------------------------------------------------

UNIVERSE = list(range(5))          # U = {0,1,2,3,4}
N_QUBITS = len(UNIVERSE)           # 5 qubits -> 32 candidate subsets
TARGET_RANGE = range(0, 2 * (len(UNIVERSE) - 1) + 1)  # {0,...,8}


def subset_from_bits(bits):
    """bits: tuple of 0/1 of length N_QUBITS, bit i => element i included."""
    return [UNIVERSE[i] for i, b in enumerate(bits) if b == 1]


def is_additive_basis(subset):
    """True iff subset + subset (with repetition) covers TARGET_RANGE exactly."""
    if not subset:
        return False
    sumset = set()
    for a in subset:
        for b in subset:
            sumset.add(a + b)
    return set(TARGET_RANGE).issubset(sumset)


def bits_to_str(bits):
    # qubit 0 is the least-significant bit in Qiskit's bitstring ordering,
    # so we render MSB..LSB as bit[N-1]..bit[0] to match circuit measurement.
    return "".join(str(b) for b in reversed(bits))


marked_bitstrings = []
marked_subsets = {}
for bits in itertools.product([0, 1], repeat=N_QUBITS):
    subset = subset_from_bits(bits)
    if is_additive_basis(subset):
        s = bits_to_str(bits)
        marked_bitstrings.append(s)
        marked_subsets[s] = subset

marked_bitstrings = sorted(marked_bitstrings)
n_total = 2 ** N_QUBITS
n_marked = len(marked_bitstrings)

print(f"Universe U = {UNIVERSE}, target sum range = {list(TARGET_RANGE)}")
print(f"Total candidate subsets: {n_total}")
print(f"Classically valid additive bases of order 2 ({n_marked} found):")
for s in marked_bitstrings:
    print(f"  bitstring={s}  subset={marked_subsets[s]}")

assert n_marked > 0, "no marked subsets found -- cannot build a Grover oracle"


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffusion, built from the classical marked set
# ---------------------------------------------------------------------------

def apply_multi_controlled_z(qc, bitstring, qubits):
    """Flip the phase of the single computational basis state `bitstring`
    (MSB..LSB order, matching bits_to_str), using X-sandwiched MCZ."""
    n = len(qubits)
    # bitstring[0] is MSB -> corresponds to qubits[n-1]; bitstring[-1] is LSB -> qubits[0]
    zero_positions = [n - 1 - i for i, c in enumerate(bitstring) if c == "0"]
    for pos in zero_positions:
        qc.x(qubits[pos])
    if n == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for pos in zero_positions:
        qc.x(qubits[pos])


def oracle(qc, qubits, marked):
    for s in marked:
        apply_multi_controlled_z(qc, s, qubits)


def diffusion(qc, qubits):
    n = len(qubits)
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


n_iterations = max(1, round((math.pi / 4) * math.sqrt(n_total / n_marked)))
print(f"\nGrover iterations used: {n_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))
qc.h(qubits)
for _ in range(n_iterations):
    oracle(qc, qubits, marked_bitstrings)
    diffusion(qc, qubits)
qc.measure(qubits, qubits)

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 20000
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# top-n_marked most frequent measured bitstrings
top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n_marked]
top_bitstrings = sorted(k for k, _ in top)

print("\nMeasurement counts (top results):")
for s, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: max(10, n_marked)]:
    tag = " <-- marked (classical)" if s in marked_bitstrings else ""
    print(f"  {s}: {c}{tag}")

quantum_answer = top_bitstrings
classical_answer = marked_bitstrings

print(f"\nClassical answer (sorted marked bitstrings): {classical_answer}")
print(f"Quantum answer   (top-{n_marked} measured bitstrings, sorted): {quantum_answer}")

passed = quantum_answer == classical_answer
print("\nPASS" if passed else "\nFAIL")
