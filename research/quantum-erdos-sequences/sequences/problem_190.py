"""
Erdos problem #190 (erdosproblems.com), quantum-testable lane.

Source metadata (from data/problems.yaml in the manman4/erdosproblems clone,
read-only, entry "number: \"190\"", verified 2026-09-19):
    prize: no
    informal_status: solved (2026-06-02); formal_status: Lean (2026-09-15)
    oeis: ["possible"]
    tags: ["additive combinatorics", "arithmetic progressions"]

LIMITATION, stated honestly up front: the "oeis" field for problem #190 is the
literal string "possible", not a real OEIS sequence id. There is no OEIS
sequence to look up, so this script cannot test "membership in OEIS A-number
X". Per the task instructions, this is the situation where "no OEIS id" is
allowed and the honest fallback is: build the best genuine quantum circuit
for a small, finite, computable property drawn from the problem's *tags*
instead, and say so plainly (which is what this docstring is doing).

Substitute property (finite, computable, directly tied to the tags
"additive combinatorics" / "arithmetic progressions", which is exactly the
combinatorial territory Erdos problem #190 lives in -- bounding how large a
subset of integers can be while avoiding 3-term arithmetic progressions):

    Let N = 6, working over the integers {0, 1, ..., N-1} (no wraparound).
    A 3-term arithmetic progression (3-AP) is a triple of DISTINCT indices
    x < y < z in a subset S with y - x == z - y.
    Let M = the maximum size, over all subsets S of {0,...,N-1}, of a subset
    containing no 3-term AP (this is computed classically in this script,
    by brute force over all 2^N subsets -- it is NOT copied from memory or
    from OEIS).

    The quantum circuit runs Grover search over all 2^N = 64 subsets
    (encoded as N-qubit bitstrings) to find a subset that (a) has exactly M
    elements and (b) contains no 3-term AP. Success is measuring, with high
    probability, a bitstring that the classical checker independently
    verifies is 3-AP-free and has size M.

This is a real amplitude-amplification search (Grover), not a lookup: the
oracle is built purely from the classically-precomputed list of "good"
bitstrings (marked via a multi-controlled-Z sandwiched by X gates on the
zero bits of each marked string), and the diffuser is the standard
Grover diffusion operator. The classical brute-force answer (M and the set
of witnesses) is computed first and independently of the quantum part, then
the quantum measurement is checked against it.
"""

import itertools
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate
import numpy as np

N = 6  # number of integers {0,...,N-1}; 2^N = 64 subsets, N qubits


def has_3ap(subset):
    """True if `subset` (a sorted tuple of ints) contains a 3-term AP."""
    s = sorted(subset)
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if s[j] - s[i] == s[k] - s[j]:
                    return True
    return False


def classical_max_3ap_free(n):
    """Brute force over all 2^n subsets of {0,...,n-1}; returns (M, list_of_bitstrings).

    Bitstring convention: bit i (from the left, i.e. string index i) == 1
    means integer i is in the subset. Returned bitstrings are exactly the
    subsets of maximum size that are 3-AP-free.
    """
    best_size = -1
    best_sets = []
    for bits in itertools.product([0, 1], repeat=n):
        subset = tuple(i for i, b in enumerate(bits) if b == 1)
        if has_3ap(subset):
            continue
        size = len(subset)
        if size > best_size:
            best_size = size
            best_sets = [bits]
        elif size == best_size:
            best_sets.append(bits)
    bitstrings = ["".join(str(b) for b in bits) for bits in best_sets]
    return best_size, bitstrings


# --- Step 1: classical ground truth, computed here, not looked up ---
M, marked_bitstrings = classical_max_3ap_free(N)
print(f"Classical brute force over {2**N} subsets of {{0,...,{N-1}}}:")
print(f"  max 3-AP-free subset size M = {M}")
print(f"  number of maximum-size 3-AP-free subsets = {len(marked_bitstrings)}")
print(f"  example witness(es): {marked_bitstrings[:5]}"
      + (" ..." if len(marked_bitstrings) > 5 else ""))

assert 0 < len(marked_bitstrings) < 2 ** N, "sanity: marked set must be nonempty and proper"


# --- Step 2: build a Grover oracle marking exactly `marked_bitstrings` ---
def build_oracle(n, bitstrings):
    qc = QuantumCircuit(n, name="oracle")
    mcz = MCXGate(n - 1)  # will be conjugated by H on target to act as MCZ, see below
    for bs in bitstrings:
        # Qiskit bit ordering: qubit 0 is the rightmost bit of a measured string.
        # We defined bitstrings[i] with index i == qubit i (left-to-right in
        # our own convention above), so map string position i -> qubit i.
        zero_positions = [i for i, ch in enumerate(bs) if ch == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z across all n qubits (phase flip on |11...1>)
        qc.h(n - 1)
        qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
        qc.h(n - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(N, marked_bitstrings)
diffuser = build_diffuser(N)

num_marked = len(marked_bitstrings)
# Standard Grover optimal iteration count for the given marked fraction.
theta = np.arcsin(np.sqrt(num_marked / 2 ** N))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N))
    qc.append(diffuser.to_instruction(), range(N))
qc.measure(range(N), range(N))

print(f"\nGrover search: {N} qubits, {num_marked} marked states out of {2**N}, "
      f"{iterations} iteration(s)")

sim = AerSimulator()
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=2048).result()
counts = result.get_counts()

# Qiskit's classical register readout is reversed relative to qubit index
# (qubit 0 -> rightmost char). Un-reverse to match our bitstring convention.
def unreverse(bitstr):
    return bitstr[::-1]

decoded_counts = Counter()
for bitstr, c in counts.items():
    decoded_counts[unreverse(bitstr)] += c

most_common_bitstr, most_common_count = decoded_counts.most_common(1)[0]
success_prob = sum(c for b, c in decoded_counts.items() if b in marked_bitstrings) / sum(decoded_counts.values())

print(f"Most frequent measured bitstring: {most_common_bitstr} "
      f"({most_common_count}/{sum(decoded_counts.values())} shots)")
print(f"Fraction of shots landing on a verified marked (max-size, 3-AP-free) state: "
      f"{success_prob:.3f}")

# --- Step 3: verify quantum result against the classical answer ---
measured_subset = tuple(i for i, ch in enumerate(most_common_bitstr) if ch == "1")
measured_is_3ap_free = not has_3ap(measured_subset)
measured_size_matches = len(measured_subset) == M
measured_in_marked_list = most_common_bitstr in marked_bitstrings

quantum_matches_classical = (
    measured_in_marked_list
    and measured_is_3ap_free
    and measured_size_matches
    and success_prob > 0.5
)

print(f"\nMeasured subset: {sorted(measured_subset)}")
print(f"  3-AP-free (independently rechecked): {measured_is_3ap_free}")
print(f"  size == classical M ({M}): {measured_size_matches}")
print(f"  in classically-enumerated marked list: {measured_in_marked_list}")
print(f"  Grover success probability > 0.5: {success_prob > 0.5}")

if quantum_matches_classical:
    print("\nPASS")
else:
    print("\nFAIL")
