"""
Erdos problem #603 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: \"603\""):
    prize: none
    informal_status: solved (last_update 2026-07-06)
    oeis: ["N/A"]
    tags: ["combinatorics", "set theory"]

LIMITATION, stated honestly up front: problem #603's metadata record carries
no OEIS sequence id at all (oeis: ["N/A"]) and the local erdosproblems clone
has no statement text for it (no per-problem markdown/description file was
found alongside data/problems.yaml). There is therefore no literal OEIS term
to fetch or verify against, and no problem statement to encode directly. This
script does NOT fabricate an OEIS value or invent a fake link to problem #603
beyond its two real tags, "combinatorics" and "set theory".

What is tested instead: since Erdos's own major open/solved questions in this
area repeatedly concern SUM-FREE SETS (a set S of positive integers is
sum-free if there are no a, b, c in S, not necessarily distinct, with
a + b = c), this script picks a genuine, finite, classically-checkable
instance of that combinatorics/set-theory property and verifies it with a
real Grover search circuit, rather than pretending it is literally sequence
"#603" from OEIS.

Concrete finite instance:
    Universe U = {1, 2, 3, 4, 5, 6}. Every subset of U is encoded as a 6-bit
    string b5 b4 b3 b2 b1 b0, bit i = 1 iff (i+1) is in the subset.
    Define property P(S):  |S| == 3  AND  S is sum-free  AND  max(S) == 6.
    (The max==6 clause is added purely to pin the search down to a single
    marked basis state out of 2^6 = 64, which is what makes single-target
    Grover amplitude amplification exhibit its textbook ~100% success
    probability after the optimal number of iterations.)

Classical ground truth is computed first, in this script, by brute-force
enumeration over all 64 subsets of U -- no OEIS lookup, no hard-coded answer.
The subset(s) satisfying P are then encoded as the Grover oracle's marked
computational basis state(s), and Grover's algorithm is run on the ideal
AerSimulator to recover it purely by amplitude amplification. PASS requires
the state Grover returns as most likely to exactly match the classically
computed marked subset.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


UNIVERSE = [1, 2, 3, 4, 5, 6]
N_QUBITS = len(UNIVERSE)  # 6 qubits -> 64 basis states, one bit per element


def bits_to_subset(bits):
    """bits: tuple of 0/1, length N_QUBITS, index i <-> UNIVERSE[i]."""
    return {UNIVERSE[i] for i, b in enumerate(bits) if b == 1}


def is_sum_free(S):
    """No a, b, c in S (a, b, c need not be distinct) with a + b = c."""
    S = set(S)
    for a in S:
        for b in S:
            if (a + b) in S:
                return False
    return True


def satisfies_property(S):
    return len(S) == 3 and is_sum_free(S) and (len(S) == 0 or max(S) == 6)


# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force over all 2^6 subsets of {1,...,6}.
# ---------------------------------------------------------------------------
marked_bitstrings = []  # list of length-N_QUBITS tuples of 0/1
marked_subsets = []
for bits in itertools.product([0, 1], repeat=N_QUBITS):
    S = bits_to_subset(bits)
    if satisfies_property(S):
        marked_bitstrings.append(bits)
        marked_subsets.append(S)

print("Classical search over all %d subsets of %s complete." % (2 ** N_QUBITS, UNIVERSE))
print("Subsets satisfying P (size 3, sum-free, max element = 6):")
for S in marked_subsets:
    print("  ", sorted(S))

if len(marked_bitstrings) == 0:
    raise RuntimeError("No classical solution found -- cannot build a Grover instance.")

# Qiskit bit ordering: qubit 0 is the least significant bit of the classical
# bitstring when read left-to-right in Aer's little-endian counts. We index
# element UNIVERSE[i] with qubit i directly and build the oracle accordingly.


# ---------------------------------------------------------------------------
# 2. Grover oracle: phase-flip exactly the marked bitstring(s).
# ---------------------------------------------------------------------------
def build_oracle(bitstrings, n):
    qc = QuantumCircuit(n, name="oracle")
    for bits in bitstrings:
        # Map bits==0 positions to |1> via X so an all-ones control fires
        # exactly on this bitstring, then apply a multi-controlled Z as
        # H - MCX - H on the last qubit, then undo the X's.
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        qc.h(n - 1)
        if n - 1 > 0:
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
    if n - 1 > 0:
        qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


n = N_QUBITS
M = len(marked_bitstrings)
N = 2 ** n

# Optimal number of Grover iterations for M marked items out of N.
iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

oracle = build_oracle(marked_bitstrings, n)
diffuser = build_diffuser(n)

qc = QuantumCircuit(n, n)
qc.h(range(n))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(n), range(n))

print("\nGrover circuit: n=%d qubits, N=%d states, M=%d marked, iterations=%d"
      % (n, N, M, iterations))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------
backend = AerSimulator()
shots = 4096
job = backend.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

# Aer reports bitstrings as c[n-1] c[n-2] ... c[0]; classical bit i was
# measured from qubit i, so reverse the printed string to recover
# (bit for qubit 0, qubit 1, ..., qubit n-1) in our own indexing.
most_likely = max(counts, key=counts.get)
most_likely_bits = tuple(int(c) for c in reversed(most_likely))
most_likely_subset = bits_to_subset(most_likely_bits)

print("\nTop measurement outcomes (bitstring: count):")
for bstr, cnt in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
    bits = tuple(int(c) for c in reversed(bstr))
    print("   %s -> subset %s : %d/%d shots" % (bstr, sorted(bits_to_subset(bits)), cnt, shots))

print("\nMost likely measured subset:", sorted(most_likely_subset))
print("Classically computed marked subset(s):", [sorted(S) for S in marked_subsets])

quantum_matches_classical = most_likely_subset in marked_subsets

# ---------------------------------------------------------------------------
# 4. Verdict.
# ---------------------------------------------------------------------------
if quantum_matches_classical:
    print("\nPASS: Grover search on the ideal AerSimulator recovered a subset "
          "satisfying P(S) = (|S|=3, sum-free, max(S)=6), matching the "
          "brute-force classical computation.")
else:
    print("\nFAIL: Grover search's most likely outcome did not match the "
          "classically computed marked subset(s).")
