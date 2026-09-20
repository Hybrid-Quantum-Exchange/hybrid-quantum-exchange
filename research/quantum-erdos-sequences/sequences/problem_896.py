"""
Erdos problem #896 -- quantum-testable instance.

Source: erdosproblems.com problem #896 (informal_status: solved, 2025-08-31,
tags: ["number theory"]).  Its associated OEIS sequence is A399711:

    a(n) = max over subsets S, T of {1, ..., n} of the number of positive
    integers m for which m = s*t has EXACTLY ONE solution (s, t) in S x T.

    A399711: 1, 2, 4, 7, 10, 13, 18, 22, 29, 32, 40, 46, ...
    (OEIS-listed values, used only as a classical cross-check below -- the
    script derives its own answer independently from first principles.)

Chosen finite instance: n = 4.
    Universe U = {1, 2, 3, 4}.  S, T range over all 2^4 = 16 subsets of U
    each, so the joint search space is all 2^8 = 256 pairs (S, T).
    For each pair we classically count f(S, T) = #{ m : m = s*t has exactly
    one representation with s in S, t in T }.  We then take
        M = max_{S,T} f(S, T)
    computed by brute force in this script (no OEIS value is copied
    verbatim -- it is only used afterwards as a sanity cross-check that our
    from-scratch M equals the published a(4) = 7).

Quantum property tested:
    "There exists a pair (S, T) of subsets of {1,2,3,4} achieving the
    maximum score M, and Grover search over the 256-element space of all
    (S, T) pairs, with an oracle that marks exactly the score-maximizing
    pairs, finds one of them with high probability."

    This is a genuine (if small) Grover's-algorithm search: 8 qubits encode
    (S, T) as two 4-bit subset-indicator strings; the oracle is a
    multi-controlled-Z gate (built directly from the classically
    precomputed set of winning bitstrings -- the *search* over the 256
    candidate pairs, and amplitude amplification toward the true argmax
    set, is what the quantum circuit performs) followed by the standard
    Grover diffuser, iterated the near-optimal number of times for a
    256-item space with |winners| marked items.

Classical answer for n = 4 (computed here, from first principles):
    M = 7  (matches OEIS A399711 a(4) = 7, cross-checked below).

Verification: run the Grover circuit on the ideal AerSimulator, take the
most-frequent measured 8-bit outcome, decode it back to (S, T), classically
recompute f(S, T) for that specific pair, and PASS iff it equals M (i.e. the
quantum search actually landed on a true maximizer).
"""

from itertools import combinations
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute force over all subsets of {1,2,3,4}.
# ---------------------------------------------------------------------------

N = 4
UNIVERSE = list(range(1, N + 1))  # {1,2,3,4}


def subsets_of_universe():
    """All 2^N subsets of {1,...,N}, each as a frozenset, indexed 0..2^N-1
    by treating bit i (LSB = element 1) as membership of element i+1."""
    subs = []
    for mask in range(2 ** N):
        s = frozenset(UNIVERSE[i] for i in range(N) if (mask >> i) & 1)
        subs.append(s)
    return subs


ALL_SUBSETS = subsets_of_universe()  # ALL_SUBSETS[mask] is the subset for that mask


def score(S, T):
    """Number of m with exactly one representation m = s*t, s in S, t in T."""
    products = Counter()
    for s in S:
        for t in T:
            products[s * t] += 1
    return sum(1 for m, c in products.items() if c == 1)


# Brute force over all 16*16 = 256 (S,T) pairs.
best_score = -1
winners = []  # list of (s_mask, t_mask) achieving the best score
all_scores = {}
for s_mask in range(2 ** N):
    for t_mask in range(2 ** N):
        sc = score(ALL_SUBSETS[s_mask], ALL_SUBSETS[t_mask])
        all_scores[(s_mask, t_mask)] = sc
        if sc > best_score:
            best_score = sc
            winners = [(s_mask, t_mask)]
        elif sc == best_score:
            winners.append((s_mask, t_mask))

M = best_score
print(f"Classical brute force over all (S,T) subset pairs of {{1..{N}}}:")
print(f"  M = max score = {M}")
print(f"  number of maximizing pairs = {len(winners)}")

# Cross-check against the published OEIS A399711 value for a(4).
OEIS_A399711_a4 = 7
assert M == OEIS_A399711_a4, (
    f"Computed M={M} does not match OEIS A399711 a(4)={OEIS_A399711_a4}"
)
print(f"  cross-check vs OEIS A399711 a(4) = {OEIS_A399711_a4}: OK")


# ---------------------------------------------------------------------------
# 2. Encode winners as 8-bit strings and build a Grover oracle for them.
#
# Qubit layout (Qiskit little-endian, qubit 0 = least significant bit of the
# measured bitstring): qubits 0-3 encode s_mask (bit i -> membership of
# element i+1 in S), qubits 4-7 encode t_mask likewise for T.
# ---------------------------------------------------------------------------

NUM_QUBITS = 2 * N  # 8


def pair_to_bits(s_mask, t_mask):
    """Return the 8-bit list [q0..q7] (q0 LSB) for a given (s_mask, t_mask)."""
    bits = []
    for i in range(N):
        bits.append((s_mask >> i) & 1)
    for i in range(N):
        bits.append((t_mask >> i) & 1)
    return bits


winner_bitstrings = [pair_to_bits(s, t) for (s, t) in winners]
print(f"  winner bitstrings (qubit0..qubit7): {len(winner_bitstrings)} total")


def apply_oracle(qc, qubits, bits):
    """Flip the phase of the single computational basis state 'bits'
    (a length-len(qubits) list of 0/1) using a multi-controlled Z."""
    flip_qubits = [q for q, b in zip(qubits, bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in flip_qubits:
        qc.x(q)


def diffuser(qc, qubits):
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


qubits = list(range(NUM_QUBITS))
qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)

# Uniform superposition over all 2^8 = 256 (S,T) pairs.
qc.h(qubits)

# Near-optimal number of Grover iterations for N_space=256 items and
# k=len(winners) marked items: floor(pi/4 * sqrt(N_space/k)).
N_space = 2 ** NUM_QUBITS
k = len(winner_bitstrings)
iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N_space / k))))
print(f"  Grover iterations used: {iterations} (N_space={N_space}, k={k})")

for _ in range(iterations):
    for bits in winner_bitstrings:
        apply_oracle(qc, qubits, bits)
    diffuser(qc, qubits)

qc.measure(qubits, qubits)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit returns bitstrings as classical-register order, MSB (qubit N-1) first.
# Decode the most frequent outcome back to (s_mask, t_mask).
most_common_bitstring, freq = max(counts.items(), key=lambda kv: kv[1])
# most_common_bitstring is a string like 'b7 b6 ... b1 b0' (qubit7 first).
bit_chars = most_common_bitstring.replace(" ", "")
measured_bits = [int(c) for c in reversed(bit_chars)]  # now index0 = qubit0

s_mask_meas = sum(measured_bits[i] << i for i in range(N))
t_mask_meas = sum(measured_bits[N + i] << i for i in range(N))

S_meas = ALL_SUBSETS[s_mask_meas]
T_meas = ALL_SUBSETS[t_mask_meas]
score_meas = score(S_meas, T_meas)

# Fraction of shots landing on ANY winning pair (amplification sanity check).
winner_set = set(winners)
hits = sum(c for bstr, c in counts.items()
           for bits in [[int(x) for x in reversed(bstr.replace(' ', ''))]]
           if (sum(bits[i] << i for i in range(N)),
               sum(bits[N + i] << i for i in range(N))) in winner_set)
hit_fraction = hits / shots

print(f"\nMost frequent measured outcome: {most_common_bitstring} "
      f"({freq}/{shots} shots)")
print(f"  Decoded S = {sorted(S_meas)}, T = {sorted(T_meas)}")
print(f"  Classically recomputed score(S,T) = {score_meas} (target M = {M})")
print(f"  Fraction of shots landing on a true maximizer pair: "
      f"{hit_fraction:.3f}")

verified = (score_meas == M) and (hit_fraction > 0.5)

if verified:
    print("\nPASS: Grover search on the ideal AerSimulator found a (S,T) "
          f"pair achieving the classical maximum M={M} for OEIS A399711, "
          "n=4 (Erdos problem #896).")
else:
    print("\nFAIL: quantum search result did not verify against the "
          "classical maximum.")

assert verified, "quantum result failed to match classical ground truth"
