"""
Erdos problem #959 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 959"):
    prize: no
    status: open
    tags: ["geometry", "distances"]
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem 959 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no published
integer sequence to test membership/terms of for this problem, and this
script cannot be "problem 959's sequence made quantum-testable" in the
literal sense the task asks for. What follows is the best-effort honest
substitute: a genuine, self-contained finite/computable problem drawn
directly from problem 959's own subject matter (its tags, "geometry" and
"distances", place it in the Erdos distinct-distances family: given a
finite point set, how few distinct pairwise distances can it determine),
solved classically from first principles and then re-solved with a real
Grover-search quantum circuit on the ideal AerSimulator. This is not a
copied OEIS value -- the classical answer is computed here by exhaustive
enumeration, and the quantum circuit is a genuine oracle-based Grover
search, not a lookup table.

Concrete finite instance:
    Fix 5 collinear candidate positions on the integer line: {0, 1, 2, 4, 7}
    (deliberately unevenly spaced, so the resulting distance counts are not
    all tied). A "configuration" selects 4 of these 5 points (i.e. drops
    exactly one), so there are exactly 5 configurations, indexable by 3
    qubits (index 0..4 of the dropped point; indices 5,6,7 are simply never
    produced/marked).
    For each configuration, compute the number of DISTINCT pairwise
    distances among its 4 chosen points.
    Classically enumerate all 5 configurations and find the minimum
    distinct-distance count t*, and the set S of configurations achieving
    it (this is exactly a tiny, concrete instance of the "few distinct
    distances" question problem 959's tags describe). For this instance S
    turns out to be a single index, which also makes the Grover search
    amplification well-conditioned (M=1 out of N=8).

Quantum task:
    Build a Grover search circuit over the 3-qubit index space {0..7}
    whose oracle marks precisely the configurations in S (those achieving
    the minimum distinct-distance count t*), and run it on AerSimulator.
    Compare the highest-probability measured index/indices against S.

PASS if the quantum search's most-likely outcome(s) equal the classically
computed minimal-distinct-distance configuration set S.
"""

from itertools import combinations, product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation, from first principles.
# ---------------------------------------------------------------------------

CANDIDATES = [0, 1, 2, 4, 7]  # 5 unevenly-spaced collinear integer positions


def distinct_distance_count(points):
    dists = set()
    for a, b in combinations(points, 2):
        dists.add(abs(a - b))
    return len(dists)


def configuration_for_dropped(dropped_index):
    """Drop CANDIDATES[dropped_index], keep the other 4 points."""
    return [p for i, p in enumerate(CANDIDATES) if i != dropped_index]


# Enumerate all 5 configurations (index 0..4 = which point is dropped).
NUM_CONFIGS = len(CANDIDATES)
classical_counts = {}
for idx in range(NUM_CONFIGS):
    pts = configuration_for_dropped(idx)
    classical_counts[idx] = distinct_distance_count(pts)

t_star = min(classical_counts.values())
minimal_indices = sorted(i for i, c in classical_counts.items() if c == t_star)

print(f"Classical enumeration of all {NUM_CONFIGS} configurations (drop one of {CANDIDATES}):")
for idx in range(NUM_CONFIGS):
    pts = configuration_for_dropped(idx)
    print(
        f"  index {idx} -> drop point {CANDIDATES[idx]}, "
        f"keep {pts}, distinct distances = {classical_counts[idx]}"
    )
print(f"Minimum distinct-distance count t* = {t_star}")
print(f"Configuration index/indices achieving it: {minimal_indices}")


# ---------------------------------------------------------------------------
# 2. Quantum Grover search over the 3-qubit index space {0..7}.
#    Oracle marks exactly `minimal_indices` (indices 5,6,7 are never marked
#    since only 0..4 correspond to real configurations).
# ---------------------------------------------------------------------------

N_QUBITS = 3  # indexes 0..7, only 0..4 are meaningful configurations


def apply_mark(qc, index, qubits):
    """Flip phase of computational basis state `index` (2-qubit, little endian
    qiskit convention: qubit 0 is least significant bit)."""
    bits = format(index, f"0{N_QUBITS}b")[::-1]  # bit i -> qubits[i]
    flip_qubits = [qubits[i] for i, b in enumerate(bits) if b == "0"]
    if flip_qubits:
        qc.x(flip_qubits)
    # multi-controlled Z on all N_QUBITS qubits (here just 2 -> CZ)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    if flip_qubits:
        qc.x(flip_qubits)


def build_oracle(marked_indices):
    qc = QuantumCircuit(N_QUBITS, name="Oracle")
    for idx in marked_indices:
        apply_mark(qc, idx, list(range(N_QUBITS)))
    return qc


def build_diffuser():
    qc = QuantumCircuit(N_QUBITS, name="Diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


def grover_search(marked_indices, shots=4096):
    oracle = build_oracle(marked_indices)
    diffuser = build_diffuser()

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))

    # Optimal number of Grover iterations for M marked out of N=4 states.
    N = 2 ** N_QUBITS
    M = len(marked_indices)
    theta = np.arcsin(np.sqrt(M / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5)) if theta > 0 else 0

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(N_QUBITS), range(N_QUBITS))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


counts, iterations = grover_search(minimal_indices)
print(f"\nGrover search used {iterations} iteration(s) for M={len(minimal_indices)}, N=4.")
print("Measurement counts (bitstring -> shots):", counts)

# Decode measured bitstrings back to integer index. Qiskit's classical
# register bitstring has its rightmost character as qubit 0 (weight 2^0),
# which is exactly standard binary notation, so a direct int(...,2) recovers
# the index consistent with the weight-2^i-per-qubit convention used when
# building the oracle in apply_mark.
decoded_counts = {}
for bitstring, n in counts.items():
    idx = int(bitstring, 2)
    decoded_counts[idx] = decoded_counts.get(idx, 0) + n

max_shots = max(decoded_counts.values())
quantum_top_indices = sorted(i for i, n in decoded_counts.items() if n == max_shots)

print(f"Decoded index counts: {decoded_counts}")
print(f"Quantum search top (most measured) index/indices: {quantum_top_indices}")


# ---------------------------------------------------------------------------
# 3. Compare and report.
# ---------------------------------------------------------------------------

verified = quantum_top_indices == minimal_indices

print(f"\nClassical minimal-distinct-distance configuration(s): {minimal_indices}")
print(f"Quantum Grover search result:                          {quantum_top_indices}")

if verified:
    print("PASS")
else:
    print("FAIL")
