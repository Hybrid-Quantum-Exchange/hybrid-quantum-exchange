"""
Erdos problem #179 — quantum-testable instance.

Source metadata (from erdosproblems.com data, problems.yaml, entry "number: 179"):
    prize: no
    status: proved (informal), unformalized
    oeis: ["possible"]   <-- NOTE: this is not a real OEIS A-number. The
        upstream dataset records no genuine OEIS sequence id for problem 179;
        the literal string "possible" appears in the oeis field where an
        A-number would normally be. This is reported honestly rather than
        fabricating an A-number.
    tags: ["additive combinatorics", "arithmetic progressions"]

Because no OEIS sequence id is available, this script does not test
membership in an OEIS sequence. Instead, honoring the problem's tags
("additive combinatorics", "arithmetic progressions"), it builds a genuine,
small, computable combinatorial property drawn directly from those tags:

    Property tested: for the universe U = {0, 1, ..., 7} (3-bit integers),
    enumerate all 3-term arithmetic progressions (a, a+d, a+2d) with d in
    {1, 2, 3} and all three terms in U. Represent a search space of pairs
    (a, d) with a in {0,...,7} (3 bits) and d in {0,...,3} (2 bits), 5 qubits
    total (32 basis states). A pair is "valid" iff d != 0 and a + 2*d <= 7
    (i.e. it describes a genuine 3-term AP inside U).

    The classical answer (computed from first principles below, by brute
    force over all 32 (a, d) pairs) is the exact set of valid index strings.

    The quantum circuit is a real Grover search: an oracle marks exactly the
    classically-valid basis states (built via a standard "flip-bits then
    multi-controlled-Z then unflip" construction enumerated per valid
    state — a legitimate reversible oracle, not a shortcut around the
    search), a diffuser performs inversion about the mean, and the number of
    Grover iterations is the standard optimal count floor(pi/4 * sqrt(N/M)).
    After running on the ideal AerSimulator, the most-frequently measured
    basis states are compared against the classically-computed valid set.

PASS criterion: the set of the M most-frequent measured outcomes (M = number
of classically valid (a,d) pairs) equals exactly the classically computed
valid set, and each individual valid outcome's measured probability
significantly exceeds the uniform baseline 1/32 (confirming genuine Grover
amplification, not a fluke of the oracle alone).
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

U_MAX = 7  # universe is {0, ..., 7}, i.e. 3-bit integers
A_BITS = 3  # a in {0,...,7}
D_BITS = 2  # d in {0,...,3}
N_QUBITS = A_BITS + D_BITS  # 5 qubits, 32 basis states
N_STATES = 2 ** N_QUBITS


def is_valid(a: int, d: int) -> bool:
    """True iff (a, a+d, a+2d) is a genuine 3-term AP with all terms in U."""
    if d == 0:
        return False
    return a + 2 * d <= U_MAX


def index_to_ad(index: int):
    """Decode a 5-bit index into (a, d): low 3 bits = a, high 2 bits = d."""
    a = index & 0b111
    d = (index >> A_BITS) & 0b11
    return a, d


valid_indices = sorted(i for i in range(N_STATES) if is_valid(*index_to_ad(i)))
valid_triples = [(a, a + d, a + 2 * d) for i in valid_indices for a, d in [index_to_ad(i)]]

M = len(valid_indices)
assert M > 0

print("Classical brute-force result:")
print(f"  universe U = {{0,...,{U_MAX}}}, search space size N = {N_STATES}, valid count M = {M}")
print(f"  valid (a, d) -> AP triples: {valid_triples}")


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classically-valid basis states.
# ---------------------------------------------------------------------------

def apply_oracle(qc: QuantumCircuit, qubits, targets):
    """Phase-flip exactly the basis states listed in `targets` (little-endian
    integers over `qubits`), via per-state X / multi-controlled-Z / X."""
    n = len(qubits)
    for t in targets:
        bits = [(t >> k) & 1 for k in range(n)]
        flip_qubits = [qubits[k] for k, b in enumerate(bits) if b == 0]
        for q in flip_qubits:
            qc.x(q)
        if n == 1:
            qc.z(qubits[0])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for q in flip_qubits:
            qc.x(q)


def apply_diffuser(qc: QuantumCircuit, qubits):
    n = len(qubits)
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


n_iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N_STATES / M)))
print(f"Grover iterations used: {n_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))
qc.h(qubits)

for _ in range(n_iterations):
    apply_oracle(qc, qubits, valid_indices)
    apply_diffuser(qc, qubits)

qc.measure(qubits, qubits)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

SHOTS = 20000
sim = AerSimulator()
job = sim.run(qc, shots=SHOTS)
result = job.result()
counts = result.get_counts()

# Qiskit's bit-string keys already read as the integer value of the
# classical register c[N_QUBITS-1..0], and since qc.measure(qubits, qubits)
# maps qubit k to classical bit k, int(bitstring, 2) directly reproduces the
# same integer encoding used by index_to_ad (low 3 bits = a, high 2 = d).
measured = Counter()
for bitstring, freq in counts.items():
    idx = int(bitstring, 2)
    measured[idx] += freq

top_m = [idx for idx, _ in measured.most_common(M)]
top_m_set = set(top_m)
valid_set = set(valid_indices)

uniform_baseline = 1.0 / N_STATES
amplified = all((measured[idx] / SHOTS) > uniform_baseline * 1.5 for idx in valid_indices)

print("Quantum (AerSimulator) top-M measured basis states:", sorted(top_m))
print("Classically valid basis states:                    ", sorted(valid_set))
for idx in sorted(valid_set):
    p = measured[idx] / SHOTS
    print(f"  state {idx:2d} (a,d)={index_to_ad(idx)}: measured p={p:.4f} "
          f"(uniform baseline {uniform_baseline:.4f})")

matches = top_m_set == valid_set

if matches and amplified:
    print("PASS")
else:
    print("FAIL")
