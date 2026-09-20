"""
Erdos problem #994 (erdosproblems.com), quantum-testable lane.

Source metadata (from manman4/erdosproblems data/problems.yaml, entry
`number: "994"`): prize "no", status "disproved", tags ["analysis",
"discrepancy"], oeis: ["N/A"]. There is NO OEIS sequence id attached to
this problem -- the listed value is the literal string "N/A". Per the task
instructions, an honest attempt is written here rather than a fabricated
sequence property, and the limitation is stated explicitly: this script
does NOT test membership in, or any term of, an actual OEIS sequence tied
to Erdos problem #994, because none is listed in the source data, and the
problem's own statement (an analysis/discrepancy question about infinite
objects) is not itself finite. ran_ok and verified_against_classical are
reported honestly for what this script actually checks.

Given the problem's tags ("analysis", "discrepancy"), the best small,
finite, computable stand-in genuinely in that spirit -- and the one
discrepancy problem most associated with Erdos in this exact area -- is a
tiny instance of the classical Erdos discrepancy problem itself: for a
sign sequence x(1..n) in {-1, +1}^n, its discrepancy along homogeneous
arithmetic progressions (HAPs) is

    D(x) = max over d >= 1, k >= 1, with k*d <= n, of | sum_{i=1}^{k} x(i*d) |.

Property tested here (finite, computable, and checked classically from
first principles, not copied from any table):

    For n = 4 (so x is a length-4 +-1 sequence, one qubit per sign, 2^4 =
    16 possible sequences), what is the minimum possible discrepancy
    D_min = min over all 16 sign sequences of D(x), and which sequences
    achieve it?

    Classically (brute force over all 16 sequences, enumerating every HAP
    for n=4: d=1 -> {1,2,3,4}; d=2 -> {2,4}; d=3 -> {3}; d=4 -> {4}; and
    every prefix of each), the script computes D_min and the full set of
    minimizing sequences.

Quantum approach: a real Grover search circuit on 4 qubits (one qubit per
sign, |0> = -1, |1> = +1) whose oracle marks exactly the classically
verified minimizing bitstrings (a diagonal phase oracle built from that
ground-truth list via multi-controlled-Z gates -- the amplification is
genuinely done by the quantum circuit; no classical answer is invented,
only the already-verified property is encoded as a phase flip so Grover
can search for it). The circuit is run on the ideal AerSimulator and its
measured samples are checked against the classical ground truth.

Limitation: this is a faithful small Grover search over a real, checkable
discrepancy property in the same area as problem #994's tags, not a test
of a specific documented OEIS integer sequence for problem #994 (none is
listed for it in the source data, and the problem itself concerns
infinite sequences, not a finite table entry).
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# Step 1: classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N = 4  # sequence length x(1..4)
N_QUBITS = N


def bits_to_signs(bits: str):
    """bits[i] = qubit i = x(i+1). '0' -> -1, '1' -> +1 (LSB-first, qubit 0
    is x(1))."""
    return [1 if b == "1" else -1 for b in bits]


def haps(n: int):
    """All homogeneous arithmetic progressions {d, 2d, 3d, ...} within
    [1, n], for every common difference d = 1..n."""
    progressions = []
    for d in range(1, n + 1):
        prog = []
        k = 1
        while k * d <= n:
            prog.append(k * d)
            k += 1
        if prog:
            progressions.append(prog)
    return progressions


PROGRESSIONS = haps(N)
assert PROGRESSIONS == [[1, 2, 3, 4], [2, 4], [3], [4]], PROGRESSIONS


def discrepancy(signs):
    """D(x) = max over HAPs {d, 2d, ...} and over every prefix length k of
    that HAP, of |sum of the first k terms x(d), x(2d), ..., x(kd)|."""
    worst = 0
    for prog in PROGRESSIONS:
        running = 0
        for idx in prog:
            running += signs[idx - 1]
            worst = max(worst, abs(running))
    return worst


all_bitstrings = ["".join(b) for b in itertools.product("01", repeat=N_QUBITS)]
disc_by_bits = {bits: discrepancy(bits_to_signs(bits)) for bits in all_bitstrings}

D_min = min(disc_by_bits.values())
marked_bitstrings = [bits for bits, d in disc_by_bits.items() if d == D_min]
M = len(marked_bitstrings)

print(f"[classical] search space: 2^{N_QUBITS} = {len(all_bitstrings)} sign sequences of length {N}")
print(f"[classical] homogeneous APs used: {PROGRESSIONS}")
print(f"[classical] minimum discrepancy D_min = {D_min}")
print(f"[classical] # minimizing sequences M = {M}")
for bits in marked_bitstrings:
    print(f"    bits={bits} -> signs={bits_to_signs(bits)} -> D={disc_by_bits[bits]}")

# Independent double-check by exhaustively recomputing D_min a second,
# differently-structured way (direct nested loop, no dict/comprehension),
# to catch any bug in the first computation.
D_min_check = None
count_check = 0
for signs in itertools.product([-1, 1], repeat=N):
    worst = 0
    for d in range(1, N + 1):
        s = 0
        k = 1
        while k * d <= N:
            s += signs[k * d - 1]
            if abs(s) > worst:
                worst = abs(s)
            k += 1
    if D_min_check is None or worst < D_min_check:
        D_min_check = worst
        count_check = 1
    elif worst == D_min_check:
        count_check += 1

assert D_min_check == D_min, (D_min_check, D_min)
assert count_check == M, (count_check, M)
assert 0 < M < len(all_bitstrings), "oracle must mark a proper non-empty subset for Grover to be meaningful"

# ---------------------------------------------------------------------------
# Step 2: build a real Grover search circuit whose oracle marks exactly the
# classically-verified minimum-discrepancy bitstrings.
# ---------------------------------------------------------------------------


def multi_controlled_z_on_bitstring(qc: QuantumCircuit, bits: str, qubits):
    """Flip the phase of exactly the computational basis state |bits>,
    leaving every other basis state unchanged."""
    zero_positions = [q for q, b in zip(qubits, bits) if b == "0"]
    for q in zero_positions:
        qc.x(q)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in zero_positions:
        qc.x(q)


def build_oracle(marked, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in marked:
        multi_controlled_z_on_bitstring(qc, bits, list(range(n_qubits)))
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(marked_bitstrings, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

N_STATES = 2 ** N_QUBITS
n_iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N_STATES / M)))
theoretical_success_prob = math.sin((2 * n_iterations + 1) * math.asin(math.sqrt(M / N_STATES))) ** 2

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(n_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))

print(f"\n[quantum] {N_QUBITS} qubits, {M}/{N_STATES} marked (min-discrepancy) states, "
      f"{n_iterations} Grover iteration(s)")
print(f"[quantum] theoretical success probability ~ {theoretical_success_prob:.4f}")

# ---------------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator and check the result against the
# classically-verified minimum-discrepancy set.
# ---------------------------------------------------------------------------

SHOTS = 4096
sim = AerSimulator()
result = sim.run(qc, shots=SHOTS).result()
counts = result.get_counts()

marked_set = set(marked_bitstrings)


def qiskit_key_to_bits(key: str) -> str:
    # Qiskit's returned bitstring is big-endian (qubit n-1 ... qubit 0);
    # our convention above is LSB-first (qubit 0 = x(1)), so reverse.
    return key[::-1]


hits = sum(c for k, c in counts.items() if qiskit_key_to_bits(k) in marked_set)
measured_success_prob = hits / SHOTS
baseline = M / N_STATES

print(f"[quantum] measured success probability over {SHOTS} shots: {measured_success_prob:.4f}")
print(f"[quantum] uniform-random baseline: {baseline:.4f}")

sample_bits = qiskit_key_to_bits(max(counts, key=counts.get))
most_common_signs = bits_to_signs(sample_bits)
most_common_d = discrepancy(most_common_signs)
print(f"[quantum] most sampled bitstring (LSB-first) = {sample_bits} -> signs {most_common_signs} "
      f"-> discrepancy {most_common_d} (D_min = {D_min})")

MARGIN_THRESHOLD = 0.60  # pre-registered: well above the uniform baseline

classical_ok = (D_min == D_min_check) and (M == count_check) and (0 < M < N_STATES)
quantum_ok = (measured_success_prob >= MARGIN_THRESHOLD) and (most_common_d == D_min)

print(f"\nclassical property (min discrepancy over length-{N} +-1 sequences) verified from first principles: {classical_ok}")
print(f"quantum amplitude amplification verified "
      f"(>= {MARGIN_THRESHOLD:.0%}, baseline {baseline:.0%}): {quantum_ok}")

if classical_ok and quantum_ok:
    print("PASS")
else:
    print("FAIL")
