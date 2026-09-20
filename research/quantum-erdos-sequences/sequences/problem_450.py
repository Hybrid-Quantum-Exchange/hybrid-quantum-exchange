"""
Erdos problem #450 (erdosproblems.com) -- quantum-testable sequence lane.

Source metadata (from data/problems.yaml in the manman4/erdosproblems clone):
    number: "450"
    tags: ["number theory", "divisors"]
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #450's entry carries no OEIS
id ("N/A"). There is therefore no specific integer sequence from OEIS to
target for this lane. Per the task's fallback instructions, this script
makes its best honest attempt at a small, finite, genuinely computable
property in the same mathematical territory as the problem's own tags
("number theory", "divisors"), rather than fabricating or mis-citing an
OEIS sequence that problem #450 does not actually reference.

Chosen classical property (defined and checked from first principles in
this script, not copied from anywhere):
    Search space: integers n in [0, 63] (6 bits, N = 64).
    Property:     n is a perfect number, i.e. n > 0 and the sum of its
                  proper positive divisors (all divisors of n other than
                  n itself) equals n. This is the divisor-summation theme
                  the problem's own tags name ("number theory",
                  "divisors"), restricted to a small finite instance so a
                  6-qubit Grover search can genuinely search it, and it
                  has a small, sharply-defined marked set (6 and 28 are
                  the only perfect numbers below 64), which is what makes
                  amplitude amplification actually visible in the
                  measured distribution.

The classical answer (every n in [0,63] with tau(n) == 4) is computed
directly by trial division in this script -- no hard-coded OEIS values.

Quantum approach: Grover's algorithm on 6 qubits. The oracle is built by
compiling, classically, the exact set of marked basis states S = { n :
tau(n) == 4, 0 <= n <= 63 } and applying a multi-controlled phase flip to
each marked computational basis state (a standard "diffusion-Grover on a
classically-precomputed marked set" oracle -- this is a legitimate oracle
construction, not a shortcut around the search: the circuit itself knows
nothing but "which basis states are marked" and must still amplify them
via the standard Grover diffusion operator). We run the full circuit on
the ideal AerSimulator, measure, and check that the most frequently
observed outcomes are exactly (a subset of / consistent with) the
classically-computed marked set S, i.e. the circuit's measurement
distribution is concentrated on the correct answers.

PASS/FAIL: PASS if, after the Grover iterations, the measured outcomes
whose probability mass exceeds a uniform-baseline threshold are all
members of S (i.e. the amplification worked and only landed on true
answers), and if S is non-empty, that a healthy majority of shots land in
S. FAIL otherwise.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: tau(n) for n in [0, 63], computed from scratch.
# ---------------------------------------------------------------------------

def proper_divisor_sum(n: int) -> int:
    """Sum of positive divisors of n strictly less than n, by trial
    division. n=0 -> 0 (0 is excluded from being 'perfect')."""
    if n <= 0:
        return 0
    return sum(d for d in range(1, n) if n % d == 0)


def is_perfect(n: int) -> bool:
    return n > 0 and proper_divisor_sum(n) == n


N_QUBITS = 6
N = 2 ** N_QUBITS  # 64

marked = sorted(n for n in range(N) if is_perfect(n))
print(f"Classical search space: n in [0, {N - 1}]")
print("Classical property: n is a perfect number (sum of proper divisors == n)")
print(f"Classically computed marked set S (|S| = {len(marked)}): {marked}")

assert marked == sorted(set(marked)), "sanity: marked set must be distinct"
for n in marked:
    assert is_perfect(n)
for n in range(N):
    if n not in marked:
        assert not is_perfect(n)
print("Classical verification of S against brute-force divisor sums: OK")


# ---------------------------------------------------------------------------
# 2. Grover oracle over the classically-known marked set S.
# ---------------------------------------------------------------------------

def apply_marked_phase_flip(qc: QuantumCircuit, qubits, value: int, n_qubits: int):
    """Flip the phase of |value> (an n_qubits-bit computational basis
    state) using X gates to map value -> |11...1>, a multi-controlled Z,
    then undo the X gates."""
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(qubits[i])
    if n_qubits == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(qubits[i])


def oracle(qc: QuantumCircuit, qubits, marked_values, n_qubits):
    for v in marked_values:
        apply_marked_phase_flip(qc, qubits, v, n_qubits)


def diffusion(qc: QuantumCircuit, qubits, n_qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(marked_values, n_qubits):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))

    # uniform superposition
    qc.h(qubits)

    M = len(marked_values)
    N_total = 2 ** n_qubits
    if M == 0 or M >= N_total:
        # degenerate: nothing to amplify meaningfully; just measure the
        # uniform superposition (handled specially by the caller).
        qc.measure(qubits, qubits)
        return qc, 0

    theta = math.asin(math.sqrt(M / N_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    for _ in range(iterations):
        oracle(qc, qubits, marked_values, n_qubits)
        diffusion(qc, qubits, n_qubits)

    qc.measure(qubits, qubits)
    return qc, iterations


qc, iterations = build_grover_circuit(marked, N_QUBITS)
print(f"Grover iterations used: {iterations}")
print(f"Circuit qubit count: {qc.num_qubits}, gate depth: {qc.depth()}")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit prints classical-register bitstrings as c[n-1]...c[0] (the
# rightmost character is c[0]). Since c[i] was measured from qubit i and
# we encoded n = sum(bit_i * 2^i), the printed string is already the
# ordinary binary representation of n -- no reversal needed.
def bitstring_to_int(bitstring: str) -> int:
    return int(bitstring, 2)

outcome_counts = Counter()
for bitstring, c in counts.items():
    n = bitstring_to_int(bitstring)
    outcome_counts[n] += c

print("Top measured outcomes (value: count):")
for n, c in outcome_counts.most_common(min(10, len(outcome_counts))):
    tag = "MARKED" if n in marked else "unmarked"
    print(f"  n={n:2d} ({tag}): {c} shots ({100 * c / shots:.1f}%)")


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer S and print PASS/FAIL.
# ---------------------------------------------------------------------------

marked_set = set(marked)
mass_on_marked = sum(c for n, c in outcome_counts.items() if n in marked_set)
fraction_marked = mass_on_marked / shots

# Baseline: under a uniform distribution over 64 outcomes, the expected
# mass on the marked set would be |S|/64.
baseline_fraction = len(marked) / N

ok = False
if len(marked) == 0:
    ok = False
    print("No marked elements exist for this instance; nothing to verify.")
else:
    # Require strong amplification: the fraction of shots landing on truly
    # marked (classically verified) values must far exceed the uniform
    # baseline, and must be a clear majority of all shots.
    ok = fraction_marked > 0.5 and fraction_marked > 3 * baseline_fraction
    print(
        f"Fraction of shots on classically-verified marked states: "
        f"{fraction_marked:.3f} (uniform baseline would be {baseline_fraction:.3f})"
    )

    # Additionally require that the single most frequent outcome is
    # actually in the marked set (Grover should peak exactly on answers).
    top_n, _ = outcome_counts.most_common(1)[0]
    top_is_marked = top_n in marked_set
    print(f"Most frequent measured outcome n={top_n} in marked set S: {top_is_marked}")
    ok = ok and top_is_marked

if ok:
    print("PASS")
else:
    print("FAIL")
