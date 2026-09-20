"""
Erdos problem #984 -- quantum-testable sequence lane.

Source metadata (erdosproblems.com data, /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: \"984\""):
    prize: no
    status: proved (as of 2025-08-31)
    oeis: ["N/A"]   <-- no OEIS sequence id is attached to this problem
    tags: ["arithmetic progressions", "additive combinatorics"]

LIMITATION, stated up front: problem 984 carries no OEIS id, so there is no
literal sequence to look up or derive a term from. Per the task instructions,
this script proceeds with the best honest attempt: it builds a genuine,
classically-verifiable finite instance of the *kind* of object the problem's
tags describe (a 3-term-arithmetic-progression-free set of integers, the
central object of "arithmetic progressions" / "additive combinatorics"
problems in this area), and tests it with a real Grover search circuit. This
is not a literal OEIS-sequence membership test -- it is the closest genuine,
computable, small-instance property available given the missing OEIS id.

The classical object
---------------------
Behrend/Salem-Spencer-style construction: write n in base 3. The set of
integers whose base-3 digits are all in {0, 1} (no digit equal to 2) is a
classic example of a set containing no 3-term arithmetic progression (if
x, y, z are in the set with y - x = z - y, then adding the "no carries in
base 3" digit strings component-wise forces x = y = z). We test this on the finite instance
n in {0, ..., 31} (5 qubits), using 3-digit base-3 representations (valid
for n < 3**3 = 27; n in {27,...,31} are simply extra, always-unmarked
padding states needed to round the search space up to a power of two).

For n in 0..26, the base-3 digits (d2 d1 d0, n = 9*d2 + 3*d1 + d0) with
every digit in {0, 1} (no digit equal to 2) give:
    S = {0, 1, 3, 4, 9, 10, 12, 13}
All other n in {0, ..., 31} (including all of 27..31) are excluded.

Classical property under test: S contains no 3-term arithmetic progression
(no triple x < y < z in S with y - x == z - y). This is verified in this
script by brute-force enumeration over all triples of S -- first
principles, no OEIS lookup.

Quantum circuit
----------------
A genuine Grover search over the 5-qubit register n in {0,...,31}:
  - The oracle phase-flips exactly the 8 basis states in S -- the AP-free
    set computed classically above (marking is built with explicit
    X/multi-controlled-Z sandwiches per target bit pattern, the standard
    construction for a Grover oracle over a known target set).
  - |S| / N = 8 / 32 = 1/4, so theta = asin(sqrt(1/4)) = 30 degrees, and the
    Grover-optimal iteration count round(pi/(4*theta) - 1/2) = 1 rotates the
    state by exactly 3*theta = 90 degrees -- i.e. one iteration should put
    (ideally) all amplitude on S.
  - The circuit is run on the ideal AerSimulator (statevector via
    qiskit_aer), 4096 shots, and the measured distribution is compared
    against the classical set S.

PASS criterion: essentially all measurement shots land in S, and no shots
(beyond simulator/shot noise, checked against a strict threshold) land
outside S -- i.e. the quantum search recovers exactly the classically
verified AP-free set.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, derived from first principles (no OEIS lookup).
# ---------------------------------------------------------------------------

def base3_digits(n: int, ndigits: int = 3):
    digits = []
    for _ in range(ndigits):
        digits.append(n % 3)
        n //= 3
    return digits  # least-significant first


def no_digit_two(n: int, ndigits: int = 3) -> bool:
    """True iff n's base-3 representation, using exactly `ndigits` digits,
    has no digit equal to 2. Only meaningful for n < 3**ndigits -- callers
    must bound n themselves, since digit extraction wraps (mod 3**ndigits)
    otherwise."""
    return all(d != 2 for d in base3_digits(n, ndigits))


N = 32       # instance size: n in {0, ..., 31}, 5 qubits
BASE3_RANGE = 3 ** 3  # = 27; base-3 digit test is only valid for n < 27
S = [n for n in range(N) if n < BASE3_RANGE and no_digit_two(n)]
NOT_S = [n for n in range(N) if n not in S]

assert S == [0, 1, 3, 4, 9, 10, 12, 13], f"unexpected base-3 construction result: {S}"


def is_ap_free(s):
    """Brute-force check: no x < y < z in s with y - x == z - y."""
    for x, y, z in combinations(sorted(s), 3):
        if y - x == z - y:
            return False, (x, y, z)
    return True, None


ap_free, witness = is_ap_free(S)
assert ap_free, f"classical construction failed to be AP-free, found {witness}"

print(f"Classical instance: N = {N}")
print(f"S (no digit '2' in base 3, expected AP-free)      = {S}")
print(f"complement (has a digit '2', expected NOT special) = {NOT_S}")
print(f"Classical brute-force check: S is 3-AP-free -> {ap_free}")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search marking exactly S = {0, 1, 3, 4}.
# ---------------------------------------------------------------------------

NUM_QUBITS = 5  # 2**5 = 32 = N


def mark_state(qc: QuantumCircuit, bitstring: str):
    """Phase-flip the basis state given by bitstring (qubit-0-first order)
    using an X-sandwiched multi-controlled-Z, the standard Grover oracle
    construction for a single known target state."""
    # bitstring[i] corresponds to qubit i (little endian, matches Qiskit's
    # qubit ordering when we read bitstring = format(n, '0{}b'.format(k))[::-1])
    zero_positions = [i for i, b in enumerate(bitstring) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    if NUM_QUBITS == 1:
        qc.z(0)
    elif NUM_QUBITS == 2:
        qc.cz(0, 1)
    else:
        qc.h(NUM_QUBITS - 1)
        qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
        qc.h(NUM_QUBITS - 1)
    for i in zero_positions:
        qc.x(i)


def build_oracle(marked_values, num_qubits):
    qc = QuantumCircuit(num_qubits, name="oracle")
    for n in marked_values:
        bits = format(n, f"0{num_qubits}b")[::-1]  # little-endian per qubit
        mark_state(qc, bits)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_grover_circuit(marked_values, num_qubits, iterations):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    oracle = build_oracle(marked_values, num_qubits)
    diffuser = build_diffuser(num_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


# Optimal-ish iteration count for |marked|/|N| = 4/8 = 1/2: theta = asin(sqrt(1/2)),
# optimal integer iterations round(pi/(4*theta) - 1/2).
theta = np.arcsin(np.sqrt(len(S) / N))
iterations = max(1, round(np.pi / (4 * theta) - 0.5))

circuit = build_grover_circuit(S, NUM_QUBITS, iterations)

SHOTS = 4096
sim = AerSimulator()
transpiled = transpile(circuit, basis_gates=["u", "cx"])
job = sim.run(transpiled, shots=SHOTS)
result = job.result()
counts = result.get_counts()

# Qiskit's classical-register count keys are written MSB (highest clbit)
# first, i.e. clbit (n-1) ... clbit 0 left to right -- the same convention
# used when indexing the statevector directly, so the key parses straight
# to the integer n with no bit reversal needed.
measured_values = {}
for bitstring, c in counts.items():
    n = int(bitstring, 2)
    measured_values[n] = measured_values.get(n, 0) + c

print(f"\nGrover circuit: {NUM_QUBITS} qubits, {iterations} iteration(s), {SHOTS} shots")
print("Measured distribution over n in 0..7:")
for n in range(N):
    print(f"  n={n} ({'in S' if n in S else 'not in S'}): {measured_values.get(n, 0)} shots")


# ---------------------------------------------------------------------------
# 3. Compare quantum result against the classical answer.
# ---------------------------------------------------------------------------

shots_in_S = sum(measured_values.get(n, 0) for n in S)
shots_not_in_S = sum(measured_values.get(n, 0) for n in NOT_S)

frac_in_S = shots_in_S / SHOTS
frac_not_in_S = shots_not_in_S / SHOTS

print(f"\nFraction of shots landing in classically-verified S: {frac_in_S:.4f}")
print(f"Fraction of shots landing outside S:                  {frac_not_in_S:.4f}")

# On the ideal simulator with a correctly built oracle+diffuser for a 50%
# marked fraction, essentially all amplitude should be on S already after
# the H layer even before Grover iterations meaningfully help (since the
# oracle only ever marks the classically-computed S); we still require an
# overwhelming majority to land in S as the pass bar.
THRESHOLD = 0.98
verified = ap_free and (frac_in_S >= THRESHOLD)

print(f"\nRESULT: {'PASS' if verified else 'FAIL'}")
if not verified:
    raise SystemExit(1)
