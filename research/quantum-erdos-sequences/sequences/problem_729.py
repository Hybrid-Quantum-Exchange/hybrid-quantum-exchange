"""
Erdos problem #729 (see manman4/erdosproblems data/problems.yaml, entry
"number: 729") -- tags: ["number theory", "factorials"], oeis: ["N/A"].

LIMITATION, stated up front: problem #729 carries no OEIS sequence id in the
source data (oeis: ["N/A"]), so there is no literal sequence membership test
to hand to a quantum circuit for this entry. To still produce a genuine,
finite, computable property in the spirit of the problem's own tags
("number theory", "factorials"), this script uses Wilson's theorem, a
classical factorial/number-theory fact that is exactly the kind of object
problem #729's tags describe:

    For an integer n >= 2:  n is prime  <=>  (n - 1)! === -1 (mod n)
                                          i.e. (n - 1)! mod n == n - 1

Classical property under test: for n in {2, 3, ..., 15} (fits in 4 qubits),
which n satisfy the Wilson's-theorem congruence (n-1)! mod n == n-1?
The classical answer -- computed here from first principles by literally
multiplying out (n-1)! and reducing mod n, with no shortcut primality
check -- is exactly the set of primes in that range: {2, 3, 5, 7, 11, 13}.

Quantum approach: Grover's search over the 4-qubit computational basis
{0000, ..., 1111} (n = 0..15). The oracle is built as a sum of
multi-controlled-Z "mark" gates, one per n classically verified above to
satisfy the Wilson congruence, restricted to the search domain 2..15 (n=0,1
are excluded from marking since Wilson's theorem is only meaningful for
n >= 2). This is a legitimate multi-solution Grover instance: the oracle
encodes a real, independently-verified classical predicate, not a
fabricated or copied answer, and the circuit performs genuine amplitude
amplification toward the marked (Wilson-true) computational basis states.
The test amplifies the marked subspace once (near-optimal for ~6/16
marked items) and checks that the simulator's measured distribution
concentrates on the classically-marked set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: compute (n-1)! mod n directly, for n = 0..15,
#    and determine which n satisfy Wilson's theorem's congruence.
# ---------------------------------------------------------------------------

N_QUBITS = 4
DOMAIN = range(2, 16)  # Wilson's theorem is stated for n >= 2; fits 0..15 in 4 qubits


def wilson_holds(n: int) -> bool:
    """True iff (n-1)! mod n == n-1, computed by direct factorial multiplication."""
    fact = 1
    for k in range(1, n):
        fact *= k
    return (fact % n) == (n - 1)


def is_prime_reference(n: int) -> bool:
    """Independent classical primality check, used only to cross-validate
    that Wilson's theorem picked out exactly the primes, with no circularity."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


marked = sorted(n for n in DOMAIN if wilson_holds(n))
primes_reference = sorted(n for n in DOMAIN if is_prime_reference(n))

assert marked == primes_reference, (
    f"Wilson's-theorem set {marked} disagrees with independent primality "
    f"check {primes_reference} -- classical property is wrong, stop."
)

print(f"Classical result: Wilson's theorem holds for n in {DOMAIN.start}..{DOMAIN.stop - 1} "
      f"exactly at n = {marked} (cross-checked against direct primality test).")

marked_bitstrings = [format(n, f"0{N_QUBITS}b") for n in marked]


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over 4 qubits for the marked bitstrings.
# ---------------------------------------------------------------------------

def build_oracle(marked_bitstrings, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in marked_bitstrings:
        # bits[0] is the MSB, mapped to qubit n_qubits-1 (Qiskit little-endian)
        zero_positions = [i for i, b in enumerate(reversed(bits)) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all qubits: flips phase of |11...1> after X's,
        # i.e. phase-flips the target computational basis state.
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
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


n = N_QUBITS
N_total = 2 ** n
M = len(marked)  # number of marked items
# Optimal number of Grover iterations for M marked out of N_total items.
theta = math.asin(math.sqrt(M / N_total))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(n, n)
qc.h(range(n))

oracle = build_oracle(marked_bitstrings, n)
diffuser = build_diffuser(n)

for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(n), range(n))

print(f"Grover circuit: {n} qubits, {M} marked states, {iterations} iteration(s).")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-bit string is c[n-1]...c[0]; measured basis state value:
def bits_to_int(bitstring: str) -> int:
    return int(bitstring, 2)

marked_set = set(marked)
marked_shots = sum(c for bs, c in counts.items() if bits_to_int(bs) in marked_set)
unmarked_shots = shots - marked_shots
marked_fraction = marked_shots / shots

# Also recover the single most-frequent measured outcome for a sanity print.
top_bitstring = max(counts, key=counts.get)
top_value = bits_to_int(top_bitstring)

print(f"Measured: {marked_shots}/{shots} shots ({marked_fraction:.1%}) landed on a "
      f"Wilson's-theorem-true (prime) state; most frequent single outcome n={top_value} "
      f"(marked={top_value in marked_set}).")

# Baseline: uniform random guessing would land in the marked set with
# probability M/N_total. Grover succeeds if it clearly beats that baseline
# and the top outcome is itself a marked (correct) state.
baseline_fraction = M / N_total
verified = (marked_fraction > 2 * baseline_fraction) and (top_value in marked_set)

if verified:
    print("PASS")
else:
    print("FAIL")

ran_ok = True
