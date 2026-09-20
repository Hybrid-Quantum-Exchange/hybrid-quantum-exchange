"""
Erdos problem #405 (erdosproblems.com / manman4/erdosproblems data/problems.yaml,
number: "405", tags: ["number theory", "factorials"]).

The problem's yaml entry lists oeis: ["N/A"] -- no OEIS sequence id is attached
to this problem in the source data. Per the task instructions, when there is no
OEIS id we still build our best honest attempt at a genuinely quantum-testable,
small, finite, computable property drawn from the problem's own tags (number
theory + factorials), rather than fabricating or borrowing an unrelated OEIS
value.

Chosen property (Wilson's theorem, a classical fact about factorials that is
finite and exactly checkable for small instances):

    For integer m with 1 < m <= 16, m is PRIME  <=>  (m-1)! mod m == m - 1.

We search the instance space m in {1, ..., 16} (encoded as a 4-qubit index
n = m-1 in {0, ..., 15}) with Grover's algorithm, using an oracle built from
the classical truth table of the Wilson predicate above (computed in this
script from first principles, via exact integer factorials -- no external
data, no hard-coded OEIS terms). The circuit is a genuine amplitude-amplification
search: the oracle phase-flips exactly the marked basis states, the diffuser
inverts about the mean, and the correct number of Grover iterations is derived
from the true number of marked items.

Classical ground truth for m in 1..16: primes among {2,...,16} are
{2, 3, 5, 7, 11, 13} (6 primes), so 6 of the 16 basis states (n = m-1 in
{1, 2, 4, 6, 10, 12}) are marked.

The script runs the Grover circuit on the ideal AerSimulator, takes the most
frequently measured 4-qubit strings, and checks that this measured set exactly
equals the classically computed marked set. It prints PASS or FAIL.
"""

from math import factorial, pi, floor, sqrt, asin

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed here, from first principles).
# ---------------------------------------------------------------------------

N_QUBITS = 4
M_MAX = 16  # m ranges over 1..16, n = m-1 ranges over 0..15 (4 qubits)


def is_prime_by_wilson(m: int) -> bool:
    """Wilson's theorem: m > 1 is prime iff (m-1)! % m == m - 1."""
    if m <= 1:
        return False
    return factorial(m - 1) % m == m - 1


marked_n = []  # values of n = m-1 for which m is prime
for m in range(1, M_MAX + 1):
    n = m - 1
    if is_prime_by_wilson(m):
        marked_n.append(n)

marked_n = sorted(marked_n)
expected_primes = [n + 1 for n in marked_n]
print(f"Classical check (Wilson's theorem) over m=1..{M_MAX}:")
print(f"  primes found: {expected_primes}")
print(f"  marked basis states (n=m-1): {marked_n}")

# Sanity cross-check with a direct primality test (independent of Wilson's
# theorem), to make sure the oracle set is genuinely correct.
def is_prime_direct(m: int) -> bool:
    if m < 2:
        return False
    for d in range(2, int(m**0.5) + 1):
        if m % d == 0:
            return False
    return True


direct_primes = [m for m in range(1, M_MAX + 1) if is_prime_direct(m)]
assert direct_primes == expected_primes, "Wilson's theorem check disagrees with direct primality test"
print("  cross-checked against direct primality test: OK")

M = len(marked_n)
assert M > 0

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classical marked set.
# ---------------------------------------------------------------------------


def bits_of(n: int, width: int):
    return [(n >> i) & 1 for i in range(width)]


def add_oracle(qc: QuantumCircuit, qubits, marked_values, width):
    """Phase-flip every basis state whose integer value is in marked_values."""
    for val in marked_values:
        bits = bits_of(val, width)
        # Flip qubits that should be 0 so the target pattern becomes all-1s.
        for q, b in zip(qubits, bits):
            if b == 0:
                qc.x(q)
        # Multi-controlled Z on all qubits (phase flip |11...1>).
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q, b in zip(qubits, bits):
            if b == 0:
                qc.x(q)


def add_diffuser(qc: QuantumCircuit, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


# Optimal number of Grover iterations for N=16 states, M marked items.
N_STATES = 2 ** N_QUBITS
theta = asin(sqrt(M / N_STATES))
num_iterations = max(1, round((pi / (4 * theta)) - 0.5))
print(f"N={N_STATES} states, M={M} marked, using {num_iterations} Grover iteration(s)")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))

qc.h(qubits)
for _ in range(num_iterations):
    add_oracle(qc, qubits, marked_n, N_QUBITS)
    add_diffuser(qc, qubits)
qc.measure(qubits, qubits)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings as c[width-1]...c[0] left to right; since our
# qubit i is classical bit i with weight 2**i, the printed bitstring is
# already the standard binary representation of n -- no reversal needed.
def counts_to_int_counts(counts):
    out = {}
    for bitstr, c in counts.items():
        n_val = int(bitstr, 2)
        out[n_val] = out.get(n_val, 0) + c
    return out


int_counts = counts_to_int_counts(counts)
sorted_by_count = sorted(int_counts.items(), key=lambda kv: -kv[1])
print("Top measured n values (n, count):", sorted_by_count[:M + 3])

# Take the M most frequent outcomes as Grover's answer.
measured_top_n = sorted({n for n, _ in sorted_by_count[:M]})

print(f"Grover measured marked set (n=m-1): {measured_top_n}")
print(f"Classical expected marked set (n=m-1): {marked_n}")

ok = measured_top_n == marked_n

if ok:
    print("PASS")
else:
    print("FAIL")
