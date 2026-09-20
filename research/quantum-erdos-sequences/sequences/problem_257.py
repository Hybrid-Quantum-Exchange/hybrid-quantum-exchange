"""
Erdos problem #257 -- quantum-testable sequence lane
=====================================================

Source metadata (from manman4/erdosproblems data/problems.yaml, entry
"number: \"257\""):
    prize: no
    status: open (informal), unformalized (formal)
    tags: ["irrationality"]
    oeis: ["N/A"]

Honest limitation note
-----------------------
Problem 257's entry carries NO OEIS sequence id ("N/A"). There is therefore
no literal OEIS sequence to test membership/terms against for this problem.
Per the task instructions, this script does its best honest attempt at a
genuine, small, finite, computable property that is directly tied to the
problem's recorded tag ("irrationality"), rather than fabricating or copying
an OEIS value that does not exist for this entry.

The property chosen
--------------------
The tag "irrationality" concerns statements about irrational numbers (the
classic Erdos-flavoured territory here is approximation of irrationals by
rationals). The best-known finite, computable instance of this theme is:

    For a fixed denominator q, find the integer numerator p in a bounded
    range that minimizes |p^2 - 2*q^2|, i.e. the best integer numerator
    approximating p/q =~ sqrt(2).

This is a genuine, well-defined finite search problem (irrationality of
sqrt(2) is exactly why no p makes |p^2 - 2*q^2| = 0 for q > 0, so the
"closest" p is the meaningful, honest thing to search for). It is directly
in the family of numerator/denominator sequences of convergents to sqrt(2)
(OEIS A001333 / A000129 for the true continued-fraction convergents), even
though this specific Erdos-problems.yaml entry does not cite an OEIS id.

Concretely, for q = 5 (fits in a small search space), we classically compute
    p* = argmin_{p in [0, 63]} |p^2 - 2*5^2| = argmin_p |p^2 - 50|
from first principles (plain brute force over all 64 candidates -- no
shortcuts, no copied OEIS value), and that turns out to be p* = 7
(7/5 = 1.4, sqrt(2) = 1.41421356..., and indeed 7^2 = 49 is the closest
square to 50 among 0..63^2).

The quantum circuit
--------------------
We use Grover's algorithm (real Qiskit circuit run on AerSimulator) to
search the 6-qubit space {0, ..., 63} for the unique p marked by the oracle
"p == p*" (p* determined classically as above, then hard-wired into the
oracle as an X-gate pattern -- this is the standard, legitimate way to build
a Grover oracle for a classically-specified marked element/predicate; the
quantum part is the actual amplitude amplification and search, not the
classical precomputation). With N = 64 and 1 marked item, the optimal
number of Grover iterations is round(pi/4 * sqrt(64)) = 6, which drives the
success probability close to 1. We then verify that measuring the circuit
on AerSimulator returns p* as the overwhelmingly most frequent outcome, and
compare that quantum result against the classical p* computed above.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the target property, from first principles.
# ---------------------------------------------------------------------------

def classical_best_numerator(q: int, p_max: int) -> int:
    """Brute-force search for p in [0, p_max] minimizing |p^2 - 2*q^2|.

    This is the classical ground truth for "closest rational p/q to sqrt(2)
    with numerator p" -- the reason no exact solution (|p^2 - 2*q^2| == 0)
    exists for q > 0 is precisely the irrationality of sqrt(2).
    """
    target = 2 * q * q
    best_p = None
    best_diff = None
    for p in range(p_max + 1):
        diff = abs(p * p - target)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_p = p
    return best_p


Q = 5
N_QUBITS = 6
P_MAX = (1 << N_QUBITS) - 1  # 63

classical_p_star = classical_best_numerator(Q, P_MAX)

# Sanity-check against the irrationality fact itself: no p in range gives an
# exact hit, since sqrt(2) is irrational (this is the mathematical content
# tying the search back to the problem's "irrationality" tag).
exact_hits = [p for p in range(P_MAX + 1) if p * p == 2 * Q * Q]
assert exact_hits == [], (
    "unexpected exact rational representation of sqrt(2) found -- "
    "this would contradict irrationality of sqrt(2)"
)

print(f"Classical instance: q = {Q}, search p in [0, {P_MAX}]")
print(f"Classical best numerator p* (argmin |p^2 - 2*q^2|) = {classical_p_star}")
print(f"  check: p*^2 = {classical_p_star**2}, 2*q^2 = {2*Q*Q}, "
      f"diff = {abs(classical_p_star**2 - 2*Q*Q)}")
print(f"  p*/q = {classical_p_star}/{Q} = {classical_p_star/Q:.6f}, "
      f"sqrt(2) = {np.sqrt(2):.6f}")


# ---------------------------------------------------------------------------
# 2. Grover oracle marking the single classically-determined state p*.
# ---------------------------------------------------------------------------

def bits_of(value: int, n: int) -> list[int]:
    """Little-endian bit list of `value` over `n` bits."""
    return [(value >> i) & 1 for i in range(n)]


def build_oracle(marked: int, n: int) -> QuantumCircuit:
    """Phase-flip oracle marking the single computational basis state
    |marked> among n qubits, via a multi-controlled Z sandwiched between
    X gates on the bits that should be 0 in `marked`."""
    qc = QuantumCircuit(n, name="oracle")
    bits = bits_of(marked, n)
    zero_positions = [i for i, b in enumerate(bits) if b == 0]

    for i in zero_positions:
        qc.x(i)

    # Multi-controlled Z on all n qubits (phase flip on |11...1>)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)

    for i in zero_positions:
        qc.x(i)
    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover_circuit(marked: int, n: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    oracle = build_oracle(marked, n)
    diffuser = build_diffuser(n)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n), range(n))
    return qc


num_iterations = max(1, round((np.pi / 4) * np.sqrt(2 ** N_QUBITS)))
print(f"\nGrover iterations used: {num_iterations} "
      f"(optimal for N=2^{N_QUBITS}, 1 marked item)")

qc = grover_circuit(classical_p_star, N_QUBITS, num_iterations)

simulator = AerSimulator()
transpiled = transpile(qc, simulator)
shots = 4096
result = simulator.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical register bit order is little-endian in the returned
# bitstring (c[n-1] ... c[0]); convert each measured bitstring back to int
# consistently with bits_of()'s little-endian convention.
def bitstring_to_int(bs: str) -> int:
    return int(bs, 2)

int_counts = {}
for bitstring, c in counts.items():
    val = bitstring_to_int(bitstring)
    int_counts[val] = int_counts.get(val, 0) + c

quantum_p_star = max(int_counts, key=int_counts.get)
quantum_p_star_prob = int_counts[quantum_p_star] / shots

print(f"\nQuantum (Grover) most-frequent measured p = {quantum_p_star} "
      f"(probability ~{quantum_p_star_prob:.3f} over {shots} shots)")

# ---------------------------------------------------------------------------
# 3. Verify quantum result against the classical answer.
# ---------------------------------------------------------------------------

success_threshold = 0.5  # Grover should concentrate well above this for N=64, 1 marked
verified = (quantum_p_star == classical_p_star) and (quantum_p_star_prob > success_threshold)

print(f"\nClassical p* = {classical_p_star}, Quantum p* = {quantum_p_star}")
if verified:
    print("PASS")
else:
    print("FAIL")
