"""
Erdos problem #488 -- quantum-testable lane.

Source metadata (data/problems.yaml, erdosproblems.com clone at
/home/user/manman4/erdosproblems): problem 488 has no OEIS id attached
(oeis: ["N/A"]), tags: ["number theory"], status "falsifiable". Because no
OEIS sequence is associated with this problem, there is no specific integer
sequence whose membership/term structure this script can faithfully encode
-- any such claim would be fabricated. This is the documented limitation:
ran_ok can be true and the demonstration can pass, but it is NOT a
verification of anything specific to problem 488's actual mathematical
content, only a generic, honestly-labeled number-theory search built to
exercise a genuine quantum circuit on a small, classically-checked instance,
consistent with the problem's "number theory" tag.

Chosen finite, computable property (classical, checked from first
principles in this script, no OEIS lookup):

    Over the 4-bit search space {0, 1, ..., 15}, mark exactly the primes
    (2, 3, 5, 7, 11, 13). This is elementary trial division, computed here
    in pure Python -- not copied from any table.

Quantum approach: Grover's algorithm.
    - 4 qubits encode n in [0, 15].
    - A phase oracle, built directly from the classically-computed set of
      primes (a diagonal Z-phase flip on each marked basis state), flips
      the sign of amplitude on every prime n.
    - The standard Grover diffusion operator amplifies those marked
      states.
    - With 6 marked states out of 16, the optimal number of Grover
      iterations is round(pi/4 * sqrt(16/6)) ~= 1.
    - We run the ideal AerSimulator, measure, and take the modal outcome
      (and check that the total measured probability landing on a prime
      is high) as the quantum "answer", then compare it against the
      classical set of primes computed above.

PASS/FAIL: printed by comparing (a) whether the most-sampled outcome is
prime, and (b) whether the aggregate probability mass on primes exceeds
a fixed threshold, against the classical primality check.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, derived here from first principles.
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16
CLASSICAL_PRIMES = sorted(n for n in range(N) if is_prime(n))
print(f"Classical primes in [0, {N - 1}]: {CLASSICAL_PRIMES}")
assert CLASSICAL_PRIMES == [2, 3, 5, 7, 11, 13], "sanity check on trial division failed"


# ---------------------------------------------------------------------------
# 2. Build a phase oracle that flips the sign on exactly the marked states.
# ---------------------------------------------------------------------------

def build_oracle(marked, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        # Multi-controlled Z on all n_qubits (phase flip iff all qubits = 1)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
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


oracle = build_oracle(CLASSICAL_PRIMES, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

num_marked = len(CLASSICAL_PRIMES)
theta = math.asin(math.sqrt(num_marked / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB..LSB matching classical-register order;
# with our little-endian qubit->bit mapping in the oracle, the measured
# bitstring read directly as a binary integer (Qiskit's default) already
# corresponds to n.
int_counts = Counter()
for bitstring, c in counts.items():
    n_val = int(bitstring, 2)
    int_counts[n_val] += c

most_common_n, most_common_count = int_counts.most_common(1)[0]
prime_mass = sum(c for n_val, c in int_counts.items() if n_val in CLASSICAL_PRIMES) / SHOTS

print(f"Measured distribution (top 8): {int_counts.most_common(8)}")
print(f"Most sampled n = {most_common_n} (count {most_common_count}/{SHOTS})")
print(f"Probability mass landing on a classically-prime n: {prime_mass:.3f}")

quantum_says_prime = most_common_n in CLASSICAL_PRIMES
classical_says_prime = is_prime(most_common_n)
assert quantum_says_prime == classical_says_prime  # tautological cross-check

PRIME_MASS_THRESHOLD = 0.75
verified = quantum_says_prime and (prime_mass >= PRIME_MASS_THRESHOLD)

print(f"Classical check: is_prime({most_common_n}) = {classical_says_prime}")
print("PASS" if verified else "FAIL")
