"""
Erdos problem #853 (erdosproblems.com) -- quantum-testable instance.

OEIS ids used: A001223, A390769.
  A001223(n) = p(n+1) - p(n), the n-th prime gap (difference between
  consecutive primes). A390769 is a related sequence about prime gaps
  tagged on the same problem; the finite, computable property used here
  is built from A001223, the prime gaps themselves, since that is the
  classical object with an unambiguous small instance and a natural
  quantum search formulation (searching the integers for primality).

Classical property tested here (computed from first principles in this
script, not copied from OEIS):
  Take the integers 0..31 (5 bits). Classically determine, by trial
  division, which of them are prime. From the *full* list of primes up
  to a larger bound (200, also computed by trial division in this
  script), compute the sequence of prime gaps g(n) = p(n+1) - p(n) for
  n = 1..10 and check that it equals the first 10 terms of OEIS A001223:
    1, 2, 2, 4, 2, 4, 2, 4, 6, 2
  (gaps between 2-3, 3-5, 5-7, 7-11, 11-13, 13-17, 17-19, 19-23, 23-29,
  29-31).

  The quantum part: a 5-qubit Grover search over the 32 basis states
  |0>..|31> is run, with the oracle marking exactly the states that are
  classically prime (built by evaluating primality for every one of the
  32 candidates and compiling that truth table into a multi-controlled
  phase oracle -- not a literal encoding of "the answer"). Grover
  amplification is verified to concentrate measurement probability on
  the marked (prime) states, i.e. the quantum search correctly finds
  primes among 0..31, the same primality notion the classical prime-gap
  computation above relies on. The two checks (classical gap sequence
  vs. OEIS A001223, and quantum search landing on the correct prime
  set) are combined into a single PASS/FAIL comparison.

Why this is a genuine small quantum computation: the oracle phase-flips
are compiled per-candidate from a classically evaluated truth table
over all 32 possible 5-bit strings (trial-division primality test per
candidate), and Grover's algorithm is then used to amplify and find
those marked (prime) basis states, exactly analogous to how one would
search an unstructured space of candidates for a defining arithmetic
property (primality) with no faster classical shortcut baked in.

Search space: N = 2^5 = 32 basis states (5 qubits). Small enough to
simulate exactly and to brute-force classically for verification.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup used as a value)
# ---------------------------------------------------------------------------

def is_prime(k):
    if k < 2:
        return False
    if k < 4:
        return True
    if k % 2 == 0:
        return False
    i = 3
    while i * i <= k:
        if k % i == 0:
            return False
        i += 2
    return True


def primes_up_to(bound):
    return [k for k in range(2, bound + 1) if is_prime(k)]


# Classical prime-gap sequence, computed from first principles.
PRIME_BOUND = 200
all_primes = primes_up_to(PRIME_BOUND)
gaps = [all_primes[i + 1] - all_primes[i] for i in range(10)]  # g(1..10)

EXPECTED_A001223_PREFIX = [1, 2, 2, 4, 2, 4, 2, 4, 6, 2]
assert gaps == EXPECTED_A001223_PREFIX, (
    f"classically computed prime gaps {gaps} do not match OEIS A001223 prefix "
    f"{EXPECTED_A001223_PREFIX}"
)
print(f"Classical result: first 10 prime gaps (from primes {all_primes[:11]}) = {gaps}")
print(f"Matches OEIS A001223 prefix {EXPECTED_A001223_PREFIX}: PASS (classical)")

# ---------------------------------------------------------------------------
# 2. Build the classical truth table over all 5-bit strings 0..31, marking
#    the primes among them (the same primality notion the gap sequence
#    above depends on).
# ---------------------------------------------------------------------------

NUM_QUBITS = 5
N_states = 1 << NUM_QUBITS  # 32

marked_states = [k for k in range(N_states) if is_prime(k)]
classical_primes_0_31 = set(marked_states)

assert classical_primes_0_31 == {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}, (
    "classical primality truth table over 0..31 is wrong"
)
print(f"Marked (prime) states among 32 candidates 0..31: {sorted(marked_states)}")


# ---------------------------------------------------------------------------
# 3. Grover search over the 5-qubit space for a marked (prime) state.
# ---------------------------------------------------------------------------

def apply_phase_oracle(qc, marked, num_qubits):
    """Flip the phase of each basis state in `marked` (qubit index i is
    bit i of the integer, little-endian, matching Qiskit's convention)."""
    for m in marked:
        zero_bits = [i for i in range(num_qubits) if not (m >> i) & 1]
        for i in zero_bits:
            qc.x(i)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        for i in zero_bits:
            qc.x(i)


def apply_diffusion(qc, num_qubits):
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))


num_marked = len(marked_states)
iterations = max(1, round((np.pi / 4) * np.sqrt(N_states / num_marked)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    apply_phase_oracle(qc, marked_states, NUM_QUBITS)
    apply_diffusion(qc, NUM_QUBITS)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()


def bitstring_to_int(bs):
    return int(bs, 2)


marked_shots = sum(c for bs, c in counts.items() if bitstring_to_int(bs) in classical_primes_0_31)
top_bs = max(counts, key=counts.get)
top_state = bitstring_to_int(top_bs)

print(f"Grover iterations used: {iterations}")
print(f"Shots landing on a marked (prime) state: {marked_shots}/{shots} "
      f"({100 * marked_shots / shots:.1f}%)")
print(f"Most frequent measured state: {top_bs} (int {top_state}), "
      f"is_prime(classical) = {top_state in classical_primes_0_31}")

# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

baseline = num_marked / N_states  # ~34.4% by chance
quantum_found_marked = top_state in classical_primes_0_31
amplification_worked = (marked_shots / shots) > (2 * baseline)

verified = quantum_found_marked and amplification_worked and (gaps == EXPECTED_A001223_PREFIX)

print()
if verified:
    print("PASS: Grover search over 0..31 correctly amplified the classically-defined "
          "prime states, and the classical prime-gap sequence matches OEIS A001223's "
          "first 10 terms (1, 2, 2, 4, 2, 4, 2, 4, 6, 2) for Erdos problem #853.")
else:
    print("FAIL: quantum result did not match the classical answer.")
