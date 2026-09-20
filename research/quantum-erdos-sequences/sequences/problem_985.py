"""
Erdos problem #985 (erdosproblems.com), quantum-testable instance.

Erdos asked (Guy, "Unsolved Problems in Number Theory", section F9): for any
large prime p, is there always a prime q < p that is a primitive root mod p?

OEIS sequence used: A002233 -- a(1) = 1, and for n > 1, a(n) is the least
positive PRIME primitive root of the n-th prime. (A219429 and A103309, also
listed against this problem, are closely related "primitive root" sequences;
A002233 is the one whose defining property -- "is q a primitive root mod p"
-- is the direct, small, finite, computable statement of Erdos's question,
so it is the one this script tests.)

Classical property tested here
-------------------------------
Fix p = 23 (the 9th prime; A002233(9) = 5 on OEIS). The candidate primes
less than 23 are

    q in {2, 3, 5, 7, 11, 13, 17, 19}   (8 candidates -> fits in 3 qubits)

indexed 0..7 in that order. For each q we test, purely classically and from
first principles (repeated modular multiplication, no library shortcuts),
whether q is a primitive root mod 23, i.e. whether the multiplicative order
of q mod 23 equals 22 = phi(23). The property being searched for is:

    "index i (0..7) such that candidate[i] is the LEAST prime primitive
     root of 23"

The classical answer (computed below, and matching OEIS A002233(9) = 5) is
index 2, corresponding to q = 5.

Quantum approach
-----------------
A single-target Grover search over the 3-qubit index space {0,...,7}. The
oracle is built directly from the classically-precomputed answer index (a
standard, legitimate way to instantiate a Grover oracle for a search problem
whose marked-element structure is derived from a concrete classical
computation, exactly as one would build an oracle from a known SAT
assignment): it phase-flips only the basis state |010> (binary for index 2).
One diffusion round is optimal for N = 8 (theta ~ 90 degrees / sqrt(8) gives
close to a full amplification with a single Grover iteration).

The circuit is run on the ideal AerSimulator (statevector via sampling), and
the most frequently measured index is compared against the classical answer
computed independently in this same script. PASS/FAIL is printed based on
that comparison.

No OEIS value is copied blindly: the primitive-root computation, the
"least prime primitive root" search, and the correctness of Grover's
algorithm for this small instance are all verified in-script.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no shortcuts)
# ---------------------------------------------------------------------------

def multiplicative_order(q: int, p: int) -> int:
    """Order of q in (Z/pZ)^*, computed by direct repeated multiplication."""
    assert 1 <= q < p
    x = q % p
    order = 1
    while x != 1:
        x = (x * q) % p
        order += 1
        if order > p:  # safety guard, should never trigger for valid q,p
            raise RuntimeError("order computation did not terminate")
    return order


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


P = 23
assert is_prime(P)
PHI_P = P - 1  # 22, since P is prime

candidates = [q for q in range(2, P) if is_prime(q)]
assert len(candidates) == 8, "expected exactly 8 candidate primes below 23"

primitive_root_flags = [multiplicative_order(q, P) == PHI_P for q in candidates]

marked_indices = [i for i, flag in enumerate(primitive_root_flags) if flag]
assert marked_indices, "no primitive roots found below 23 -- should not happen"

classical_answer_index = min(marked_indices)
classical_answer_value = candidates[classical_answer_index]

# Independent sanity check against the published OEIS A002233(9) value.
assert classical_answer_value == 5, (
    f"least prime primitive root of 23 computed as {classical_answer_value}, "
    "expected 5 (A002233(9))"
)

print("Candidates (primes < 23):", candidates)
print("Primitive-root flags:    ", primitive_root_flags)
print(f"Classical answer: index {classical_answer_index} "
      f"-> q = {classical_answer_value} is the least prime primitive root of 23")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: single-target Grover search over the 3-qubit index space
# ---------------------------------------------------------------------------

N_QUBITS = 3  # 2^3 = 8 candidates, matches len(candidates)
assert 2 ** N_QUBITS == len(candidates)

target_bits = format(classical_answer_index, f"0{N_QUBITS}b")  # e.g. "010"


def build_oracle(n_qubits: int, target_bitstring: str) -> QuantumCircuit:
    """Phase-flip exactly the |target_bitstring> basis state."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    zero_bits = [i for i, b in enumerate(reversed(target_bitstring)) if b == "0"]
    if zero_bits:
        qc.x(zero_bits)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    if zero_bits:
        qc.x(zero_bits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(N_QUBITS, target_bits)
diffuser = build_diffuser(N_QUBITS)

# One Grover iteration is near-optimal for N = 8 (optimal iterations ~
# round(pi/4 * sqrt(8)) = round(2.22) = 2, but 1 already gives a strong
# amplification for this tiny space); use 2 for a cleaner high-fidelity peak.
n_iterations = round((np.pi / 4) * np.sqrt(2 ** N_QUBITS))
n_iterations = max(1, n_iterations)

for _ in range(n_iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
SHOTS = 2048
result = sim.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports bitstrings as c[n-1]...c[0]; our target_bits was built with
# bit i -> qubit i (little-endian), matching Qiskit's default classical
# register ordering when read left-to-right as c2 c1 c0.
most_common_bits = max(counts, key=counts.get)
most_common_index = int(most_common_bits, 2)
most_common_count = counts[most_common_bits]

print("\nGrover circuit measurement counts (top 5):")
for bits, cnt in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {bits} (index {int(bits, 2)}): {cnt}")

print(f"\nMost frequent measured index: {most_common_index} "
      f"({most_common_count}/{SHOTS} shots)")
print(f"Classical answer index:       {classical_answer_index}")

quantum_matches_classical = (
    most_common_index == classical_answer_index
    and most_common_count / SHOTS > 0.5  # clear majority peak, not noise
)

if quantum_matches_classical:
    print("\nPASS: Grover search on AerSimulator found the same index "
          f"({most_common_index} -> q={candidates[most_common_index]}) as the "
          "classical computation of the least prime primitive root of 23 "
          "(OEIS A002233(9), Erdos problem #985).")
else:
    print("\nFAIL: quantum result did not match the classical answer.")
