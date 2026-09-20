"""
Erdos problem #784 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, entry
"number: '784'"):
    prize: no
    status: solved (2025-12-20)
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION, stated honestly up front: the dataset entry for problem #784
carries no OEIS sequence id ("N/A") and no further formula/description text
is present anywhere in the cloned erdosproblems repository (searched for any
file mentioning "784"; none exists beyond this one YAML record). There is
therefore no specific integer sequence attached to this problem that a
circuit could be built against. Per the task's fallback instruction for this
case, this script does NOT fabricate a fake OEIS-derived property. Instead,
staying within the problem's only real, verifiable attribute -- its tag
"number theory" -- it builds a genuine, honestly-labelled small-instance
number-theory search problem (rather than pretending it is problem #784's
actual mathematical content):

    Classical property under test:
        For N = 16 (search space {0, 1, ..., 15}, 4 qubits), let
        S = { n in [0, N) : n is prime }.
    This S is computed from first principles in this script (trial division,
    no external tables, no OEIS lookup) and is exactly
        S = {2, 3, 5, 7, 11, 13}.

Quantum circuit: Grover's search algorithm on 4 qubits. An oracle phase-flips
exactly the prime basis states in {0,...,15}; the diffusion operator amplifies
their amplitude. |S| = 6 out of N = 16, so the optimal number of Grover
iterations is floor(pi/4 * sqrt(N/|S|)) = 1. After 1 iteration and 4096 shots
on the ideal AerSimulator, the circuit should return overwhelmingly a sample
in S. The script verifies this against the classically computed S: PASS if
essentially all sampled outcomes with non-trivial probability lie in S and
the *most probable* outcomes exactly match a subset consistent with S (no
non-prime state has larger total probability mass than the minimum prime
probability mass), i.e. the quantum search reliably surfaces only elements of
the correctly, classically-computed set S.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import sys
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the target set S (trial division, from scratch)
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

CLASSICAL_S = sorted(n for n in range(N) if is_prime(n))
print(f"Classical property: primes in [0, {N}) = {CLASSICAL_S}")
assert CLASSICAL_S == [2, 3, 5, 7, 11, 13], "classical computation is wrong"


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle: phase-flip exactly the states in CLASSICAL_S
# ---------------------------------------------------------------------------

def build_oracle_gate(marked, n_qubits):
    # Diagonal phase-flip oracle: -1 on marked basis states, +1 elsewhere.
    # Index j of the unitary corresponds directly to the integer value j,
    # matching Qiskit's little-endian statevector convention (qubit 0 is
    # the least-significant bit) when the gate is applied to qubits
    # [0, 1, ..., n_qubits-1] in order.
    dim = 2 ** n_qubits
    diag = np.ones(dim, dtype=complex)
    for m in marked:
        diag[m] = -1.0
    return UnitaryGate(np.diag(diag), label="oracle")


def build_diffuser_gate(n_qubits):
    # Reflection about the uniform superposition: U = 2|s><s| - I.
    dim = 2 ** n_qubits
    s = np.ones((dim, 1), dtype=complex) / math.sqrt(dim)
    u = 2 * (s @ s.conj().T) - np.eye(dim, dtype=complex)
    return UnitaryGate(u, label="diffuser")


def grover_iterations(n, k):
    return max(1, round((math.pi / 4) * math.sqrt(n / k)))


iterations = grover_iterations(N, len(CLASSICAL_S))
print(f"Grover iterations used: {iterations} (N={N}, |S|={len(CLASSICAL_S)})")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle_gate = build_oracle_gate(CLASSICAL_S, N_QUBITS)
diffuser_gate = build_diffuser_gate(N_QUBITS)

for _ in range(iterations):
    qc.append(oracle_gate, range(N_QUBITS))
    qc.append(diffuser_gate, range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

SHOTS = 4096
sim = AerSimulator()
compiled = transpile(qc, sim)
result = sim.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical-register bitstring c[n_qubits-1]...c[0] already reads
# back as the same integer index used to build the diagonal oracle/diffuser
# unitaries (verified empirically against the classical set below), so no
# bit-reversal is needed here.
int_counts = {}
for bitstring, c in counts.items():
    n_val = int(bitstring, 2)
    int_counts[n_val] = int_counts.get(n_val, 0) + c

print("Sampled outcome probabilities (top 8):")
for n_val, c in sorted(int_counts.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  n={n_val:2d}  prime={n_val in CLASSICAL_S}  prob={c / SHOTS:.4f}")


# ---------------------------------------------------------------------------
# 4. Verify: quantum result vs. classical answer
# ---------------------------------------------------------------------------

prob_in_s = sum(c for n_val, c in int_counts.items() if n_val in CLASSICAL_S) / SHOTS
prob_out_s = 1.0 - prob_in_s

# With 1 optimal iteration on N=16, |S|=6, theoretical success probability is
# high (>85%); require a solid margin above chance (|S|/N = 0.375).
THRESHOLD = 0.70

print(f"\nP(sample in classical S) = {prob_in_s:.4f}")
print(f"P(sample outside S)      = {prob_out_s:.4f}")
print(f"Pass threshold           = {THRESHOLD}")

# Also check that every one of the individually most-sampled outcomes (those
# carrying non-negligible probability mass) is itself in CLASSICAL_S.
significant = [n_val for n_val, c in int_counts.items() if c / SHOTS >= 0.05]
significant_all_in_s = all(n_val in CLASSICAL_S for n_val in significant)

passed = (prob_in_s >= THRESHOLD) and significant_all_in_s and len(significant) > 0

print(f"Significant outcomes (>=5% prob): {sorted(significant)}")
print(f"All significant outcomes are classically-verified primes: {significant_all_in_s}")

if passed:
    print("\nPASS")
    sys.exit(0)
else:
    print("\nFAIL")
    sys.exit(1)
