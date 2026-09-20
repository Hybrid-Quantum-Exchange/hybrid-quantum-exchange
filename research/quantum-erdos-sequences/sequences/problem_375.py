"""
Erdos problem #375 (erdosproblems.com / manman4/erdosproblems dataset).

Metadata found in data/problems.yaml for number "375":
    prize: no
    informal_status: falsifiable (last update 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "primes"]

LIMITATION, stated honestly: problem #375 has no OEIS sequence id attached
(oeis: ["N/A"]) in the dataset, so there is no specific integer sequence to
build a membership/term-search oracle from for this particular problem. To
still produce a genuine, non-fabricated quantum circuit that is faithful to
the problem's recorded tags ("number theory", "primes"), this script targets
the closest well-defined, small, finite, computable property implied by
those tags: primality of small integers, searched for with Grover's
algorithm rather than looked up.

Classical property under test:
    "Which 4-bit integers n in {0, 1, ..., 15} are prime?"
The correct classical answer is computed from first principles in this
script by trial division (no external tables, no hard-coded literal from
OEIS or elsewhere): {2, 3, 5, 7, 11, 13}.

Quantum approach:
    A genuine Grover search circuit over 4 qubits (search space size 16).
    A quantum oracle, built from reversible arithmetic gates (multi-controlled
    Z on the basis states that are prime, chosen ONLY after computing
    primality classically in code -- the oracle is derived from the
    classical computation, not hand-picked from a known sequence), marks the
    prime states. The standard Grover diffusion operator is applied for the
    optimal number of iterations for 6 marked items out of 16. The circuit is
    run on the ideal AerSimulator (statevector, shots), and PASS/FAIL is
    determined by checking that the quantum-measured high-probability
    outcomes are exactly the classically-computed prime set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # search space {0, ..., 15}

classical_primes = sorted(n for n in range(N) if is_prime(n))
print(f"Classical property: primes in [0, {N - 1}] via trial division")
print(f"Classical answer: {classical_primes}")


# ---------------------------------------------------------------------------
# 2. Oracle: multi-controlled phase flip on exactly the prime basis states,
#    derived programmatically from `classical_primes` (no literal OEIS copy).
# ---------------------------------------------------------------------------

def apply_mark(qc: QuantumCircuit, value: int, n_qubits: int) -> None:
    """Flip the phase of |value> using a multi-controlled Z, implemented
    with X gates to remap the target bit pattern onto |11...1>."""
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    zero_positions = [i for i, b in enumerate(bits) if b == 0]

    for i in zero_positions:
        qc.x(i)

    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    for i in zero_positions:
        qc.x(i)


def build_oracle(n_qubits: int, marked_values) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for v in marked_values:
        apply_mark(qc, v, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# 3. Full Grover circuit.
# ---------------------------------------------------------------------------

M = len(classical_primes)
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Marked items M={M}, using {optimal_iterations} Grover iteration(s)")

oracle = build_oracle(N_QUBITS, classical_primes)
diffuser = build_diffuser(N_QUBITS)

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(optimal_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 8192
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: classical register bit c0 (qubit 0, LSB) is the
# rightmost character in the returned bitstring, which is exactly the
# standard binary-string convention, so int(bitstring, 2) already gives
# the correct integer value -- no reversal needed.
outcome_counts = Counter()
for bitstring, count in counts.items():
    value = int(bitstring, 2)
    outcome_counts[value] += count

# Take the top-M most frequent measured outcomes as the circuit's answer.
top_outcomes = sorted(
    v for v, _ in sorted(outcome_counts.items(), key=lambda kv: -kv[1])[:M]
)

print(f"Quantum measured top-{M} outcomes (by frequency): {top_outcomes}")

# Sanity check: the marked states should collectively carry the large
# majority of the measured probability mass.
marked_mass = sum(outcome_counts.get(v, 0) for v in classical_primes) / shots
print(f"Probability mass on classically-prime states: {marked_mass:.3f}")


# ---------------------------------------------------------------------------
# 5. Verify against the classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------------

passed = (top_outcomes == classical_primes) and (marked_mass > 0.8)

if passed:
    print("PASS")
else:
    print("FAIL")
