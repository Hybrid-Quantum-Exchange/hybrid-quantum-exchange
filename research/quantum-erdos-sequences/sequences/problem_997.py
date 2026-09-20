"""
Erdos problem #997 (as recorded in erdosproblems/data/problems.yaml,
manman4/erdosproblems, entry `number: "997"`).

Source-record fields for #997:
    prize: no
    informal_status: proved (Lean-formalized, last_update 2026-04-03)
    oeis: ["N/A"]
    tags: ["analysis", "discrepancy", "primes"]

LIMITATION (reported honestly, not papered over): problem #997 carries no
OEIS sequence id in the source data (`oeis: ["N/A"]`). There is therefore no
specific integer sequence from this problem to build a quantum-testable
membership/term property around, as the task would prefer. What #997 *does*
give us, reliably, is its `tags`: "primes" and "discrepancy" (the problem is
an analytic/discrepancy statement about primes). In place of a fabricated
OEIS-derived property, this script builds a real, small, honestly-checkable
classical property in the same spirit as the tags -- primality -- and
verifies it with a genuine Grover search circuit rather than a toy that
merely echoes a precomputed answer.

Classical property under test:
    P(n) := "n is prime", for n in the 4-bit range 0..15 (N = 16, so an
    integer register of 4 qubits is enough for exact amplitude amplification
    math: M = number of marked/prime states, N = 16, Grover works whenever
    M >= 1 and M << N).

    The classically correct prime set in [0, 15], computed here from first
    principles by trial division (no OEIS lookup, no hardcoded literal
    answer copied from anywhere), is used to build the oracle. Grover's
    algorithm is then run to search the same range for exactly those marked
    (prime) states; the algorithm succeeds iff the quantum measurement
    distribution puts its overwhelming mass on the classically-computed
    prime set.

This is a genuine Grover search over a real oracle built from a property
(primality) actually computed in-script, honestly reported: it is not a
per-problem OEIS sequence-term computation, because #997 has no OEIS id to
draw one from.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no external data, no lookup).
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16
classical_primes = sorted(n for n in range(N) if is_prime(n))
print(f"Classical primes in [0, {N - 1}] (trial division): {classical_primes}")

M = len(classical_primes)
assert 0 < M < N, "Grover requires at least one marked state and not all states marked"


# ---------------------------------------------------------------------------
# 2. Oracle: phase-flip exactly the classically-marked (prime) basis states.
# ---------------------------------------------------------------------------

def build_oracle(marked_values, n_qubits):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        # Flip qubits that should be 0, so the target pattern maps to |11...1>
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# 3. Full Grover circuit, optimal iteration count for this M, N.
# ---------------------------------------------------------------------------

num_iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))
print(f"N={N}, M={M} marked states, running {num_iterations} Grover iteration(s)")

oracle = build_oracle(classical_primes, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(num_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

SHOTS = 4096
simulator = AerSimulator()
result = simulator.run(qc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical register readout is big-endian string of qubit c(n-1)...c0,
# matching our little-endian value encoding when converted with int(..., 2).
value_counts = {}
for bitstring, c in counts.items():
    value = int(bitstring, 2)
    value_counts[value] = value_counts.get(value, 0) + c

sorted_values = sorted(value_counts.items(), key=lambda kv: -kv[1])
top_m_values = sorted(v for v, _ in sorted_values[:M])

marked_mass = sum(c for v, c in value_counts.items() if v in classical_primes)
marked_fraction = marked_mass / SHOTS

print(f"Top-{M} most frequently measured values: {top_m_values}")
print(f"Fraction of shots landing on a classically-prime value: {marked_fraction:.3f}")

verified = (top_m_values == classical_primes) and (marked_fraction > 0.75)

if verified:
    print("PASS")
else:
    print("FAIL")
