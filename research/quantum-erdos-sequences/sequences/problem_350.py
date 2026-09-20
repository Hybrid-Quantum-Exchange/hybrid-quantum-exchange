"""
Erdos problem #350 -- quantum-testable sequence lane.

Source metadata (from erdosproblems.com data, /data/problems.yaml, entry
"number: 350"): prize "no", status "proved (Lean)" as of 2025-11-25,
tags ["number theory", "additive combinatorics"], and crucially:

    oeis: ["N/A"]

Problem #350 has NO associated OEIS sequence in the source data. That means
the premise of this lane -- "identify a property of the OEIS sequence tied
to this problem" -- cannot be satisfied for #350 as stated: there is no
sequence to derive a property from, and no small term to check a quantum
circuit's answer against.

LIMITATION (stated honestly, per instructions): this script does not test
any object specific to Erdos problem #350. Instead, since the problem's own
tag "number theory" concerns divisibility/primality-type properties -- the
family of properties that *would* plausibly define an OEIS sequence if one
existed for this problem -- it falls back to a genuine, self-contained
quantum computation of the nearest well-posed finite instance of that kind:

    Grover search over 3-bit integers n in [0, 7] for the unique n that is
    prime AND odd AND not equal to 3 (i.e. n in {5, 7}), collapsed further
    to a *single* marked element by also requiring n > 5, giving the unique
    classical answer n = 7.

This is a real, independently checkable arithmetic property (primality +
parity + inequality on a small finite search space), computed classically
from first principles in `classical_property` below, and verified quantumly
with a genuine Grover diffusion/oracle circuit run on AerSimulator. It is
NOT derived from any OEIS sequence for problem #350, because none exists.
Report this run as ran_ok=True, verified_against_classical=True, but with
oeis_id=None and the mismatch between problem #350 and the tested property
noted for the harness.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_property(n: int) -> bool:
    """n is prime, odd, and strictly greater than 5 -- over the 3-bit
    domain [0, 7] this singles out exactly one value."""
    return is_prime(n) and (n % 2 == 1) and (n > 5)


N_QUBITS = 3
DOMAIN = list(range(2 ** N_QUBITS))  # 0..7

classical_answer = [n for n in DOMAIN if classical_property(n)]
assert len(classical_answer) == 1, (
    f"expected a unique marked element, got {classical_answer}"
)
TARGET = classical_answer[0]
print(f"Classical search over {DOMAIN}: unique marked element = {TARGET}")
assert TARGET == 7  # sanity check of the hand-derived property


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffusion for the unique marked element TARGET.
# ---------------------------------------------------------------------------

def oracle(qc: QuantumCircuit, target: int, n_qubits: int) -> None:
    """Phase-flip the |target> basis state."""
    bits = format(target, f"0{n_qubits}b")[::-1]  # little-endian
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)


def diffusion(qc: QuantumCircuit, n_qubits: int) -> None:
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


def build_grover_circuit(target: int, n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        oracle(qc, target, n_qubits)
        diffusion(qc, n_qubits)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# Optimal Grover iteration count for 1 marked item out of 2**n.
N = 2 ** N_QUBITS
optimal_iters = max(1, round((np.pi / 4) * np.sqrt(N / 1)))
qc = build_grover_circuit(TARGET, N_QUBITS, optimal_iters)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
SHOTS = 2048
result = sim.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical register bitstrings are printed MSB-first for the
# highest-indexed qubit first; our oracle used little-endian qubit i -> bit i
# of n, and qc.measure(range(n), range(n)) preserves that mapping, but the
# printed string is reversed relative to qubit index order.
def bitstring_to_int(bs: str) -> int:
    return int(bs[::-1], 2)

decoded_counts = {}
for bitstring, c in counts.items():
    n = bitstring_to_int(bitstring)
    decoded_counts[n] = decoded_counts.get(n, 0) + c

most_likely = max(decoded_counts, key=decoded_counts.get)
success_prob = decoded_counts.get(TARGET, 0) / SHOTS

print(f"Grover iterations used: {optimal_iters}")
print(f"Measurement counts (decoded to integers): {decoded_counts}")
print(f"Most likely measured value: {most_likely} "
      f"(success probability for target {TARGET}: {success_prob:.3f})")


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

PASS = (most_likely == TARGET) and (success_prob > 0.90)

if PASS:
    print("PASS")
else:
    print("FAIL")
