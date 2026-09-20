"""
Erdos problem #491 — quantum-testable sequence entry (LIMITATION NOTICE)
==========================================================================

Source metadata (data/problems.yaml, entry "number: '491'"):
    prize: no
    informal_status: proved (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION: problem #491 carries no OEIS sequence id in the source data
(the field is literally "N/A") and the erdosproblems.com clone available
here has no per-problem statement file for #491 either (checked: no file
matching "*491*" anywhere under the repository). There is therefore no
actual sequence to build a faithful quantum-testable property from, and
this script does NOT claim to test problem #491's real mathematical
content — doing so would mean fabricating a statement that isn't in the
source. Per instructions, this is the best honest attempt rather than a
faked pass: a genuine, self-contained quantum circuit exercising a real,
finite, computable number-theoretic property (primality), which is at
least in the same subject area as the problem's only tag ("number
theory"). Anyone assembling the real problem statement for #491 later
should replace this file's classical property with the actual one.

Chosen property (self-contained, unrelated to any unverified OEIS value):
    Grover search over N = 16 (4 qubits) for numbers n in [0, 15] that are
    prime. The "oracle" marks n such that is_prime(n) is True. The
    classical answer (computed here from first principles by trial
    division, not copied from any table) is the set of primes < 16:
    {2, 3, 5, 7, 11, 13} -- 6 marked states out of 16.

    The circuit runs a diffusion-based Grover search on all 4 qubits,
    with an oracle built from an explicit multi-controlled-Z per marked
    computational basis state (no lookup tables, no built-in primality
    gate). After the optimal number of Grover iterations, we measure and
    check that the top measured outcomes are exactly the classical prime
    set.

Verification: PASS if, over many shots, the classical primes below 16
account for the overwhelming majority of the measured probability mass
(a real amplitude-amplification success, not just a lucky sample).
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import math


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
N = 2 ** N_QUBITS  # 16

classical_primes = [n for n in range(N) if is_prime(n)]
print(f"Classical property: primes in [0, {N - 1}) by trial division")
print(f"Classical answer set: {classical_primes} ({len(classical_primes)} marked states)")

assert classical_primes == [2, 3, 5, 7, 11, 13], "classical computation sanity check failed"


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle: mark each prime basis state with a
#    multi-controlled-Z (phase flip), driven purely by each n's own
#    bit pattern -- no external lookup gate.
# ---------------------------------------------------------------------------

def mark_state(qc: QuantumCircuit, n: int, n_qubits: int) -> None:
    """Flip the phase of computational basis state |n> using an
    X-sandwiched multi-controlled-Z (MCZ) built from a Hadamard + MCX."""
    bits = format(n, f"0{n_qubits}b")[::-1]  # little-endian bit string
    zero_qubits = [i for i, b in enumerate(bits) if b == "0"]

    for q in zero_qubits:
        qc.x(q)

    # Multi-controlled Z on all n_qubits: H on target, MCX, H on target.
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    for q in zero_qubits:
        qc.x(q)


def oracle(n_qubits: int, marked: list) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for n in marked:
        mark_state(qc, n, n_qubits)
    return qc


def diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# 3. Assemble the full Grover circuit with the optimal iteration count.
# ---------------------------------------------------------------------------

M = len(classical_primes)
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations} (M={M} marked out of N={N})")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

orc = oracle(N_QUBITS, classical_primes)
dif = diffuser(N_QUBITS)
for _ in range(iterations):
    qc.compose(orc, inplace=True)
    qc.compose(dif, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 20000
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first as c[n-1]...c[0]; convert to ints.
measured_probs = {}
for bitstring, cnt in counts.items():
    n = int(bitstring, 2)
    measured_probs[n] = measured_probs.get(n, 0) + cnt / SHOTS

prime_mass = sum(measured_probs.get(n, 0.0) for n in classical_primes)
nonprime_mass = 1.0 - prime_mass

top_states = sorted(measured_probs.items(), key=lambda kv: -kv[1])[:len(classical_primes)]
top_state_ints = sorted(n for n, _ in top_states)

print(f"Measured probability mass on classical prime states: {prime_mass:.4f}")
print(f"Measured probability mass on non-prime states:        {nonprime_mass:.4f}")
print(f"Top {M} most-measured states: {top_state_ints}")

# Success criteria for a genuine amplitude-amplification result:
#  (a) the amplified probability mass is concentrated on the true prime
#      set well above the uninformed baseline M/N, and
#  (b) the M most-frequently measured states are exactly the classical
#      prime set.
baseline = M / N
verified = prime_mass > 0.8 and top_state_ints == sorted(classical_primes)

print()
if verified:
    print("PASS")
else:
    print("FAIL")

print()
print(f"(baseline uninformed mass would be {baseline:.4f}; "
      f"achieved {prime_mass:.4f})")
