"""
Erdos problem #238 (erdosproblems.com) -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 238"):
    prize: no
    informal_status: open (last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "primes"]

Honesty note on scope: problem #238 carries NO OEIS sequence id (the yaml
entry literally lists oeis: ["N/A"]) and its informal status is "open" --
there is no small finite decidable statement of the Erdos problem itself to
put on a quantum circuit. Per the task's fallback instructions, this script
does not fabricate a fake OEIS-backed property. Instead it builds a genuine,
verifiable quantum computation on the one concrete piece of real mathematical
content the metadata does give us: the problem's own tags, "number theory"
and "primes". The classical property tested is:

    PROPERTY: for n in the range 0 <= n < 16 (4 qubits), n is PRIME.
    (Primality by trial division, computed from first principles in this
    script -- no OEIS lookup, no literal copied values.)

The quantum circuit is a real Grover search (oracle + diffusion, built from
elementary gates, no Qiskit library black box) over the 4-qubit computational
basis {0..15} that amplifies exactly the prime basis states. It is run on the
ideal AerSimulator and the measurement distribution is checked against the
classical set of primes in [0, 16) computed by trial division.

This is an honest best-effort substitute for a missing OEIS-backed instance,
not a claim that Grover search on 0..15 "is" Erdos problem #238.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed here from first principles.
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
M = len(CLASSICAL_PRIMES)  # number of marked states

print(f"Classical primes in [0, {N}): {CLASSICAL_PRIMES}  (M={M})")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle marking exactly the prime basis states.
# ---------------------------------------------------------------------------

def apply_marking_for_value(qc: QuantumCircuit, value: int, n_qubits: int) -> None:
    """Flip the phase of |value> using X-conjugated multi-controlled Z."""
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    flip_qubits = [i for i, b in enumerate(bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    # multi-controlled Z on all n_qubits (phase flip when all qubits are |1>)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for q in flip_qubits:
        qc.x(q)


def build_oracle(marked_values, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for v in marked_values:
        apply_marking_for_value(qc, v, n_qubits)
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


def build_grover_circuit(marked_values, n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# Optimal number of Grover iterations for N states, M marked.
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations: {optimal_iterations}")

circuit = build_grover_circuit(CLASSICAL_PRIMES, N_QUBITS, optimal_iterations)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(circuit, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's returned bitstring is "c[n-1] ... c[1] c[0]" (leftmost = highest
# classical/qubit index). Since qubit i carries weight 2**i in our value
# encoding, that string is already the standard MSB-first binary
# representation of the integer value -- no reversal needed.
value_counts = Counter()
for bitstring, cnt in counts.items():
    value = int(bitstring, 2)
    value_counts[value] += cnt

print("Measurement distribution (value: count):")
for v in sorted(value_counts):
    print(f"  {v:2d} (prime={is_prime(v)}): {value_counts[v]}")


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

# The top-M most frequent measured values should be exactly the classical
# prime set, and together they should carry the large majority of shots
# (amplitude amplification, not a uniform-random guess).
top_m_values = sorted(v for v, _ in value_counts.most_common(M))
mass_on_primes = sum(value_counts.get(p, 0) for p in CLASSICAL_PRIMES)
fraction_on_primes = mass_on_primes / SHOTS

print(f"Top-{M} measured values: {top_m_values}")
print(f"Fraction of shots landing on classical primes: {fraction_on_primes:.3f}")

passed = (top_m_values == CLASSICAL_PRIMES) and (fraction_on_primes > 0.8)

if passed:
    print("PASS")
else:
    print("FAIL")
