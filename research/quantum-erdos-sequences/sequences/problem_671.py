"""
Erdos problem #671 — quantum-testable sequence entry (best-effort placeholder).

LIMITATION (read first): Erdos problem #671's entry in erdosproblems/data/problems.yaml
lists `oeis: ["N/A"]` — it has no associated OEIS sequence at all. Its only metadata is
prize "$250", status "open", and tag "analysis" (an analysis-flavored open problem with
no finite combinatorial sequence attached). That means there is no OEIS-derived small,
finite, computable sequence property to genuinely build a Grover/QPE/QAE circuit around
for THIS problem, as the task requires (deriving/checking an actual OEIS-backed term).

Per the task's own fallback instructions ("if no OEIS id ... write the script anyway
with your best honest attempt, note the limitation clearly, and report ran_ok/
verified_against_classical accurately rather than faking a pass"), this script does NOT
fabricate a fake connection to problem #671's actual mathematical content. Instead it
demonstrates the same *category* of technique (Grover search implementing an oracle for
a decidable arithmetic property) on a small, honestly unrelated, well-defined instance:
primality testing by trial division over a fixed finite domain. This is offered only as
a good-faith placeholder circuit, not as a claim that it tests problem #671.

Chosen instance (unrelated to #671, chosen only because it is small/finite/computable):
  - Domain: integers N in [0, 15] (4 qubits).
  - Property: N is prime (trial division, computed classically first-principles below).
  - Classical answer: the prime subset of {0,...,15} = {2,3,5,7,11,13}.
  - Quantum method: Grover's algorithm with an oracle marking exactly the prime states,
    run on AerSimulator, compared against the classical set.

Result semantics:
  - verified_against_classical = True means the quantum circuit reproduced the correct
    classical answer for the (unrelated placeholder) instance it actually implements.
  - It does NOT mean problem #671 itself was tested, since #671 has no OEIS sequence to
    test in the first place.
"""

from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate
import itertools


# ---------------------------------------------------------------------------
# Step 1: classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
DOMAIN = list(range(2 ** N_QUBITS))  # 0..15
CLASSICAL_PRIMES = sorted(n for n in DOMAIN if is_prime(n))
print(f"Classical primes in [0,{2**N_QUBITS - 1}]: {CLASSICAL_PRIMES}")


# ---------------------------------------------------------------------------
# Step 2: build a Grover oracle marking exactly the prime basis states.
# ---------------------------------------------------------------------------
def bits_of(n: int, width: int) -> str:
    return format(n, f"0{width}b")


def add_oracle(qc: QuantumCircuit, qubits, ancilla, target_values, width):
    """Phase-flip all computational basis states in target_values (list[int])."""
    for val in target_values:
        bitstring = bits_of(val, width)
        # X on qubits that should be 0 for this pattern, so the pattern becomes all-1s.
        for i, b in enumerate(bitstring):
            if b == "0":
                qc.x(qubits[i])
        # Multi-controlled Z via H-MCX-H on the last qubit, using the ancilla-free MCX.
        qc.h(qubits[-1])
        qc.append(MCXGate(width - 1), qubits[:-1] + [qubits[-1]])
        qc.h(qubits[-1])
        for i, b in enumerate(bitstring):
            if b == "0":
                qc.x(qubits[i])


def add_diffuser(qc: QuantumCircuit, qubits, width):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.append(MCXGate(width - 1), qubits[:-1] + [qubits[-1]])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(target_values, width, n_iterations):
    q = QuantumRegister(width, "q")
    qc = QuantumCircuit(q)
    qc.h(q)
    for _ in range(n_iterations):
        add_oracle(qc, list(q), None, target_values, width)
        add_diffuser(qc, list(q), width)
    qc.measure_all()
    return qc


# Optimal Grover iteration count ~ (pi/4) * sqrt(N / M)
import math

N = 2 ** N_QUBITS
M = len(CLASSICAL_PRIMES)
n_iter = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Using {n_iter} Grover iteration(s) for N={N}, M={M} marked states")

qc = build_grover_circuit(CLASSICAL_PRIMES, N_QUBITS, n_iter)


# ---------------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator.
# ---------------------------------------------------------------------------
sim = AerSimulator()
job = sim.run(qc, shots=2000)
result = job.result()
counts = result.get_counts()

# Qiskit bit ordering: rightmost char is qubit 0 -> reverse to read as our width-bit int.
measured_counts = {}
for bitstring, c in counts.items():
    val = int(bitstring[::-1], 2)
    measured_counts[val] = measured_counts.get(val, 0) + c

# Take the top-M most frequently measured values as Grover's proposed answer set.
top_values = sorted(measured_counts.items(), key=lambda kv: -kv[1])[:M]
quantum_primes = sorted(v for v, _ in top_values)

print(f"Quantum (Grover) top-{M} measured values: {quantum_primes}")
print(f"Full measurement histogram (value: count): "
      f"{dict(sorted(measured_counts.items()))}")

verified = quantum_primes == CLASSICAL_PRIMES

if verified:
    print("PASS")
else:
    print("FAIL")
