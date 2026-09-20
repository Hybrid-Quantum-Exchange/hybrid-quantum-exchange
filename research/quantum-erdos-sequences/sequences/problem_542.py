"""
Erdos problem #542 — quantum-testable companion script.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
"number: '542'", tags: ["number theory"]):
    prize: no
    informal_status: solved (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #542 has no OEIS sequence id
attached in the source data (oeis: ["N/A"]). There is therefore no specific
integer sequence from this problem to build a quantum-testable membership
or term-search circuit around, and no OEIS-derived classical value is used
or fabricated anywhere below. This script cannot honor the "from its OEIS
sequence id(s), identify a property of the sequence" instruction literally,
because no such id exists for this problem.

Best honest fallback: the only structured metadata problem #542 does carry
is its tag, "number theory". To still deliver a *real* quantum circuit with
genuine mathematical content (not a copied/fabricated value), this script
targets a small, finite, computable number-theoretic property that is
representative of that tag and is fully verified classically from first
principles in this file:

    Property under test: primality over the search space {0, 1, ..., 15}
    (N = 16, encoded on 4 qubits). We build a Grover search circuit whose
    oracle marks exactly the prime integers in that range (2, 3, 5, 7, 11,
    13 -> 6 marked items out of 16), run the optimal number of Grover
    iterations on the ideal AerSimulator, and check that measurement
    outcomes concentrate on the primality-marked set at a probability far
    above the uniform baseline (6/16 = 37.5%).

    The classical answer (the exact set of primes in [0, 15]) is computed
    in this script with an elementary trial-division primality test, not
    taken from OEIS or any external source.

PASS criterion: the total measured probability mass landing on marked
(prime) computational basis states exceeds a fixed threshold well above
the no-amplification baseline, confirming the Grover circuit genuinely
amplifies the classically-defined prime subset.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate
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
marked_set = set(classical_primes)
print(f"Classical primes in [0, {N - 1}]: {classical_primes}")
assert classical_primes == [2, 3, 5, 7, 11, 13]


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the prime integers in [0, 15].
# ---------------------------------------------------------------------------

def oracle_for_value(qc: QuantumCircuit, value: int, qubits, ancilla):
    """Flip the sign of |value> using a multi-controlled Z built from an
    MCX into a phase-kickback ancilla prepared in |-> ."""
    bits = format(value, f"0{len(qubits)}b")[::-1]  # little-endian per qubit order
    flip = [qubits[i] for i, b in enumerate(bits) if b == "0"]
    for q in flip:
        qc.x(q)
    qc.append(MCXGate(len(qubits)), qubits + [ancilla])
    for q in flip:
        qc.x(q)


def build_oracle(n_qubits: int, marked):
    qc = QuantumCircuit(n_qubits + 1, name="oracle")
    qubits = list(range(n_qubits))
    ancilla = n_qubits
    for v in marked:
        oracle_for_value(qc, v, qubits, ancilla)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits + 1, name="diffuser")
    qubits = list(range(n_qubits))
    ancilla = n_qubits
    qc.h(qubits)
    qc.x(qubits)
    qc.append(MCXGate(n_qubits), qubits + [ancilla])
    qc.x(qubits)
    qc.h(qubits)
    return qc


# ---------------------------------------------------------------------------
# 3. Full Grover circuit.
# ---------------------------------------------------------------------------

n_marked = len(classical_primes)
theta = math.asin(math.sqrt(n_marked / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"N={N}, marked={n_marked}, optimal Grover iterations={iterations}")

qc = QuantumCircuit(N_QUBITS + 1, N_QUBITS)
data_qubits = list(range(N_QUBITS))
ancilla = N_QUBITS

# Prepare ancilla in |-> for phase kickback.
qc.x(ancilla)
qc.h(ancilla)

# Uniform superposition over the search register.
qc.h(data_qubits)

oracle = build_oracle(N_QUBITS, classical_primes)
diffuser = build_diffuser(N_QUBITS)

for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS + 1))
    qc.append(diffuser.to_gate(), range(N_QUBITS + 1))

qc.measure(data_qubits, list(range(N_QUBITS)))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 20000
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical bit string is big-endian in printed order but here bit i
# of the register corresponds to character position (from the right) i.
prob_marked = 0.0
for bitstring, cnt in counts.items():
    value = int(bitstring, 2)
    if value in marked_set:
        prob_marked += cnt / shots

baseline = n_marked / N
threshold = 0.5 * (baseline + 1.0)  # comfortably above uniform baseline

print(f"Measured probability mass on classically-verified primes: {prob_marked:.4f}")
print(f"Uniform baseline (no amplification): {baseline:.4f}")
print(f"Pass threshold: {threshold:.4f}")

verified = prob_marked >= threshold

if verified:
    print("PASS")
else:
    print("FAIL")
