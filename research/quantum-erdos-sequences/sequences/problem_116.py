"""
Erdos problem #116 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com repo, entry "number: 116"):
    prize: no
    status: proved (Lean)
    oeis: ["N/A"]
    tags: ["polynomials", "analysis"]

HONESTY NOTE / LIMITATION
--------------------------
Problem 116 has no associated OEIS sequence -- the metadata field is literally
`oeis: ["N/A"]`. There is therefore no OEIS "term membership" property to test
with a quantum circuit for this problem, and this script does NOT claim one.
Per the task's fallback instructions, this is the "best honest attempt": a
genuine, real, self-contained Grover search built around the problem's actual
tags ("polynomials", "analysis" -> elementary polynomial/number-theoretic
root-finding), on a small finite instance, with the classical answer derived
from first principles in this script (not copied from anywhere) and checked
against the quantum result.

Chosen finite, computable property
-----------------------------------
Let f(x) = x^2 - 1 (an integer polynomial, tying to the "polynomials" tag).
Search space: x in {0, 1, ..., 7} (3 qubits, N = 8 <= 64 as required).
Property tested: f(x) ≡ 0 (mod 15), i.e. x^2 ≡ 1 (mod 15).

This is computed classically first (first principles, brute force over the
8-element search space), giving the marked set S. A 3-qubit Grover search
then searches the same space for exactly that set S, using an oracle that
phase-flips precisely the elements of S (the oracle is built directly from
the classically-derived set, which is standard practice for Grover: you must
know what you are searching for in order to build the marking oracle -- the
quantum circuit's job, and what is being verified here, is that amplitude
amplification correctly concentrates measurement probability onto the
classically-verified solution set).

PASS criterion: sampling the final circuit on AerSimulator, the two most
frequent measured outcomes (highest probability mass) equal the classical
solution set S, and their combined measured probability substantially
exceeds the uniform baseline of |S|/8.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np

N_QUBITS = 3
N = 2 ** N_QUBITS  # 8

# ---------------------------------------------------------------------------
# Step 1: classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------
def f(x: int) -> int:
    return x * x - 1

MODULUS = 15
classical_solutions = sorted(x for x in range(N) if f(x) % MODULUS == 0)
print(f"Classical search space: x in 0..{N - 1}")
print(f"Property: x^2 - 1 ≡ 0 (mod {MODULUS})")
print(f"Classical solutions (brute force): {classical_solutions}")

assert len(classical_solutions) >= 1, "expected at least one solution in range"
marked = classical_solutions  # e.g. [1, 4]

# ---------------------------------------------------------------------------
# Step 2: build a 3-qubit Grover circuit whose oracle marks exactly `marked`.
# ---------------------------------------------------------------------------
def oracle(qc: QuantumCircuit, targets):
    """Phase-flip each basis state in `targets` (little-endian integers)."""
    for t in targets:
        bits = format(t, f"0{N_QUBITS}b")[::-1]  # little-endian bit string
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(N_QUBITS - 1)
        qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
        qc.h(N_QUBITS - 1)
        for i in zero_positions:
            qc.x(i)


def diffuser(qc: QuantumCircuit):
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))


# Optimal iteration count for |marked| solutions out of N states.
theta = np.arcsin(np.sqrt(len(marked) / N))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    oracle(qc, marked)
    diffuser(qc)
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator.
# ---------------------------------------------------------------------------
sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Convert measured bitstrings (Qiskit prints classical bits MSB..LSB, our
# register is little-endian in `oracle`/`diffuser` above) back to integers.
def bitstring_to_int(bs: str) -> int:
    return int(bs[::-1], 2)

freq = {}
for bitstring, c in counts.items():
    x = bitstring_to_int(bitstring)
    freq[x] = freq.get(x, 0) + c

sorted_freq = sorted(freq.items(), key=lambda kv: -kv[1])
top_k = sorted(x for x, _ in sorted_freq[: len(marked)])
top_k_prob = sum(c for x, c in freq.items() if x in top_k) / shots
baseline_prob = len(marked) / N

print(f"Grover iterations used: {iterations}")
print(f"Measured frequencies (top): {sorted_freq[: len(marked) + 2]}")
print(f"Top-{len(marked)} measured states: {top_k}  (classical solutions: {marked})")
print(f"Top-{len(marked)} measured probability mass: {top_k_prob:.3f}"
      f"  (uniform baseline would be {baseline_prob:.3f})")

quantum_matches_classical = (top_k == marked) and (top_k_prob > baseline_prob * 1.5)

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
