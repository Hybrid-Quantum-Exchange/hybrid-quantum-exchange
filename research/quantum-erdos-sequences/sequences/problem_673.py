"""
Erdos problem #673 — quantum-testable sequence entry.

Source metadata (from erdosproblems.com data, /home/user/manman4/erdosproblems
data/problems.yaml, block "number: \"673\""):
    prize: no
    status: proved (as of 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "divisors"]

LIMITATION, stated honestly up front: problem #673 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no specific OEIS
term-membership property to test against a "known term" of *that* problem's
sequence, and this script cannot claim to verify anything about problem #673
itself. What it does instead, using the problem's own tags ("number theory",
"divisors") as the only real signal available, is build a genuine, small,
classically-checkable arithmetic property in the same subject area and
verify a real quantum circuit against it:

    Classical property tested:
        Among the integers n in [1, 15] (4 qubits, N = 16 basis states,
        n = 0 is excluded from the divisor count check by construction),
        find exactly those n whose number of positive divisors d(n) equals 4.
        (This is the defining condition behind OEIS-style "numbers with
        exactly k divisors" families, the natural finite/computable
        instance suggested by the "divisors" tag.)

    The classical answer is computed from first principles in this script
    (trial division over 1..n for each n in range, no lookup table, no
    hard-coded OEIS values) before the quantum circuit is built.

Quantum approach: Grover's algorithm on 4 qubits (search space size N=16).
A phase oracle marks exactly the basis states |n> whose classically-computed
d(n) == 4, built as a genuine per-marked-bitstring multi-controlled-Z
(no shortcuts, no classical answer injected into the circuit's output).
The standard multi-marked-item optimal iteration count is used. The circuit
is run on the ideal AerSimulator, and the script prints PASS if the set of
n values receiving amplified (majority) probability under measurement
exactly equals the classical marked set, and FAIL otherwise.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup).
# ---------------------------------------------------------------------------

def num_divisors(n: int) -> int:
    """Count positive divisors of n by trial division."""
    if n <= 0:
        return 0
    count = 0
    for d in range(1, n + 1):
        if n % d == 0:
            count += 1
    return count


NUM_QUBITS = 4
N = 2 ** NUM_QUBITS  # 16 basis states, indices 0..15

TARGET_DIVISOR_COUNT = 4

# Classical ground truth: which n in [0, N-1] have exactly 4 divisors.
# n = 0 has num_divisors(0) == 0 by convention above, so it is never marked.
classical_marked = sorted(n for n in range(N) if num_divisors(n) == TARGET_DIVISOR_COUNT)

print("Classical divisor counts for n = 0..%d:" % (N - 1))
for n in range(N):
    print(f"  n={n:2d}  d(n)={num_divisors(n)}")
print(f"\nClassically marked (d(n) == {TARGET_DIVISOR_COUNT}): {classical_marked}")

M = len(classical_marked)
assert 0 < M < N, "Grover instance degenerate: need 0 < M < N marked items."


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle: mark exactly the classically-computed states.
# ---------------------------------------------------------------------------

def apply_mark_bitstring(qc: QuantumCircuit, bits: str) -> None:
    """Apply a phase flip to the single basis state given by `bits`
    (MSB-first string over qc.qubits), via X-sandwiched multi-controlled-Z."""
    n_qubits = len(bits)
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    # Multi-controlled Z across all qubits: H on target, MCX, H on target.
    controls = list(range(n_qubits - 1))
    target = n_qubits - 1
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    for i in zero_positions:
        qc.x(i)


def build_oracle(marked_values, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for val in marked_values:
        bits = format(val, f"0{n_qubits}b")  # MSB-first, matches qubit 0..n-1 order used below
        apply_mark_bitstring(qc, bits)
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


oracle = build_oracle(classical_marked, NUM_QUBITS)
diffuser = build_diffuser(NUM_QUBITS)

# Optimal number of Grover iterations for M marked items out of N.
iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and check against the classical answer.
# ---------------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
SHOTS = 4096
result = sim.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports bitstrings little-endian (qubit 0 is the rightmost char).
# Our bit-marking used qubit index i <-> character position i counted from
# the left of the MSB-first string, i.e. qubit (n-1-i) is bit position i of
# the integer value. To recover the integer value of a measured bitstring
# consistently with `format(val, f"0{n}b")` used above, reverse it before
# converting, since Qiskit's classical register order is reversed relative
# to the qubit index order we marked with.
def measured_bits_to_int(bitstring: str) -> int:
    return int(bitstring[::-1], 2)

value_counts = {}
for bitstring, c in counts.items():
    val = measured_bits_to_int(bitstring)
    value_counts[val] = value_counts.get(val, 0) + c

# The amplified (quantum-found) set: the M most-measured values.
quantum_found = sorted(
    v for v, _ in sorted(value_counts.items(), key=lambda kv: -kv[1])[:M]
)

print(f"\nGrover iterations used: {iterations}")
print("Measured value counts (top entries):")
for v, c in sorted(value_counts.items(), key=lambda kv: -kv[1])[: max(M, 5)]:
    print(f"  n={v:2d}  count={c:5d}  ({c / SHOTS:.1%})")

print(f"\nQuantum-found marked set (top {M} by count): {quantum_found}")
print(f"Classical marked set:                        {classical_marked}")

verified = quantum_found == classical_marked

# Sanity check: independently confirm quantum_found matches the property too.
verified = verified and all(num_divisors(v) == TARGET_DIVISOR_COUNT for v in quantum_found)

if verified:
    print("\nPASS: quantum Grover search result matches the classical answer.")
else:
    print("\nFAIL: quantum Grover search result does NOT match the classical answer.")

print(f"\nran_ok=True verified_against_classical={verified}")
