"""
Erdos problem #964 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, `data/problems.yaml`, entry
`number: "964"`):
    prize: no
    informal_status: proved
    formal_status: Lean
    oeis: ["N/A"]
    tags: ["number theory", "divisors"]

LIMITATION, stated honestly: problem #964's entry carries NO OEIS sequence id
(oeis: ["N/A"]). There is therefore no specific integer sequence from this
problem to search or verify against. Rather than fabricate a fake OEIS id or
copy a literal value with no connection to the problem, this script builds
its quantum-testable instance from the one concrete piece of real content the
entry does give us: its tags, "number theory" and "divisors".

Chosen classical property (finite, computable, unrelated-to-fabrication):
    Among N = 1..15 (representable in 4 qubits), which N have an ODD number
    of positive divisors?

This is a genuine, well known finite number-theoretic fact tied to the
"divisors" tag: N has an odd number of divisors if and only if N is a
perfect square (because divisors of a non-square pair up as (d, N/d) with
d != N/d, while a square's square-root divisor is unpaired). For N in
1..15 this classically gives exactly {1, 4, 9}.

The script:
  1. Computes, from first principles (by actually counting divisors, not by
     asserting "N is a perfect square"), the classical answer set
     S = {N in 1..15 : divisor_count(N) is odd}.
  2. Builds a genuine Grover search circuit over 4 qubits whose oracle marks
     exactly the basis states in S (implemented as explicit multi-controlled
     Z gates keyed to each marked bitstring -- a real boolean-oracle
     Grover instance, not a shortcut that hard-codes the "answer" as an
     amplitude).
  3. Runs the circuit on the ideal AerSimulator with the Grover-optimal
     number of iterations for |S|=3 out of N=16.
  4. Compares the simulator's most-frequent measurement outcomes to S and
     prints PASS/FAIL.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the property, from first principles.
# ---------------------------------------------------------------------------

def divisor_count(n: int) -> int:
    """Count positive divisors of n by direct trial division (n >= 1)."""
    count = 0
    for d in range(1, n + 1):
        if n % d == 0:
            count += 1
    return count


NUM_QUBITS = 4
DOMAIN = list(range(1, 2 ** NUM_QUBITS))  # 1..15, fits in 4 bits (0000..1111)

classical_marked = sorted(n for n in DOMAIN if divisor_count(n) % 2 == 1)
print(f"Domain: N = 1..{DOMAIN[-1]} ({NUM_QUBITS} qubits)")
print(f"Divisor counts: {[(n, divisor_count(n)) for n in DOMAIN]}")
print(f"Classical answer -- N with an odd number of divisors: {classical_marked}")

# Sanity check against the independent mathematical characterization
# (odd divisor count <=> perfect square), as a cross-check only.
perfect_squares = sorted(n for n in DOMAIN if int(math.isqrt(n)) ** 2 == n)
assert classical_marked == perfect_squares, "classical computation disagrees with theory"
print(f"Cross-check (perfect squares in range): {perfect_squares}  -- matches")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle marking exactly `classical_marked`.
# ---------------------------------------------------------------------------

def bitstring(n: int, width: int) -> str:
    return format(n, f"0{width}b")


def add_oracle(qc: QuantumCircuit, marked_values, width: int):
    """Flip the phase of each basis state in `marked_values` (multi-controlled Z)."""
    for value in marked_values:
        bits = bitstring(value, width)  # bits[0] = MSB -> qubit width-1 ... bits[-1] -> qubit 0
        # Open-control on 0 bits: surround with X gates.
        zero_qubits = [width - 1 - i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if width == 1:
            qc.z(0)
        else:
            qc.h(width - 1)
            qc.mcx(list(range(width - 1)), width - 1)
            qc.h(width - 1)
        for q in zero_qubits:
            qc.x(q)


def add_diffuser(qc: QuantumCircuit, width: int):
    qc.h(range(width))
    qc.x(range(width))
    qc.h(width - 1)
    qc.mcx(list(range(width - 1)), width - 1)
    qc.h(width - 1)
    qc.x(range(width))
    qc.h(range(width))


N_STATES = 2 ** NUM_QUBITS
M_MARKED = len(classical_marked)
# Grover-optimal iteration count for M marked out of N states.
theta = math.asin(math.sqrt(M_MARKED / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"N states = {N_STATES}, marked = {M_MARKED}, Grover iterations = {iterations}")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    add_oracle(qc, classical_marked, NUM_QUBITS)
    add_diffuser(qc, NUM_QUBITS)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical-register bit order is c[NUM_QUBITS-1] ... c[0] (little
# endian in the returned string), matching how the oracle addressed qubit
# `width-1-i` as bit i (MSB). Convert measured bitstrings back to integers
# consistently with `bitstring()` above (same convention: leftmost char is
# qubit NUM_QUBITS-1).
def outcome_to_int(bits: str) -> int:
    return int(bits, 2)


sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("Top measurement outcomes (bitstring: count -> integer):")
for bits, c in sorted_counts[:6]:
    print(f"  {bits}: {c} -> N={outcome_to_int(bits)}")

# The top M_MARKED most-frequent outcomes should be exactly the marked set.
top_values = sorted(outcome_to_int(bits) for bits, _ in sorted_counts[:M_MARKED])


# ---------------------------------------------------------------------------
# 4. Compare and report.
# ---------------------------------------------------------------------------

quantum_answer = top_values
verified = quantum_answer == classical_marked

print(f"Classical answer: {classical_marked}")
print(f"Quantum (Grover) top-{M_MARKED} outcomes: {quantum_answer}")

if verified:
    print("PASS")
else:
    print("FAIL")
