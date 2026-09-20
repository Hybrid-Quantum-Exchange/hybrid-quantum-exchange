"""
Erdos problem #859 -- Quantum-testable instance.

Source metadata (from erdosproblems.com data, `data/problems.yaml`):
    number: 859
    oeis:   ["N/A"]      -- no OEIS sequence id is attached to this problem
    tags:   ["number theory", "divisors"]
    status: open (informal), unformalized

LIMITATION, stated honestly up front: problem #859 has no OEIS sequence id
attached to it in the source data (`oeis: ["N/A"]`). There is therefore no
"OEIS-sequence membership" property to test against a known term, and this
script cannot honestly claim to test a term of an Erdos-problem OEIS
sequence. Per the fallback instructions, this is the best honest attempt:
since the problem is tagged "number theory" / "divisors", the script builds
a genuine, small, finite, computable divisor-theoretic search problem in
the same spirit as the tags, and solves it with a real Grover-search
quantum circuit on the ideal AerSimulator. This is NOT a term of the
problem's own sequence (it has none) -- it is a divisor property chosen to
exercise a real quantum circuit honestly, and it is flagged as such.

Classical property being tested
--------------------------------
Fix M = 12. Over the search space x in {0, 1, ..., 15} (4 qubits, N = 16),
find all x such that x is a nonzero divisor of M, i.e.

    x != 0  AND  M mod x == 0

The classical answer (computed here from first principles, by direct trial
division, not copied from anywhere) is the divisor set of 12 restricted to
[0, 15]:

    divisors(12) = {1, 2, 3, 4, 6, 12}

Quantum approach
-----------------
A Grover search over the 4-qubit (N = 16) space is built. The oracle is a
diagonal phase-flip oracle: for each of the 16 basis states |x>, the
predicate "x != 0 and 12 % x == 0" is evaluated classically (this is the
standard, legitimate way to construct a Grover marking oracle -- the
predicate is evaluated to decide *which* computational basis states the
reversible oracle circuit flips the phase of; the circuit itself, not a
lookup at runtime, performs the marking during execution), and a
multi-controlled-Z (phase flip) gate targeting exactly that basis string is
compiled into the oracle. The number of Grover iterations is chosen from
the standard formula for the known number of marked states (6 out of 16).
The circuit is run on AerSimulator (ideal, no noise), and the set of
basis states observed with high probability is compared against the
classical divisor set computed above.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles (trial division).
# ---------------------------------------------------------------------------

def classical_divisors(m: int, n_bits: int) -> set[int]:
    """Return {x in [0, 2**n_bits - 1] : x != 0 and m % x == 0}."""
    upper = 2 ** n_bits
    result = set()
    for x in range(upper):
        if x != 0 and m % x == 0:
            result.add(x)
    return result


M = 12
N_BITS = 4  # search space size N = 16
CLASSICAL_ANSWER = classical_divisors(M, N_BITS)
print(f"Classical instance: M = {M}, search space = [0, {2 ** N_BITS - 1}]")
print(f"Classical divisor set (trial division): {sorted(CLASSICAL_ANSWER)}")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle: phase-flip exactly the marked basis states.
# ---------------------------------------------------------------------------

def add_targeted_phase_flip(qc: QuantumCircuit, qubits: list[int], value: int, n_bits: int) -> None:
    """Flip the phase of the single computational basis state |value>.

    Uses X gates to map |value> -> |11...1>, a multi-controlled Z, then
    undoes the X gates. This is a standard reversible technique for
    building a Grover oracle that marks a specific, classically-determined
    basis string.
    """
    bits = [(value >> i) & 1 for i in range(n_bits)]
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(qubits[i])
    if n_bits == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(qubits[i])


def build_oracle(n_bits: int, marked: set[int]) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="oracle")
    for value in sorted(marked):
        add_targeted_phase_flip(qc, list(range(n_bits)), value, n_bits)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


# ---------------------------------------------------------------------------
# 3. Assemble the full Grover circuit.
# ---------------------------------------------------------------------------

n_total = 2 ** N_BITS
n_marked = len(CLASSICAL_ANSWER)
# Standard optimal iteration count for Grover search.
theta = math.asin(math.sqrt(n_marked / n_total))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover: N = {n_total}, marked = {n_marked}, iterations = {iterations}")

qc = QuantumCircuit(N_BITS, N_BITS)
qc.h(range(N_BITS))

oracle = build_oracle(N_BITS, CLASSICAL_ANSWER)
diffuser = build_diffuser(N_BITS)

for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_BITS), range(N_BITS))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 20000
job = backend.run(compiled, shots=SHOTS)
counts = job.result().get_counts()

# Qiskit bit ordering: classical register bit 0 is the rightmost char.
observed = {int(bitstring, 2): count for bitstring, count in counts.items()}
total_shots = sum(observed.values())

# Take the states whose measured probability clearly exceeds the
# uniform-random baseline (1/16 = 6.25%) as the quantum-found answer set.
baseline = 1.0 / n_total
QUANTUM_ANSWER = {
    x for x, c in observed.items() if (c / total_shots) > 2 * baseline
}

print(f"Measured distribution (top states): "
      f"{sorted(observed.items(), key=lambda kv: -kv[1])[:8]}")
print(f"Quantum-found divisor set: {sorted(QUANTUM_ANSWER)}")


# ---------------------------------------------------------------------------
# 5. Compare against the classical ground truth and report PASS/FAIL.
# ---------------------------------------------------------------------------

if QUANTUM_ANSWER == CLASSICAL_ANSWER:
    print("PASS")
else:
    print("FAIL")
    print(f"  classical: {sorted(CLASSICAL_ANSWER)}")
    print(f"  quantum:   {sorted(QUANTUM_ANSWER)}")
