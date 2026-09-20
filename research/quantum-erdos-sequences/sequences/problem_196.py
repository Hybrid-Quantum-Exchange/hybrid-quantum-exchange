"""
Erdos problem #196 -- quantum-testable instance.

Source metadata (data/problems.yaml, entry "number: '196'", tags:
["arithmetic progressions"]):

    oeis: ["N/A"]

Problem #196 has NO associated OEIS sequence id in the source data ("N/A"),
so there is no OEIS term list to derive a "membership of an integer in the
sequence" style property from. This is the honest limitation for this
lane: no genuine OEIS-anchored sequence property exists to encode.

Rather than fabricate an OEIS-backed claim, this script instead builds a
REAL, non-trivial quantum circuit on the problem's actual tag,
"arithmetic progressions": Grover's algorithm searching for the starting
index of a 3-term arithmetic progression (common difference 1) contained
in a fixed small subset S of {0, ..., 7}.

Concretely:
    N = 8 (so 3 qubits suffice to index x in {0,...,7})
    S = {0, 1, 3, 4, 5, 7}  (a fixed boolean membership set over 0..7)
    Property tested: "x is a valid start of a 3-term AP (x, x+1, x+2) with
    all three terms in S and x+2 < N".

The classical answer (the exact set of marked x) is computed directly in
this script by brute-force enumeration over all 8 values of x -- no OEIS
value is copied or assumed. Grover's algorithm is then run on the ideal
AerSimulator to search the same 8-element space for a marked x, and the
most frequently measured outcome is compared against the classical marked
set.

This script:
  - has no external dependencies beyond qiskit, qiskit_aer, numpy
  - computes the classical answer from first principles (no OEIS lookup)
  - builds and runs a real Grover circuit (oracle + diffuser) on
    AerSimulator
  - prints PASS/FAIL comparing the quantum result to the classical answer

Because problem #196 carries no OEIS id, this is reported as a best-effort
circuit anchored to the problem's tag, NOT a verified property of an OEIS
sequence. See the reported `verified_against_classical` value: it reflects
agreement between the classical brute force and the quantum search result
for this specific constructed instance, not a claim about problem #196
itself.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup)
# ---------------------------------------------------------------------------

N = 8  # search space size, 3 qubits
NUM_QUBITS = 3
S = {0, 1, 3, 4, 5, 7}  # fixed membership set over {0, ..., 7}


def is_marked(x: int) -> bool:
    """x is a valid start of a 3-term AP (x, x+1, x+2), all in S, in range."""
    if x + 2 >= N:
        return False
    return x in S and (x + 1) in S and (x + 2) in S


classical_marked = sorted(x for x in range(N) if is_marked(x))
print(f"Classical brute force over x in 0..{N - 1}, S = {sorted(S)}")
print(f"Classical marked set (3-term AP starts, all terms in S): {classical_marked}")

assert len(classical_marked) >= 1, "instance must have at least one marked element"

# ---------------------------------------------------------------------------
# 2. Grover oracle + diffuser for this fixed marked set
# ---------------------------------------------------------------------------


def build_oracle_phase(marked_values, num_qubits):
    """Marks each value in marked_values with a -1 phase using a proper
    multi-controlled Z (phase kickback via H-MCX-H on an ancilla-free
    target)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for val in marked_values:
        bits = format(val, f"0{num_qubits}b")[::-1]
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on qubit num_qubits-1 as target, phase flip
        # when all qubits are |1>
        qc.h(num_qubits - 1)
        qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
        qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


num_marked = len(classical_marked)
# optimal number of Grover iterations for this instance
theta = math.asin(math.sqrt(num_marked / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

oracle = build_oracle_phase(classical_marked, NUM_QUBITS)
diffuser = build_diffuser(NUM_QUBITS)

for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(NUM_QUBITS))
    qc.append(diffuser.to_instruction(), range(NUM_QUBITS))

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 4096
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Qiskit count keys are written most-significant-classical-bit first
# (c_{n-1} ... c1 c0), i.e. exactly a standard binary string for
# x = c0 + 2*c1 + 4*c2 + ..., so int(bitstring, 2) already gives x with no
# reversal needed.
measured_values = {}
for bitstring, count in counts.items():
    x = int(bitstring, 2)
    measured_values[x] = measured_values.get(x, 0) + count

sorted_measured = sorted(measured_values.items(), key=lambda kv: -kv[1])
print("\nMeasured outcome distribution (top 5):")
for x, count in sorted_measured[:5]:
    marker = "*" if x in classical_marked else " "
    print(f"  x={x} ({count} shots) {marker}")

# success = combined probability mass on classically-marked outcomes
marked_shots = sum(c for x, c in measured_values.items() if x in classical_marked)
success_rate = marked_shots / shots

top_x = sorted_measured[0][0]
quantum_found_marked = top_x in classical_marked

print(f"\nMost frequent measured x: {top_x}")
print(f"Classical marked set: {classical_marked}")
print(f"Fraction of shots landing on a classically-marked x: {success_rate:.3f}")

# Grover with the near-optimal iteration count should concentrate strongly
# (much better than the uniform baseline num_marked/N) on marked states.
baseline = num_marked / N
verified = quantum_found_marked and success_rate > baseline * 1.5

if verified:
    print("\nPASS: Grover search's top outcome is a classically-verified "
          "3-term-AP start, with amplified success probability "
          f"({success_rate:.3f} vs uniform baseline {baseline:.3f}).")
else:
    print("\nFAIL: quantum search result did not match/exceed the "
          "classical baseline expectation.")

print(
    "\nNOTE: Erdos problem #196 has oeis: ['N/A'] in the source data -- "
    "there is no OEIS sequence to anchor this circuit to. The instance "
    "above is a best-effort construction on the problem's tag "
    "('arithmetic progressions'), not a verified OEIS-sequence property."
)
