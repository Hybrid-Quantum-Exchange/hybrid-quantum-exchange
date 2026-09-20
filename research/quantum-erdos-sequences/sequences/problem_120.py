"""
Erdos problem #120 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, block "number: 120"):
    prize: $100
    informal_status: open (last update 2025-08-31)
    comments: "Erdos similarity problem"
    tags: [combinatorics]
    oeis: ["N/A"]

HONEST LIMITATION: Problem #120 (the Erdos similarity problem) has NO OEIS
sequence id attached in the source data (oeis: ["N/A"]). It is also an open
research question about which real number sets are "universal" for
affine copies of infinite sequences -- there is no small finite decision
procedure that faithfully captures the open problem itself. Per the task
instructions, since no OEIS id is available, this script does not attempt to
fabricate a sequence-membership property tied to problem #120. Instead it
gives a best-honest-effort genuine quantum computation on a small, finite,
classically-checkable arithmetic search problem in the same spirit as the
"combinatorics" tag (a Diophantine/counting search over a finite domain),
and is transparent that the connection to problem #120's actual open
conjecture is illustrative only, not a resolution or formalization of it.

Chosen finite, computable property
-----------------------------------
Over the finite domain N = {0, 1, ..., 15} (4 bits), consider the triangular
number function T(x) = x*(x+1)/2 mod 16. We search for the unique x in this
domain with T(x) == 6 (computed classically below, first principles, no
lookup). This is a genuine small search/oracle problem: exactly the kind of
"find the unique x satisfying an arithmetic property" task Grover's
algorithm solves with quadratic speedup, and it is fully classically
verifiable.

Circuit
-------
A real Grover search (Qiskit + AerSimulator, statevector/qasm simulation,
no shortcuts): a phase oracle built directly from the classically-computed
target bitstring, a standard diffusion operator, and the optimal number of
Grover iterations for a 4-qubit (16-item) search with a single marked item.

The script computes the classical answer first, builds and runs the circuit,
and prints PASS/FAIL by comparing the most frequent measured bitstring to
the classical answer.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the target, from first principles.
# ---------------------------------------------------------------------------

N_QUBITS = 4
N = 2 ** N_QUBITS  # 16

def triangular_mod(x, mod):
    return (x * (x + 1) // 2) % mod

TARGET_VALUE = 6  # T(x) mod 16 == 6

classical_matches = [x for x in range(N) if triangular_mod(x, N) == TARGET_VALUE]
assert len(classical_matches) == 1, (
    f"Expected a unique solution in [0, {N}) for this instance, "
    f"got {classical_matches}"
)
CLASSICAL_ANSWER = classical_matches[0]
CLASSICAL_BITSTRING = format(CLASSICAL_ANSWER, f"0{N_QUBITS}b")

print(f"Classical search over N = {N} values (T(x) = x(x+1)/2 mod {N}):")
print(f"  unique x with T(x) == {TARGET_VALUE}: x = {CLASSICAL_ANSWER} "
      f"(bitstring '{CLASSICAL_BITSTRING}', qiskit little-endian "
      f"'{CLASSICAL_BITSTRING[::-1]}')")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle for the classically-known target bitstring.
# ---------------------------------------------------------------------------
# Qiskit orders qubit 0 as the least-significant bit of the classical
# register, so bit i of the target corresponds to qubit i.
target_bits = [int(b) for b in CLASSICAL_BITSTRING[::-1]]  # bit i -> qubit i


def build_oracle(n_qubits, bits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    # Flip qubits that should be 0 in the target so the target maps to |1..1>.
    zero_positions = [i for i, b in enumerate(bits) if b == 0]
    for i in zero_positions:
        qc.x(i)
    # Multi-controlled Z (phase flip) on all-ones, implemented via H-MCX-H
    # on the last qubit.
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i in zero_positions:
        qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(N_QUBITS, target_bits)
diffuser = build_diffuser(N_QUBITS)

# Optimal number of Grover iterations for 1 marked item out of N.
n_iterations = max(1, round((math.pi / 4) * math.sqrt(N)))
print(f"Grover iterations used: {n_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(n_iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))
qc = qc.decompose().decompose()


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------
simulator = AerSimulator()
shots = 2048
result = simulator.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit reports classical-bit strings as 'c_{n-1} ... c_1 c_0' (c0 on the
# right), and here clbit i was measured from qubit i, so the string read
# directly as a standard big-endian binary numeral already reconstructs x.
most_common_bits, most_common_count = Counter(counts).most_common(1)[0]
quantum_answer = int(most_common_bits, 2)

print(f"Measured counts (top 5): {Counter(counts).most_common(5)}")
print(f"Most frequent outcome: x = {quantum_answer} "
      f"({most_common_count}/{shots} shots)")

verified = quantum_answer == CLASSICAL_ANSWER

print()
if verified:
    print("PASS: Grover search recovered the classically-computed unique "
          f"solution x = {CLASSICAL_ANSWER} for T(x) mod {N} == {TARGET_VALUE}.")
else:
    print("FAIL: quantum result did not match the classical answer "
          f"(classical={CLASSICAL_ANSWER}, quantum={quantum_answer}).")
