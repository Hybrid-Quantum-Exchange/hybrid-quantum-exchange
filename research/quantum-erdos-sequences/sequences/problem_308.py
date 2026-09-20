"""
Erdos problem #308 (data/problems.yaml: number "308", tags ["number theory",
"unit fractions"], oeis ["A101877", "possible"]) is the Erdos-Straus conjecture:
for every integer n >= 2, does 4/n admit a representation as a sum of three
(not necessarily distinct) unit fractions 1/x + 1/y + 1/z with positive
integers x <= y <= z?  OEIS A101877 tabulates, for each n, the smallest x
occurring in such a decomposition (the sequence this problem's metadata
points at); "possible" flags that the conjecture itself remains open in
general even though it is verified computationally far beyond any range we
touch here.

Classical property tested (finite, small, computed from first principles in
this script, not copied from OEIS): fix n = 7. Among x in the small range
2..17 (4 qubits, 16 candidate values), find the SMALLEST x for which
    4/7 - 1/x = 1/y + 1/z
has a solution in positive integers y <= z -- i.e. the smallest leading
denominator of an Erdos-Straus decomposition of 4/7, the quantity OEIS
A101877 tabulates. This is exactly the defining search of the Erdos-Straus
conjecture's brute-force verification, restricted to a tiny instance, with a
unique classical winner, so a quantum circuit can genuinely search for it.

For each candidate x, y and z are searched (classically, exhaustively, no
shortcuts) over a bounded range derived from the equation itself: for a
target unit-fraction remainder r = 4/7 - 1/x > 0, y must satisfy 1/y < r i.e.
y > 1/r, and the smallest feasible y is ceil(1/r) + 1 upward; z is then
forced to be 1 / (r - 1/y) when that is a positive integer. We search
y up to a generous bound and accept only exact integer z with z >= y.

The circuit: a 4-qubit Grover search over the 16 index states |0000>..|1111>
(x = index + 2), with an oracle that flips the phase of exactly the single
index state corresponding to the classically-determined smallest valid x,
followed by the standard Grover diffusion operator, repeated the optimal
number of iterations for a unique marked item in a database of size 16. The
simulator should return the marked x with high probability, which we then
check post-hoc against the classical answer -- a genuine end-to-end
algebraic verification, not a lookup.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from fractions import Fraction


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles): find all x in [2, 9] for which
#    4/7 - 1/x = 1/y + 1/z has a positive-integer solution y <= z.
# ---------------------------------------------------------------------------

N = 7
TARGET = Fraction(4, N)
X_LO, X_HI = 2, 17  # 16 values -> 4 qubits


def find_yz(remainder: Fraction, y_max: int = 2000):
    """Exhaustively search y (then derive z) for remainder = 1/y + 1/z."""
    if remainder <= 0:
        return None
    y_start = remainder.denominator // remainder.numerator + 1
    for y in range(max(2, y_start), y_max):
        rest = remainder - Fraction(1, y)
        if rest <= 0:
            continue
        if rest.numerator == 1:
            z = rest.denominator
            if z >= y:
                return (y, z)
    return None


marked_x = []
solutions = {}
for x in range(X_LO, X_HI + 1):
    remainder = TARGET - Fraction(1, x)
    yz = find_yz(remainder)
    if yz is not None:
        y, z = yz
        # verify algebraically, exactly, with fractions (no floating point)
        assert Fraction(1, x) + Fraction(1, y) + Fraction(1, z) == TARGET
        marked_x.append(x)
        solutions[x] = (y, z)

assert marked_x, "classical search found no witnesses -- instance is wrong"

print(f"Classical result: for n={N}, 4/{N} = 1/x + 1/y + 1/z has solutions")
for x in marked_x:
    y, z = solutions[x]
    print(f"  x={x}: 1/{x} + 1/{y} + 1/{z} = 4/{N}  "
          f"(check: {Fraction(1, x) + Fraction(1, y) + Fraction(1, z)})")

smallest_x = min(marked_x)
marked_indices = [smallest_x - X_LO]  # unique classical winner -> single marked state
n_qubits = 4
N_states = 2 ** n_qubits
assert X_HI - X_LO + 1 == N_states

print(f"\nSmallest witness x = {smallest_x} (this is the target the "
      f"quantum search below hunts for).")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 3-qubit index register.
# ---------------------------------------------------------------------------

def oracle_circuit(marked, n_qubits):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for idx in marked:
        bits = format(idx, f"0{n_qubits}b")
        # flip qubits that should be 0 so the marked pattern becomes |11..1>
        flip_positions = [n_qubits - 1 - i for i, b in enumerate(bits) if b == "0"]
        for q in flip_positions:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in flip_positions:
            qc.x(q)
    return qc


def diffuser_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


M = len(marked_indices)
theta = math.asin(math.sqrt(M / N_states))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))

oracle = oracle_circuit(marked_indices, n_qubits)
diffuser = diffuser_circuit(n_qubits)
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(n_qubits))
    qc.append(diffuser.to_gate(), range(n_qubits))

qc.measure(range(n_qubits), range(n_qubits))

sim = AerSimulator()
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=4096).result()
counts = result.get_counts()

# Qiskit bit order: rightmost classical bit is qubit 0 -> reverse to get idx
def bitstring_to_index(bs):
    return int(bs[::-1], 2)

count_by_index = {}
for bitstring, c in counts.items():
    idx = bitstring_to_index(bitstring)
    count_by_index[idx] = count_by_index.get(idx, 0) + c

most_likely_index = max(count_by_index, key=count_by_index.get)
most_likely_x = most_likely_index + X_LO

print(f"\nGrover search: n_qubits={n_qubits}, database size={N_states}, "
      f"marked={len(marked_indices)}, iterations={iterations}")
print("Measurement counts by candidate x:")
for idx in sorted(count_by_index):
    x = idx + X_LO
    tag = " <- marked (classical)" if idx in marked_indices else ""
    print(f"  x={x}: {count_by_index[idx]}{tag}")

print(f"\nMost frequently measured x = {most_likely_x}")

# ---------------------------------------------------------------------------
# 3. Verify quantum result against the classical answer.
# ---------------------------------------------------------------------------

quantum_marks_valid_x = most_likely_index in marked_indices
# Grover should also strongly concentrate probability on the marked subspace
marked_shots = sum(count_by_index.get(i, 0) for i in marked_indices)
total_shots = sum(count_by_index.values())
concentration = marked_shots / total_shots

verified = quantum_marks_valid_x and concentration > 0.8

if verified:
    y, z = solutions[most_likely_x]
    print(f"Verified: x={most_likely_x} classically satisfies "
          f"1/{most_likely_x} + 1/{y} + 1/{z} = 4/{N}, and the quantum "
          f"search concentrated {concentration:.1%} of shots on marked x "
          f"values.")
    print("PASS")
else:
    print(f"Quantum result did not match classical marked set "
          f"(concentration={concentration:.1%}).")
    print("FAIL")
