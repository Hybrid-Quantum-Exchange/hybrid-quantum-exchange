"""
Erdos problem #7 -- quantum-testable instance.

Source: erdosproblems.com problem 7 (data/problems.yaml entry `number: "7"`),
tags ["number theory", "covering systems"]. The `oeis` field for this entry
is ["N/A"] -- problem 7 has no associated OEIS sequence id in the source
data, so no OEIS id is used here. (Checked directly against
/home/user/manman4/erdosproblems/data/problems.yaml lines 100-114.)

Because there is no OEIS sequence to draw a property from, this script
instead builds a genuine finite, computable instance from the problem's own
subject matter -- covering systems of congruences -- which is exactly what
the "covering systems" tag names.

Classical property tested
--------------------------
A classical example of a covering system (a finite set of congruences
a_i (mod m_i) with distinct moduli m_i such that every integer satisfies at
least one of them) is:

    0 (mod 2), 0 (mod 3), 1 (mod 4), 5 (mod 6), 7 (mod 12)

This is the standard textbook covering system (moduli {2,3,4,6,12}, period
lcm = 12). The script first verifies classically, by brute-force residue
checking, that this system covers every integer in a finite window
n = 0..63 (a 6-qubit index register), i.e. it computes -- from first
principles, no table lookup -- the set of n in that window landing in the
single congruence class 7 (mod 12), which is the *last* congruence of the
covering system and the one that closes the cover (without it, integers
n = 7 mod 12 would be missed by the other four classes).

The finite, computable search problem posed to the quantum circuit is:

    find every index n in {0, ..., 63} with n === 7 (mod 12)

This is computed classically first (direct modular arithmetic on the 64
candidates), giving the ground-truth marked set. A Grover search oracle is
then built mechanically from that classically computed set (not hand-picked)
and run on the ideal AerSimulator with 6 search qubits. The number of Grover
iterations is chosen from the classically known count of marked items. The
test passes if the quantum circuit's high-probability measurement outcomes
are exactly the classically computed marked index set, and additionally that
the classical brute-force check confirms the covering system leaves no
integer in 0..63 uncovered (the actual covering-system property from problem
7's tag), which is reported alongside the Grover result.

This is a genuine amplitude-amplification search (Grover) over an
unstructured 6-bit index space for the positions in a real number-theoretic
condition (a specific residue class of a covering system), with the oracle
derived mechanically from the classical computation.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---- Classical ground truth -------------------------------------------------

# The covering system: list of (residue, modulus) pairs.
COVERING_SYSTEM = [(0, 2), (0, 3), (1, 4), (5, 6), (7, 12)]

NUM_QUBITS = 6
N = 2 ** NUM_QUBITS  # 64 indices, 6-qubit search register


def is_covered(n, system):
    """True if integer n satisfies at least one congruence in `system`."""
    return any(n % m == a for a, m in system)


# Verify, from first principles, that the covering system actually covers
# every integer in the test window (the real "covering systems" property).
uncovered = [n for n in range(N) if not is_covered(n, COVERING_SYSTEM)]
print(f"Covering system: {COVERING_SYSTEM}")
print(f"Integers 0..{N - 1} left uncovered by the system: {uncovered}")
covering_system_verified = len(uncovered) == 0
print(f"Covering-system property holds on 0..{N - 1}: {covering_system_verified}")

# The finite search problem handed to Grover: find all n in 0..63 with
# n === 7 (mod 12), the closing congruence of the covering system.
TARGET_RESIDUE, TARGET_MODULUS = 7, 12
marked_indices = sorted(n for n in range(N) if n % TARGET_MODULUS == TARGET_RESIDUE)

print(f"Classically marked indices where n mod {TARGET_MODULUS} == {TARGET_RESIDUE}: "
      f"{marked_indices}")

M = len(marked_indices)
assert 0 < M < N, "Grover search needs a nontrivial, proper subset marked"


# ---- Grover oracle built mechanically from the classical marked set --------

def apply_multi_controlled_z_on_bits(qc, qubits):
    """Apply a Z controlled on all qubits being |1> (multi-controlled Z)."""
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def oracle(qc, index, qubits):
    """Flip the phase of basis state |index> (little-endian bit order)."""
    bits = format(index, f"0{len(qubits)}b")[::-1]  # bit i -> qubits[i]
    flip_qubits = [q for q, b in zip(qubits, bits) if b == "0"]
    if flip_qubits:
        qc.x(flip_qubits)
    apply_multi_controlled_z_on_bits(qc, qubits)
    if flip_qubits:
        qc.x(flip_qubits)


def diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    apply_multi_controlled_z_on_bits(qc, qubits)
    qc.x(qubits)
    qc.h(qubits)


qubits = list(range(NUM_QUBITS))
qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(qubits)

# Standard optimal iteration count for M marked items out of N.
iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))
print(f"Grover iterations used: {iterations} (N={N}, M={M})")

for _ in range(iterations):
    for idx in marked_indices:
        oracle(qc, idx, qubits)
    diffuser(qc, qubits)

qc.measure(qubits, qubits)

# ---- Run on the ideal AerSimulator ------------------------------------------

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-bit string already has bit 0 as its rightmost
# character, i.e. the same convention as int(bitstring, 2).
counted = {}
for bitstring, c in counts.items():
    idx = int(bitstring, 2)
    counted[idx] = counted.get(idx, 0) + c

sorted_counts = sorted(counted.items(), key=lambda kv: -kv[1])
print("Measurement counts by index (most frequent first, top 10 shown):")
for idx, c in sorted_counts[:10]:
    marker = " <- classically marked" if idx in marked_indices else ""
    print(f"  index {idx}: {c}/{shots}{marker}")

# The top-M most measured outcomes should be exactly the classically marked set.
top_m_indices = set(idx for idx, _ in sorted_counts[:M])
quantum_marked = top_m_indices

grover_verified = quantum_marked == set(marked_indices)

print()
print(f"Classical marked set : {sorted(marked_indices)}")
print(f"Quantum-found set    : {sorted(quantum_marked)}")
print(f"Covering-system property (classical, all 64 covered): {covering_system_verified}")

verified = grover_verified and covering_system_verified

if verified:
    print("PASS")
else:
    print("FAIL")
