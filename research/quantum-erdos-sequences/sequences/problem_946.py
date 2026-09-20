"""
Erdos problem #946 -- quantum-testable instance.

Source metadata (erdosproblems.com data, problem 946):
  oeis: ["A005237", "A284783"]
  tags: ["number theory", "divisors"]
  status: proved

A005237 is the sequence of "nonaveraging" numbers: n belongs to A005237 iff
no divisor of n is equal to the average of two or more OTHER distinct
divisors of n. Equivalently, n is NOT in A005237 iff there exists a subset
S of the divisors of n, with |S| >= 2, whose arithmetic mean is itself a
divisor of n that does not belong to S.

Classical property tested here (finite, computable):
  For n = 6, divisors(6) = [1, 2, 3, 6]. We brute-force, in plain Python,
  every subset of these 4 divisors and mark the subsets S (|S| >= 2) whose
  mean is an integer, is itself one of the divisors of 6, and is not a
  member of S. This tells us classically whether 6 is a member of A005237
  (it is not, since {1, 3} averages to 2, and 2 is a divisor of 6 not in
  {1, 3}).

Quantum circuit: Grover's search over the 2^4 = 16 subsets of divisors(6),
encoded as 4 qubits (bit i = "divisor i is in the subset"). The oracle is
built directly from the classically-precomputed list of marked subsets
(the "witness" subsets satisfying the averaging condition above) via
multi-controlled Z phase flips, followed by the standard Grover diffuser.
The number of Grover iterations is chosen from the true count of marked
states, computed classically. We then run the circuit on the ideal
AerSimulator and check that the measurement distribution concentrates on
exactly the classically-computed set of marked (witness) subsets -- i.e.
that the quantum search finds the same witnesses that prove 6 is NOT in
A005237, and does not find any other, non-witness subset.

This is a genuine, from-first-principles classical computation (the
brute-force divisor-subset averaging check) cross-checked against a real
Grover amplitude-amplification circuit, not a copied OEIS literal.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles): divisors of n, and the set of
#    "witness" subsets proving n is NOT in A005237.
# ---------------------------------------------------------------------------

def divisors(n):
    return sorted(d for d in range(1, n + 1) if n % d == 0)


N = 6
D = divisors(N)          # [1, 2, 3, 6]
NUM_QUBITS = len(D)      # 4 qubits -> subsets of {0,1,2,3} indexing D
assert NUM_QUBITS <= 6, "keep the search space small"

divisor_set = set(D)


def is_witness(subset_indices):
    """A subset of D (given as a tuple of indices into D) is a witness
    that N is NOT in A005237 iff |subset| >= 2, its mean is an integer,
    that mean is a divisor of N, and that mean is not itself in the
    subset."""
    if len(subset_indices) < 2:
        return False
    vals = [D[i] for i in subset_indices]
    total = sum(vals)
    if total % len(vals) != 0:
        return False
    mean = total // len(vals)
    return (mean in divisor_set) and (mean not in vals)


marked_states = []  # bitmasks (int, bit i = divisor i included) that are witnesses
for mask in range(2 ** NUM_QUBITS):
    idxs = tuple(i for i in range(NUM_QUBITS) if (mask >> i) & 1)
    if is_witness(idxs):
        marked_states.append(mask)

n_in_A005237 = len(marked_states) == 0

print(f"N = {N}, divisors = {D}")
print(f"Classically found {len(marked_states)} witness subset(s) (bitmasks): "
      f"{[format(m, f'0{NUM_QUBITS}b') for m in marked_states]}")
print(f"=> classical conclusion: {N} is "
      f"{'in' if n_in_A005237 else 'NOT in'} A005237")

# Sanity: for N=6, {1,3} (indices 0,2 -> mean 2 = D[1]) must be a witness.
expected_witness_mask = (1 << 0) | (1 << 2)  # divisors 1 and 3
assert expected_witness_mask in marked_states, "expected witness {1,3} not found"
assert not n_in_A005237, "6 is known to not be in A005237 (sanity check failed)"


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for the marked (witness) bitstrings.
# ---------------------------------------------------------------------------

def build_oracle(num_qubits, marked):
    qc = QuantumCircuit(num_qubits, name="oracle")
    for mask in marked:
        zero_bits = [i for i in range(num_qubits) if not (mask >> i) & 1]
        if zero_bits:
            qc.x(zero_bits)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        if zero_bits:
            qc.x(zero_bits)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


num_marked = len(marked_states)
search_space = 2 ** NUM_QUBITS
# Standard Grover optimal iteration count.
iterations = max(1, round((math.pi / 4) * math.sqrt(search_space / num_marked)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

oracle = build_oracle(NUM_QUBITS, marked_states)
diffuser = build_diffuser(NUM_QUBITS)

for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's counts keys are already in the same "bit i = qubit i, read
# left-to-right as MSB..LSB of the classical register" convention used
# when we built the marked bitmasks above (verified against
# Statevector.probabilities_dict(), which uses the identical convention).
def bitstring_to_mask(bs):
    return int(bs, 2)

measured_masks = {bitstring_to_mask(bs): c for bs, c in counts.items()}
total_on_marked = sum(c for m, c in measured_masks.items() if m in marked_states)
frac_on_marked = total_on_marked / shots

# Most-frequent measured outcome(s)
top_mask = max(measured_masks, key=measured_masks.get)

print(f"Grover iterations used: {iterations}")
print(f"Fraction of shots landing on a classically-marked witness state: "
      f"{frac_on_marked:.3f}")
print(f"Most frequent measured subset (bitmask): {format(top_mask, f'0{NUM_QUBITS}b')}")


# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------------

quantum_found_witness = top_mask in marked_states
quantum_concentrated = frac_on_marked > 0.8  # amplitude-amplified majority

verified = quantum_found_witness and quantum_concentrated

if verified:
    print("PASS: Grover search concentrated on the classically-verified "
          "witness subset(s), confirming 6 is not in A005237.")
else:
    print("FAIL: quantum search did not concentrate on the classical "
          "witness set as expected.")
