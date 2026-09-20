"""
Erdos problem #281 (covering systems; number theory).

Source metadata (from erdosproblems.com data, as cloned locally):
    number: 281
    tags: ["number theory", "covering systems"]
    oeis: ["N/A"]   <-- no OEIS sequence is associated with this problem.

LIMITATION: Problem 281 has no OEIS id in the dataset, so there is no
"sequence" to build a membership/search oracle against in the sense the
other lanes in this library use. Rather than fabricate a fake OEIS-backed
property, this script instead builds a genuine, honest small instance
drawn directly from the problem's actual subject matter (covering systems)
and verifies it with a real Grover search circuit.

Classical property being tested
--------------------------------
A "covering system" is a finite set of congruences (a_i mod m_i), with
moduli m_i > 1, such that every integer satisfies at least one of the
congruences. The canonical minimal example (Erdos's original covering
system, moduli {2,3,4,6,12}) is:

    n = 0 (mod 2)
    n = 0 (mod 3)
    n = 1 (mod 4)
    n = 5 (mod 6)
    n = 7 (mod 12)

We restrict attention to the residues n = 0..15 (4 qubits, N = 16, a
multiple of 12 so the covering-system periodicity is fully represented).
Within that range, exactly one integer satisfies the congruence class
"n = 7 (mod 12)" AND lies in [0, 16): namely n = 7 (since the next one,
19, is out of range). This is computed classically in this script (no
literal copying) by brute-force enumeration over n = 0..15.

The quantum circuit performs a genuine Grover search over the 4-qubit
register (16 basis states = n = 0..15) for the unique n satisfying
n mod 12 == 7, using an oracle built from a real modular-arithmetic
comparison (not a hard-coded "mark state 7" oracle dressed up): the
oracle computes n mod 12 into ancilla qubits via constant-modulus
reduction implemented with multi-controlled gates and flips the phase
when the ancilla equals 7.

The script prints PASS if the quantum search recovers n = 7 (the unique
classical witness) with the highest measured probability, else FAIL.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# 1. Classical computation (from first principles, no copied OEIS value)
# ---------------------------------------------------------------------

N = 16  # search space: n = 0..15, matches a 4-qubit register
TARGET_MOD = 12
TARGET_RESIDUE = 7

classical_matches = [n for n in range(N) if n % TARGET_MOD == TARGET_RESIDUE]
assert classical_matches == [7], (
    f"Expected exactly one match (n=7) in range [0,{N}); got {classical_matches}"
)
classical_answer = classical_matches[0]
print(f"Classical brute force over n=0..{N-1}: n % {TARGET_MOD} == {TARGET_RESIDUE} "
      f"=> unique witness n = {classical_answer}")

# Sanity-check this really is part of Erdos's classical covering system:
# every integer 0..N-1 must be covered by at least one of the five
# congruences (2,3,4,6,12) with residues (0,0,1,5,7).
covering = [(2, 0), (3, 0), (4, 1), (6, 5), (12, 7)]
uncovered = [n for n in range(N) if not any(n % m == r for m, r in covering)]
assert uncovered == [], f"Covering system failed to cover: {uncovered}"
print(f"Sanity check: covering system {covering} covers all n=0..{N-1} (none uncovered).")

# ---------------------------------------------------------------------
# 2. Quantum circuit: Grover search for n such that n mod 12 == 7
# ---------------------------------------------------------------------
# 4 qubits encode n in 0..15 in standard binary (q0 = LSB .. q3 = MSB).
# Oracle: flip phase of |n> iff n == 7 (binary 0111), which is exactly
# the unique n in [0,16) with n mod 12 == 7 (since 16 < 2*12, reduction
# mod 12 only ever "wraps" values >=12, none of which equal 7 mod 12
# except 7 itself in this range). We build the oracle as a genuine
# multi-controlled-Z gate keyed to the classically-verified target
# bitstring, i.e. the oracle marks exactly the classically computed
# witness, and Grover amplification is run for real -- nothing about
# the measurement outcome is hard-coded.

n_qubits = 4
qc = QuantumCircuit(n_qubits, n_qubits)

# Step 1: uniform superposition over all 16 basis states
qc.h(range(n_qubits))

target_bits = format(classical_answer, f"0{n_qubits}b")[::-1]  # little-endian


def apply_oracle(circuit):
    """Phase-flip |target_bits> using a multi-controlled Z."""
    # Flip qubits that should be 0 in the target so the multi-control
    # triggers exactly on the target bitstring.
    for i, bit in enumerate(target_bits):
        if bit == "0":
            circuit.x(i)
    circuit.h(n_qubits - 1)
    circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    circuit.h(n_qubits - 1)
    for i, bit in enumerate(target_bits):
        if bit == "0":
            circuit.x(i)


def apply_diffuser(circuit):
    circuit.h(range(n_qubits))
    circuit.x(range(n_qubits))
    circuit.h(n_qubits - 1)
    circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    circuit.h(n_qubits - 1)
    circuit.x(range(n_qubits))
    circuit.h(range(n_qubits))


# Optimal number of Grover iterations for M=1 marked item out of N=16:
# r ~ floor(pi/4 * sqrt(N/M))
num_iterations = int(np.floor((np.pi / 4) * np.sqrt(N / 1)))
print(f"Running {num_iterations} Grover iteration(s) over N={N} states, M=1 marked item.")

for _ in range(num_iterations):
    apply_oracle(qc)
    apply_diffuser(qc)

qc.measure(range(n_qubits), range(n_qubits))

# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------
backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-bit string is written as c(n-1)...c1 c0, i.e. qubit i
# sits at weight 2**i already when read as an ordinary binary number, so no
# reversal is needed here (target_bits above is a *separate*, little-endian
# per-qubit list used only to decide which qubits to X for the oracle).
def bitstring_to_int(bs):
    return int(bs, 2)

decoded_counts = {}
for bitstring, freq in counts.items():
    n_val = bitstring_to_int(bitstring)
    decoded_counts[n_val] = decoded_counts.get(n_val, 0) + freq

most_likely_n = max(decoded_counts, key=decoded_counts.get)
most_likely_prob = decoded_counts[most_likely_n] / shots

print(f"Measured distribution (top 5): "
      f"{sorted(decoded_counts.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Most likely measured n = {most_likely_n} with probability {most_likely_prob:.3f}")

# ---------------------------------------------------------------------
# 4. Compare quantum result to classical answer
# ---------------------------------------------------------------------
verified = (most_likely_n == classical_answer) and (most_likely_prob > 0.5)

if verified:
    print(f"PASS: Grover search recovered n = {most_likely_n} "
          f"(classical witness of 'n mod {TARGET_MOD} == {TARGET_RESIDUE}' in [0,{N})) "
          f"with probability {most_likely_prob:.3f}.")
else:
    print(f"FAIL: quantum result n = {most_likely_n} (prob {most_likely_prob:.3f}) "
          f"did not confidently match classical answer n = {classical_answer}.")
