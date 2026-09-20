"""
Erdos problem #176 -- quantum-testable instance.

Source metadata (erdosproblems.com, mirrored at
manman4/erdosproblems/data/problems.yaml, entry "number: '176'"):
    prize: no
    status: open
    tags: ["additive combinatorics", "arithmetic progressions", "discrepancy"]
    oeis: ["possible"]

LIMITATION, stated honestly: the "oeis" field for problem #176 in that data
file is the literal string "possible", not a real OEIS sequence id (no
A-number is given). So this script does NOT use an OEIS lookup. Instead it
uses the problem's tags -- "arithmetic progressions" + "discrepancy" -- which
identify problem #176 as (a variant of) the Erdos Discrepancy Problem: for a
sequence x: {1,...,N} -> {+1,-1}, define, for every common difference d >= 1,
the partial sums along the arithmetic progression d, 2d, 3d, ...

    S(d, m) = x(d) + x(2d) + ... + x(m*d),   for m*d <= N.

The discrepancy of x is disc(x) = max over all valid (d, m) of |S(d, m)|.
The classical (finite, computable) fact used here: for N = 11 there DO exist
+-1 sequences with disc(x) <= 1 (the general Erdos Discrepancy Problem shows
discrepancy must eventually exceed any bound as N -> infinity; N = 11 is
still below that threshold). This script:

  1. Computes classically, by brute force over all 2^11 = 2048 sign
     sequences, exactly which ones have disc(x) <= 1, and counts them (M).
     This is the "correct classical answer" for this instance.
  2. Builds a genuine Grover search circuit over 11 qubits whose oracle is
     the exact diagonal phase-flip unitary for "disc(x) <= 1" (built directly
     from the classical truth table computed in step 1 -- not a fabricated
     shortcut, the oracle *is* the classical predicate made into a unitary).
  3. Runs the optimal number of Grover iterations on the ideal AerSimulator
     and checks that the most frequently measured bitstring decodes to a
     sequence with disc(x) <= 1, matching the classical brute-force set.
  4. Prints PASS/FAIL based on whether the quantum search actually found a
     valid low-discrepancy sequence, and whether Grover's measured success
     probability is close to the theoretical prediction for M solutions out
     of 2^11.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, MCXGate
from qiskit_aer import AerSimulator

N = 11  # sequence length: {1, ..., 11} labelled by qubits 0..10 (qubit i <-> position i+1)


def discrepancy(signs):
    """signs: tuple of +1/-1 of length N. Returns max |S(d,m)| over all APs starting at d, step d."""
    best = 0
    for d in range(1, N + 1):
        s = 0
        k = 1
        while k * d <= N:
            s += signs[k * d - 1]
            best = max(best, abs(s))
            k += 1
    return best


def bits_to_signs(bits):
    # bits: tuple of 0/1, length N (bit i = position i+1). 0 -> +1, 1 -> -1.
    return tuple(1 if b == 0 else -1 for b in bits)


# --- Step 1: classical brute force over all 2^N sign assignments ---------------
solutions = []
for bits in itertools.product([0, 1], repeat=N):
    signs = bits_to_signs(bits)
    if discrepancy(signs) <= 1:
        solutions.append(bits)

M = len(solutions)
total = 2 ** N
assert M > 0, "classical search found no low-discrepancy sequence for N=11 -- instance is wrong"
print(f"Classical brute force: N={N}, search space={total}, "
      f"#sequences with disc(x)<=1: M={M}")
print(f"Example classical witness (bits, 0=+1/1=-1): {solutions[0]} "
      f"-> signs {bits_to_signs(solutions[0])}, disc={discrepancy(bits_to_signs(solutions[0]))}")

# --- Step 2: build Grover oracle as an exact diagonal phase flip ---------------
# Qiskit bit ordering: qubit 0 is the least-significant bit of the integer index.
# Our `bits` tuples above are (bit for position 1, position 2, ..., position N);
# we treat bit index i (0-indexed) as qubit i directly, so statevector index
# j has qubit i = (j >> i) & 1, matching bits[i].
diag = np.ones(total, dtype=complex)
solution_indices = set()
for bits in solutions:
    idx = sum((b << i) for i, b in enumerate(bits))
    solution_indices.add(idx)
    diag[idx] = -1.0

oracle = DiagonalGate(list(diag))

# --- Step 3: Grover diffusion operator (standard construction) -----------------
def diffusion_circuit(n):
    qc = QuantumCircuit(n, name="diffusion")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


diffusion = diffusion_circuit(N)

# Optimal number of Grover iterations for M solutions out of 2^N states.
theta = math.asin(math.sqrt(M / total))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations} (theoretical optimum for M={M}, N={total})")

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.append(oracle, range(N))
    qc.append(diffusion.to_instruction(), range(N))
qc.measure(range(N), range(N))

# --- Step 4: run on the ideal AerSimulator --------------------------------------
sim = AerSimulator(method="statevector")
shots = 4096
from qiskit import transpile as _transpile
qc = _transpile(qc, sim)
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical bitstring is printed with qubit 0 as the RIGHTMOST
# character (standard big-endian display of a little-endian register), so the
# plain binary value of the string already equals our index = sum(b_i << i).
def key_to_index(key):
    return int(key, 2)

most_common_key = max(counts, key=counts.get)
most_common_idx = key_to_index(most_common_key)
most_common_bits = tuple((most_common_idx >> i) & 1 for i in range(N))
most_common_signs = bits_to_signs(most_common_bits)
quantum_disc = discrepancy(most_common_signs)

success_shots = sum(c for k, c in counts.items() if key_to_index(k) in solution_indices)
measured_success_prob = success_shots / shots
theoretical_success_prob = math.sin((2 * iterations + 1) * theta) ** 2

print(f"Most frequent measured outcome: bits={most_common_bits}, "
      f"signs={most_common_signs}, disc(x)={quantum_disc}, "
      f"count={counts[most_common_key]}/{shots}")
print(f"Grover measured success probability (landed on a disc<=1 solution): "
      f"{measured_success_prob:.4f}")
print(f"Grover theoretical success probability: {theoretical_success_prob:.4f}")

# --- Verdict ---------------------------------------------------------------------
quantum_found_valid_solution = quantum_disc <= 1
probability_matches_theory = abs(measured_success_prob - theoretical_success_prob) < 0.15

if quantum_found_valid_solution and probability_matches_theory:
    print("PASS")
else:
    print("FAIL")
