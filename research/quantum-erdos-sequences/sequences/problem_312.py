"""
Erdos problem #312 (erdosproblems.com), tags: ["number theory", "unit fractions"].

Source metadata for problem 312, as recorded in erdosproblems/data/problems.yaml:
    oeis: ["N/A"]
    tags: ["number theory", "unit fractions"]
There is no OEIS sequence id attached to problem 312 (oeis: "N/A"), so there is no
canonical integer sequence to build a membership/term-search circuit against for
*this specific* problem. Per the task's fallback instructions, this script makes an
honest, small, genuinely computable "unit fractions" instance in the same spirit as
problem 312's subject matter (representing a unit fraction 1/n as a sum of two
*distinct* unit fractions with bounded denominators), rather than fabricating or
borrowing an unrelated OEIS value.

Classical property being tested
--------------------------------
For n = 6 and denominator bound B = 16, does there exist a pair of distinct
integers (a, b) with 1 <= a, b <= B, a != b, such that

    1/a + 1/b = 1/n

The script enumerates all pairs classically (first principles, no OEIS lookup) to
find every solution in range. Using the standard substitution (a-n)(b-n) = n^2,
the divisor pairs of n^2 = 36 give the classical solutions with a,b <= 16:
    (a, b) = (10, 15) and (15, 10)      [1/10 + 1/15 = 3/30 + 2/30 = 5/30 = 1/6]
(The other divisor pairs of 36, e.g. (7,42), (8,24), (9,18), (12,12), are excluded
because either a denominator exceeds B=16 or a == b, which the search also
requires to exclude by construction.)

Quantum circuit
----------------
This is treated as an unstructured search problem over the N = 256 possible pairs
(a, b) in {1,...,16} x {1,...,16}, encoded as an 8-qubit register (4 qubits for
a-1, 4 qubits for b-1). Grover's algorithm is used to amplify the two marked
basis states corresponding to the classically-verified solutions (10,15) and
(15,10). The oracle is built directly from the classical solution set found by
brute-force search in this script (a standard construction for demonstrating
Grover search on a function whose marked set has already been computed/verified
classically) — it is not a hardcoded OEIS value, it is the output of the
classical solver run in this file. The oracle and diffuser are implemented with
elementary Qiskit gates (multi-controlled Z via mcx + H sandwich) on the ideal
AerSimulator (statevector method), with the optimal number of Grover iterations
computed from N and the number of marked states.

Pass/fail
---------
The script classically computes the true solution set, runs the Grover circuit,
and checks that the most frequently measured bitstring decodes to a pair (a, b)
that is actually in the classically-verified solution set. PASS/FAIL is printed
based on that comparison.
"""

import itertools
import sys

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles) of the unit-fraction property.
# ---------------------------------------------------------------------------

N_TARGET = 6      # n: we test 1/n = 1/a + 1/b
BOUND = 16        # denominators a, b range over 1..BOUND
NUM_BITS = 4       # bits needed to index 0..BOUND-1 (BOUND=16 -> 4 bits)
assert BOUND == 2 ** NUM_BITS


def classical_solutions(n: int, bound: int):
    """Brute-force all (a, b) with 1<=a,b<=bound, a!=b, 1/a+1/b == 1/n exactly."""
    sols = []
    for a, b in itertools.product(range(1, bound + 1), repeat=2):
        if a == b:
            continue
        # exact rational check via cross multiplication: n*(a+b) == a*b
        if n * (a + b) == a * b:
            sols.append((a, b))
    return sols


solutions = classical_solutions(N_TARGET, BOUND)
if not solutions:
    print(f"No solutions found classically for 1/{N_TARGET} = 1/a + 1/b with "
          f"a,b <= {BOUND}; cannot build a Grover instance.")
    sys.exit(1)

print(f"Classical solutions for 1/{N_TARGET} = 1/a + 1/b, 1<=a,b<={BOUND}, a!=b:")
for a, b in solutions:
    print(f"  (a={a}, b={b})  check: 1/{a} + 1/{b} = {1/a + 1/b:.6f}  vs 1/{N_TARGET} = {1/N_TARGET:.6f}")

# Encode each solution (a, b) as an 8-bit index: high 4 bits = a-1, low 4 bits = b-1
# (bit ordering must match how the quantum register is built below).
marked_indices = set()
for a, b in solutions:
    idx = ((a - 1) << NUM_BITS) | (b - 1)
    marked_indices.add(idx)

marked_bitstrings = {format(idx, f"0{2 * NUM_BITS}b") for idx in marked_indices}
print(f"Marked basis states (a_bits b_bits, MSB first): {sorted(marked_bitstrings)}")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle + diffuser for these marked states.
# ---------------------------------------------------------------------------

NUM_QUBITS = 2 * NUM_BITS  # 8 qubits: [a3 a2 a1 a0 b3 b2 b1 b0] in circuit order


def add_oracle(qc: QuantumCircuit, bitstring: str):
    """Flip the phase of the single basis state matching `bitstring`.

    `bitstring` is MSB-first over qubits [q_{n-1}, ..., q_0] as usually printed;
    here we index qubit i (0 = least significant) directly against the string.
    """
    n = len(bitstring)
    # bitstring[0] corresponds to the most significant qubit -> qubit (n-1)
    zero_qubits = [n - 1 - i for i, c in enumerate(bitstring) if c == "0"]
    qc.x(zero_qubits)
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    qc.x(zero_qubits)


def add_diffuser(qc: QuantumCircuit):
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))


num_states = 2 ** NUM_QUBITS
num_marked = len(marked_bitstrings)
# Optimal number of Grover iterations for M marked out of N states.
optimal_iters = max(1, round((np.pi / 4) * np.sqrt(num_states / num_marked)))
print(f"N={num_states} states, M={num_marked} marked, Grover iterations={optimal_iters}")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

for _ in range(optimal_iters):
    for bs in marked_bitstrings:
        add_oracle(qc, bs)
    add_diffuser(qc)

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

sim = AerSimulator(method="statevector")
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit prints bitstrings MSB-first over the classical register, matching our
# qubit convention (qubit NUM_QUBITS-1 is the most significant printed bit).
best_bitstring = max(counts, key=counts.get)
best_count = counts[best_bitstring]
print(f"Most frequent measured bitstring: {best_bitstring} "
      f"({best_count}/{shots} shots, {100 * best_count / shots:.1f}%)")

decoded_a = int(best_bitstring[:NUM_BITS], 2) + 1
decoded_b = int(best_bitstring[NUM_BITS:], 2) + 1
print(f"Decoded pair: a={decoded_a}, b={decoded_b}")

is_valid_classically = (N_TARGET * (decoded_a + decoded_b) == decoded_a * decoded_b) and decoded_a != decoded_b
quantum_found_marked_state = best_bitstring in marked_bitstrings

print(f"Decoded pair satisfies 1/{N_TARGET} = 1/a + 1/b exactly (classical check): {is_valid_classically}")
print(f"Decoded pair is among the classically pre-computed marked states: {quantum_found_marked_state}")

# Also check total probability mass Grover placed on the marked subspace, as a
# sanity measure that the amplification genuinely worked (not just luck on the
# single most-frequent outcome).
marked_shots = sum(counts.get(bs, 0) for bs in marked_bitstrings)
marked_fraction = marked_shots / shots
print(f"Total shot fraction landing on marked states: {marked_fraction:.4f} "
      f"({marked_shots}/{shots})")

PASS = is_valid_classically and quantum_found_marked_state and marked_fraction > 0.5

if PASS:
    print("PASS")
else:
    print("FAIL")
    sys.exit(1)
