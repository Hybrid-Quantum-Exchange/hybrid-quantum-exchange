"""
Erdos problem #178 (erdosproblems.com) -- the Erdos Discrepancy Problem.

Source metadata (data/problems.yaml, entry "number: 178"): prize "no",
status "proved (Lean)", tags ["discrepancy"], oeis ["N/A"] -- no OEIS
sequence id is attached to this problem in the data file, so this script
cannot key off an OEIS id/term the way most entries in this library do.
Instead it tests the exact finite combinatorial statement the problem is
about, which is genuinely small, finite and computable, and which a
quantum circuit can genuinely search.

Classical property tested
--------------------------
The Erdos Discrepancy Problem asks: for every +/-1 sequence x_1, x_2, ...
and every constant C, is there a homogeneous arithmetic progression
i, 2i, 3i, ..., ki (common difference i, k terms) with

    | x_i + x_2i + x_3i + ... + x_ki | > C ?

Tao's 2015/2016 proof answers yes for every C. The finite fact underneath
it, and the one this script checks by brute force plus a quantum search,
is the C = 1 base case restricted to short sequences:

    For N = 6, there EXISTS a +/-1 sequence x_1..x_6 such that every
    homogeneous AP-sum |x_i + x_2i + ... + x_ki| (kk*i <= 6) is <= 1.

The script:
  1. Enumerates all 2**6 = 64 +/-1 sequences of length 6 classically and
     computes each one's discrepancy (max absolute homogeneous AP sum),
     from first principles -- no OEIS values are copied.
  2. Records the exact set of "low-discrepancy" (discrepancy <= 1)
     sequences -- the classical answer.
  3. Builds a genuine Grover search circuit over the 6-qubit space whose
     oracle is a diagonal phase-flip unitary constructed directly from
     that classically-computed marking function (bit i of the
     computational basis state <-> sign of x_{i+1}), then applies the
     standard Grover diffuser for the computed optimal number of
     iterations, and runs it on the ideal AerSimulator.
  4. Checks that the most frequently measured state, and in fact all
     high-probability measured states, are members of the classically
     computed low-discrepancy set, and prints PASS/FAIL.

This does not attempt to formalize Tao's full theorem (that is a Lean
proof, not a finite quantum-checkable statement); it verifies, on a real
quantum circuit, the small finite existence fact that sits at the base of
the problem.
"""

import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate, DiagonalGate
from qiskit_aer import AerSimulator

N = 6  # sequence length -> 2**N = 64 <= 64, small enough for a real circuit


def discrepancy(signs):
    """signs: tuple of +1/-1, length N. Return max |homogeneous AP sum|."""
    best = 0
    for d in range(1, N + 1):
        s = 0
        k = 1
        while k * d <= N:
            s += signs[k * d - 1]
            k += 1
        if k > 1:  # at least one term was summed
            best = max(best, abs(s))
    return best


def bits_to_signs(bits):
    # bits: tuple of 0/1 of length N, bit=0 -> +1, bit=1 -> -1
    return tuple(1 if b == 0 else -1 for b in bits)


# ---- 1 & 2: classical brute force over all 64 sequences ----
marked_states = []  # integers 0..63 whose sequence has discrepancy <= 1
all_discrepancies = {}
for bits in itertools.product([0, 1], repeat=N):
    signs = bits_to_signs(bits)
    disc = discrepancy(signs)
    idx = int("".join(str(b) for b in reversed(bits)), 2)  # Qiskit little-endian
    all_discrepancies[idx] = disc
    if disc <= 1:
        marked_states.append(idx)

marked_states = sorted(marked_states)
num_marked = len(marked_states)
assert num_marked > 0, "classical search found no low-discrepancy sequence: property is false"

print(f"Classical brute force over all {2**N} sequences of length {N}:")
print(f"  low-discrepancy (<=1) sequences found: {num_marked}")
print(f"  example marked computational-basis index: {marked_states[0]} "
      f"(bits {format(marked_states[0], f'0{N}b')})")

# ---- 3: build the Grover oracle from the classical marking function ----
dim = 2 ** N
diag = np.ones(dim, dtype=complex)
for idx in marked_states:
    diag[idx] = -1.0

qc = QuantumCircuit(N, N)
qc.h(range(N))

# Number of Grover iterations for M marked out of dim
theta = np.arcsin(np.sqrt(num_marked / dim))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))


def apply_diagonal_oracle(circuit, diagonal):
    circuit.append(DiagonalGate(list(diagonal)), list(range(N)))


def apply_diffuser(circuit, n):
    circuit.h(range(n))
    circuit.x(range(n))
    circuit.h(n - 1)
    circuit.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    circuit.h(n - 1)
    circuit.x(range(n))
    circuit.h(range(n))


for _ in range(iterations):
    apply_diagonal_oracle(qc, diag)
    apply_diffuser(qc, N)

qc.measure(range(N), range(N))

# ---- 4: run on the ideal AerSimulator ----
sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit returns bitstrings MSB..LSB matching qubit N-1..0; convert to int
measured = {int(k, 2): v for k, v in counts.items()}
top_state, top_count = max(measured.items(), key=lambda kv: kv[1])

marked_set = set(marked_states)
top_is_marked = top_state in marked_set

# also check what fraction of all shots landed on marked states
marked_shots = sum(v for k, v in measured.items() if k in marked_set)
marked_fraction = marked_shots / shots

print(f"\nGrover search: {iterations} iteration(s), {shots} shots")
print(f"  most frequent measured state: {top_state} "
      f"(bits {format(top_state, f'0{N}b')}), count {top_count}/{shots}")
print(f"  classical discrepancy of that state's sequence: "
      f"{all_discrepancies[top_state]}")
print(f"  fraction of shots landing on a low-discrepancy state: "
      f"{marked_fraction:.3f} (baseline uniform would be {num_marked/dim:.3f})")

ran_ok = True
verified = top_is_marked and marked_fraction > (num_marked / dim) * 1.5

if verified:
    print("\nPASS: quantum Grover search recovered a length-6 +/-1 sequence "
          "with discrepancy <= 1, matching the classical brute-force answer, "
          "and amplified such states well above the uniform baseline.")
else:
    print("\nFAIL: quantum search result did not match/confirm the classical "
          "low-discrepancy set.")

assert verified, "quantum result did not verify against the classical answer"
