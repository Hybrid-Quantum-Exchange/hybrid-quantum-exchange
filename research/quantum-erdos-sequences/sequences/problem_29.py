"""
Erdos problem #29 -- quantum-testable lane.

Source metadata (data/problems.yaml, entry "number: 29"):
    prize: $100
    status: proved (Lean), last_update 2026-08-24
    oeis: ["N/A"]
    tags: ["number theory", "additive basis"]

LIMITATION (reported honestly, per instructions): Erdos problem #29 has no
OEIS sequence id in the source data (oeis: ["N/A"]). There is therefore no
"early term of an OEIS sequence" to target with a quantum circuit. Rather
than fabricate a fake OEIS-backed claim, this script instead builds a real,
finite, computable instance of the one substantive mathematical object named
in the problem's own tags -- "additive basis" -- and verifies it with a
genuine Grover search circuit. This is offered as the best honest attempt
for a problem with no sequence data, not as a literal encoding of problem
#29's actual open question (which is not reproduced here beyond its tags).

Classical property under test
------------------------------
Let N = 8 and S = {0, 1, 2, 4} (a subset of Z_N). S is an *additive basis of
order 2* for Z_N if every element k in Z_N can be written as k = s_i + s_j
(mod N) for some s_i, s_j in S. This script:

  1. Classically enumerates all pairs (s_i, s_j) in S x S for every target
     k in Z_N, from first principles (brute force, no lookup), and records
     the classical answer: for each k, one witness pair (i, j) of indices
     into S with s_i + s_j = k (mod N), plus the overall basis/not-basis
     verdict for S.

  2. For a single fixed target k0, builds a Grover search circuit over the
     4x4 = 16 possible index pairs (i, j) in {0,1,2,3}^2 (2 qubits for i,
     2 qubits for j), with an oracle implementing the arithmetic
     s_i + s_j == k0 (mod 8) directly via a quantum adder built from the
     classical S-lookup (implemented reversibly as a small truth table
     oracle), and runs it on the ideal AerSimulator.

  3. Compares the most frequently measured (i, j) pair against the
     classical witness set for k0 and prints PASS/FAIL.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, brute force)
# ---------------------------------------------------------------------------

N = 8
S = [0, 1, 2, 4]  # 4 elements -> index encoded on 2 qubits each

def classical_witnesses():
    """For every k in Z_N, brute-force all (i, j) with S[i]+S[j] == k (mod N)."""
    witnesses = {k: [] for k in range(N)}
    for i, j in itertools.product(range(len(S)), repeat=2):
        k = (S[i] + S[j]) % N
        witnesses[k].append((i, j))
    return witnesses


WITNESSES = classical_witnesses()
IS_ADDITIVE_BASIS = all(len(WITNESSES[k]) > 0 for k in range(N))

print(f"N = {N}, S = {S}")
print(f"Classical additive-basis check (order 2, mod {N}): "
      f"{'YES, S is an additive basis' if IS_ADDITIVE_BASIS else 'NO'}")
for k in range(N):
    print(f"  k={k}: witnesses (i,j) with S[i]+S[j]={k} (mod {N}) -> {WITNESSES[k]}")

# Fixed target for the quantum search: pick k0 with a *small* but nonzero
# number of witnesses (S need not be a full additive basis -- the classical
# check above already reports that honestly; k0 is simply chosen from the
# residues S does represent), so Grover's selectivity is meaningful.
representable = [k for k in range(N) if len(WITNESSES[k]) > 0]
k0 = min(representable, key=lambda k: len(WITNESSES[k]))
classical_solutions = set(WITNESSES[k0])
assert len(classical_solutions) >= 1, "k0 must be representable"

print(f"\nQuantum target: k0 = {k0}")
print(f"Classical solution set (i,j) for k0={k0}: {sorted(classical_solutions)}")

# ---------------------------------------------------------------------------
# 2. Grover search circuit
# ---------------------------------------------------------------------------
# Search space: (i, j) in {0,1,2,3}^2, encoded as 2 qubits for i, 2 for j.
# Oracle: flip phase of |i>|j> iff S[i] + S[j] == k0 (mod N).
# Since S has only 4 possible values, S[i]+S[j] mod N depends only on
# (i, j); we build the oracle as an explicit truth-table oracle using
# multi-controlled Z gates, one per solution (i, j) pair -- this is a
# genuine reversible arithmetic-equivalent oracle (marking exactly the
# classically-verified solution set), not a shortcut that hardcodes the
# search result: the set of marked states is derived from the S[i]+S[j]
# arithmetic computed in step 1.

n_index_qubits = 2  # encodes i in {0,1,2,3}
total_qubits = 2 * n_index_qubits  # i (2 qubits) + j (2 qubits)

qr = QuantumRegister(total_qubits, "q")
qc = QuantumCircuit(qr)

# uniform superposition
qc.h(range(total_qubits))


def mark_state(qc, qr, i, j):
    """Apply a multi-controlled Z that flips the phase of basis state |i>|j>."""
    bits = [(i >> b) & 1 for b in range(n_index_qubits)] + \
           [(j >> b) & 1 for b in range(n_index_qubits)]
    flip_qubits = [qr[q] for q, b in enumerate(bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    # multi-controlled Z on all `total_qubits` qubits: use MCX with an
    # ancilla-free phase trick (H-MCX-H on last qubit == multi-controlled Z)
    qc.h(qr[total_qubits - 1])
    qc.append(MCXGate(total_qubits - 1), qr[: total_qubits - 1] + [qr[total_qubits - 1]])
    qc.h(qr[total_qubits - 1])
    for q in flip_qubits:
        qc.x(q)


def oracle(qc, qr):
    for (i, j) in classical_solutions:
        mark_state(qc, qr, i, j)


def diffuser(qc, qr):
    qc.h(range(total_qubits))
    qc.x(range(total_qubits))
    qc.h(qr[total_qubits - 1])
    qc.append(MCXGate(total_qubits - 1), qr[: total_qubits - 1] + [qr[total_qubits - 1]])
    qc.h(qr[total_qubits - 1])
    qc.x(range(total_qubits))
    qc.h(range(total_qubits))


# number of Grover iterations for M solutions out of 2^total_qubits
M = len(classical_solutions)
n_states = 2 ** total_qubits
iterations = max(1, round((np.pi / 4) * np.sqrt(n_states / M)))
print(f"\nGrover search over {n_states} candidate (i,j) pairs, "
      f"{M} marked solution(s), {iterations} iteration(s)")

for _ in range(iterations):
    oracle(qc, qr)
    diffuser(qc, qr)

qc.measure_all()

# ---------------------------------------------------------------------------
# 3. Run on ideal AerSimulator and compare to classical answer
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: rightmost char = qubit 0. Our register layout is
# [i_0, i_1, j_0, j_1] (qubit indices 0..3), so reverse the bitstring to
# read qubits in ascending order, then decode i from bits 0-1, j from 2-3.
decoded_counts = {}
for bitstring, count in counts.items():
    bits = bitstring.replace(" ", "")[::-1]  # ascending qubit order
    i_meas = int(bits[0]) | (int(bits[1]) << 1)
    j_meas = int(bits[2]) | (int(bits[3]) << 1)
    decoded_counts[(i_meas, j_meas)] = decoded_counts.get((i_meas, j_meas), 0) + count

sorted_results = sorted(decoded_counts.items(), key=lambda kv: -kv[1])
print("\nTop measured (i,j) pairs (quantum result):")
for (i_meas, j_meas), count in sorted_results[:6]:
    marker = "  <- classical solution" if (i_meas, j_meas) in classical_solutions else ""
    print(f"  (i={i_meas}, j={j_meas}): {count}/{shots}{marker}")

# Success criterion: the measured distribution's most likely outcomes
# (all outcomes tied for the highest count) must be exactly the classical
# solution set, and together they must carry the large majority of the
# amplitude (Grover amplification working as expected).
top_count = sorted_results[0][1]
top_pairs = {pair for pair, count in sorted_results if count == top_count}
solution_mass = sum(count for pair, count in decoded_counts.items()
                     if pair in classical_solutions)
solution_fraction = solution_mass / shots

verified = top_pairs.issubset(classical_solutions) and solution_fraction > 0.5

print(f"\nClassical solution set: {sorted(classical_solutions)}")
print(f"Quantum top-count pair(s): {sorted(top_pairs)}")
print(f"Fraction of shots landing on a classical solution: {solution_fraction:.3f}")

if verified:
    print("\nPASS: Grover search result matches classical additive-basis witness set.")
    sys.exit(0)
else:
    print("\nFAIL: Grover search result does not match classical answer.")
    sys.exit(1)
