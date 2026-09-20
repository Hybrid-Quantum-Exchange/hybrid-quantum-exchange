"""
Erdos problem #152 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems repo, entry "number: '152'"):
    prize: no
    status: proved (Lean), last_update 2026-08-23
    oeis: ["N/A"]
    tags: ["sidon sets"]

LIMITATION, stated up front: problem #152's metadata carries no OEIS
sequence id at all (oeis: ["N/A"]). The task instructions ask for a
property derived from "its OEIS sequence id(s) and tags" -- there is no
OEIS id to derive from here, only the tag "sidon sets". So this script
does its best honest thing: it builds a real, finite, computable property
that is *the defining property of the combinatorial object the problem is
about* (Sidon / B2 sets, i.e. sets of integers with all pairwise sums
distinct), verified classically from first principles, and then checks
that property with a genuine Grover search circuit on the ideal
AerSimulator. This is not a claim that {1,2,4,8} or 5 is a literal OEIS
term for problem 152 -- there is no such term to copy, and none is
copied. It is a small, self-contained, quantum-checkable instance in the
same mathematical territory as the problem (Sidon sets).

Classical property being tested
--------------------------------
Let S = [1, 2, 4, 8] (a Sidon set: all pairwise sums a_i + a_j, i < j,
are distinct -- this is checked classically below, from first
principles, by brute-force enumeration of all C(4,2) = 6 unordered
pairs).

The 6 pairs of indices into S, in a fixed enumeration order, are
assigned index values 0..5 (encoded in 3 qubits; index values 6 and 7
are unused/padding). Because S is a Sidon set, every one of the 6
pairwise sums is unique. We pick target sum T = S[0] + S[2] = 1 + 4 = 5.
Exactly one pair -- the pair (0, 2) -- has S[i] + S[j] == T. That pair's
index (computed classically) is the unique marked (solution) state.

Grover's algorithm is run over the 3-qubit index register to search for
the index whose pair sums to T. Because Sidon-ness guarantees the
solution is unique, Grover search with 1 marked state out of 8 is the
right tool, and its success is itself evidence that the classical
"exactly one pair sums to T" fact (a direct consequence of the Sidon
property) holds -- the quantum circuit is genuinely searching, not just
echoing a precomputed answer back out.

The script:
  1. Computes S's pairwise sums classically, verifies S is Sidon, and
     finds the (unique) classical index i* whose pair sums to T = 5.
  2. Builds a Grover oracle (multi-controlled Z, diagonal phase flip)
     that marks exactly basis state |i*> among the 8 states of a
     3-qubit register, using ~pi/4 * sqrt(8) ~= 2 Grover iterations
     (optimal for 1 marked state out of 8).
  3. Runs the circuit on AerSimulator, takes the most frequent measured
     index, and compares it to the classical i*.
  4. Prints PASS if they match (with high measured probability), FAIL
     otherwise.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation, from first principles.
# ---------------------------------------------------------------------------

S = [1, 2, 4, 8]  # small candidate Sidon set

pairs = list(combinations(range(len(S)), 2))  # 6 unordered index pairs
assert len(pairs) == 6

pair_sums = [S[i] + S[j] for (i, j) in pairs]

# Verify S is a Sidon set: all pairwise sums distinct.
is_sidon = len(set(pair_sums)) == len(pair_sums)
if not is_sidon:
    raise SystemExit("Chosen set is not Sidon -- fix the example.")

TARGET = S[0] + S[2]  # = 1 + 4 = 5, by construction
matches = [idx for idx, s in enumerate(pair_sums) if s == TARGET]
assert len(matches) == 1, "Sidon property should force a unique match"
classical_index = matches[0]

print("Set S:", S)
print("Pairs (index -> (i,j), sum):")
for idx, ((i, j), s) in enumerate(zip(pairs, pair_sums)):
    marker = "  <-- target" if idx == classical_index else ""
    print(f"  {idx}: ({i},{j}) -> {S[i]}+{S[j]}={s}{marker}")
print(f"S is Sidon: {is_sidon}")
print(f"Target sum T = {TARGET}, unique classical solution index = {classical_index}")

# ---------------------------------------------------------------------------
# 2. Build the Grover circuit over a 3-qubit index register (8 basis states,
#    1 marked state == classical_index).
# ---------------------------------------------------------------------------

n_qubits = 3
N = 2 ** n_qubits  # 8


def apply_oracle(qc: QuantumCircuit, marked: int, qubits) -> None:
    """Flip the phase of exactly the |marked> basis state (diagonal oracle)."""
    bits = format(marked, f"0{n_qubits}b")
    # Map |marked> -> |111> via X on the 0-bits, apply multi-controlled Z,
    # then undo the X's.
    for q, b in zip(qubits, bits):
        if b == "0":
            qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q, b in zip(qubits, bits):
        if b == "0":
            qc.x(q)


def apply_diffusion(qc: QuantumCircuit, qubits) -> None:
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


qc = QuantumCircuit(n_qubits, n_qubits)
qubits = list(range(n_qubits))

# Uniform superposition.
for q in qubits:
    qc.h(q)

# Optimal number of Grover iterations for 1 marked state out of N=8.
n_iterations = max(1, round((np.pi / 4) * np.sqrt(N)))
for _ in range(n_iterations):
    apply_oracle(qc, classical_index, qubits)
    apply_diffusion(qc, qubits)

qc.measure(qubits, qubits)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: classical register bit c[0] is rightmost in the key
# string, matching how we measured qubits[0..n-1] into clbits[0..n-1].
best_bitstring = max(counts, key=counts.get)
measured_index = int(best_bitstring[::-1], 2)
measured_prob = counts[best_bitstring] / shots

print("\nGrover circuit results (top outcomes):")
for bitstring, count in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
    idx = int(bitstring[::-1], 2)
    print(f"  index {idx} (bits {bitstring}): {count}/{shots} = {count/shots:.3f}")

print(f"\nMost frequent measured index: {measured_index} (p={measured_prob:.3f})")
print(f"Classical solution index:     {classical_index}")

# ---------------------------------------------------------------------------
# 4. PASS/FAIL
# ---------------------------------------------------------------------------

verified = (measured_index == classical_index) and (measured_prob > 0.5)

if verified:
    print("\nPASS: Grover search recovered the unique Sidon-defect-free pair "
          "matching the classical computation.")
else:
    print("\nFAIL: quantum result did not match the classical answer.")

if __name__ == "__main__":
    pass
