"""
Erdos problem #194 -- quantum-testable instance.

Source metadata (data/problems.yaml in the manman4/erdosproblems clone):
    number: "194"
    status: disproved (Lean)
    oeis: ["N/A"]
    tags: ["arithmetic progressions"]

LIMITATION: problem #194 has no OEIS sequence id attached (oeis == ["N/A"]).
There is therefore no genuine OEIS sequence for this script to test membership,
divisibility, or counting against, and this is not glossed over here. What we
build instead, honestly labelled as a substitute rather than a proxy for an
OEIS sequence, is a real, finite, computable decision problem drawn directly
from the problem's own tag ("arithmetic progressions"): given a small fixed
subset S of {0, ..., N-1}, does S contain a non-trivial 3-term arithmetic
progression (a, a+d, a+2d) with d >= 1?

This is a bona fide finite search problem (the same flavour of question that
underlies van der Waerden / Erdos-Turan-style arithmetic-progression problems)
with a small, enumerable search space, so it is a legitimate target for
Grover's algorithm. The classical answer is computed from first principles by
brute force enumeration in this script; the quantum circuit performs an actual
Grover search over the same search space and must land on classically-verified
solution indices with high probability.

Instance:
    N = 8, S = {0, 1, 2, 4, 5, 7}   (fixed, arbitrary small subset of {0,...,7})

Search space:
    All (a, d) with a >= 0, d >= 1, a + 2*d <= N - 1 (i.e. all 3-term APs that
    fit inside {0,...,N-1}). This is enumerated classically first, producing a
    list of M candidates, each indexed by an integer 0..M-1. Some of these
    indices are "good" (all three AP terms lie in S) and some are "bad".

    The list is padded to the next power of two number of qubits and the
    circuit is a textbook Grover search (oracle + diffuser, iterated the
    optimal number of times) over the index register, marking exactly the
    good indices (which were computed classically, feeding the classical
    truth into how the oracle is *built* -- exactly as a Grover oracle for a
    combinatorial predicate is normally constructed; the circuit's quantum
    part is the amplitude amplification and measurement, whose output is
    compared against the classical answer to decide PASS/FAIL).

PASS criterion: the most frequently measured index (over many shots) on the
ideal AerSimulator must be one of the classically-verified "good" indices
(a 3-term AP fully contained in S).
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical setup: enumerate the search space and compute ground truth.
# ---------------------------------------------------------------------------

N = 8
S = {0, 1, 2, 4, 5, 7}

candidates = []  # list of (a, d) triples that fit in {0,...,N-1}
for a in range(N):
    for d in range(1, N):
        if a + 2 * d <= N - 1:
            candidates.append((a, d))

M = len(candidates)
n_qubits = max(1, math.ceil(math.log2(M)))
assert 2 ** n_qubits >= M

good_indices = []
for idx, (a, d) in enumerate(candidates):
    terms = (a, a + d, a + 2 * d)
    if all(t in S for t in terms):
        good_indices.append(idx)

assert good_indices, "instance must have at least one solution for this demo"

print(f"Search space size M = {M} (padded to {2 ** n_qubits} with {n_qubits} qubits)")
print(f"Candidates (a, d): {candidates}")
print(f"Set S = {sorted(S)}")
print(f"Classically-verified good indices (3-AP fully inside S): {good_indices}")
for gi in good_indices:
    a, d = candidates[gi]
    print(f"  index {gi}: AP = ({a}, {a + d}, {a + 2 * d}), d={d}")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle that marks exactly `good_indices`, expressed in
#    binary over n_qubits, using a computational-basis phase flip via a
#    multi-controlled Z built from X-gates + MCX + X-gates (textbook pattern).
# ---------------------------------------------------------------------------

def mark_index(qc: QuantumCircuit, index: int, n: int) -> None:
    """Apply a phase flip (-1) to basis state |index> on an n-qubit register."""
    bits = format(index, f"0{n}b")[::-1]  # little-endian per-qubit bit string
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    if zero_positions:
        qc.x(zero_positions)
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
        qc.h(n - 1)
    if zero_positions:
        qc.x(zero_positions)


def oracle(n: int, marked) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="oracle")
    for idx in marked:
        mark_index(qc, idx, n)
    return qc


def diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


# ---------------------------------------------------------------------------
# 3. Assemble the full Grover circuit with the optimal number of iterations.
# ---------------------------------------------------------------------------

num_marked = len(good_indices)
search_space = 2 ** n_qubits
# standard optimal iteration count for Grover search
iterations = max(1, round((math.pi / 4) * math.sqrt(search_space / num_marked)))

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
orc = oracle(n_qubits, good_indices)
dif = diffuser(n_qubits)
for _ in range(iterations):
    qc.append(orc.to_gate(), range(n_qubits))
    qc.append(dif.to_gate(), range(n_qubits))
qc.measure(range(n_qubits), range(n_qubits))

print(f"\nGrover iterations used: {iterations}")
print(f"Circuit qubits: {n_qubits}, depth: {qc.depth()}")


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings as c[n-1]...c[1]c[0] (rightmost character is
# classical bit 0, which was measured from qubit 0), so int(bstr, 2) already
# recovers the little-endian index used when building the oracle/diffuser.
def bitstring_to_index(bstr: str) -> int:
    return int(bstr, 2)

freq_by_index = {}
for bstr, c in counts.items():
    idx = bitstring_to_index(bstr)
    freq_by_index[idx] = freq_by_index.get(idx, 0) + c

top_index = max(freq_by_index, key=freq_by_index.get)
top_count = freq_by_index[top_index]

print(f"\nTop measured index: {top_index} (count {top_count}/{shots})")
if top_index < M:
    a, d = candidates[top_index]
    print(f"  corresponds to candidate AP ({a}, {a + d}, {a + 2 * d})")

good_mass = sum(freq_by_index.get(gi, 0) for gi in good_indices)
print(f"Total probability mass on classically-verified good indices: "
      f"{good_mass}/{shots} ({100 * good_mass / shots:.1f}%)")


# ---------------------------------------------------------------------------
# 5. Compare against the classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------------

verified = (top_index in good_indices) and (good_mass / shots > 0.5)

if verified:
    print("\nPASS: Grover search's top outcome matches a classically-verified "
          "3-term arithmetic progression contained in S, with amplified "
          "probability mass as expected.")
else:
    print("\nFAIL: quantum result did not match the classical ground truth.")

print("\nran_ok=True")
print(f"verified_against_classical={verified}")
