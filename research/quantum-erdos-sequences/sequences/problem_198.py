"""
Erdos problem #198 — quantum-testable instance.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry
`number: "198"`): status "disproved (Lean)", oeis: ["N/A"], tags:
["additive combinatorics", "sidon sets", "arithmetic progressions"].

LIMITATION: problem #198 carries no OEIS sequence id (oeis: ["N/A"]), so
there is no OEIS term-membership property to test. Per the task's fallback
instructions, this script instead builds a genuine, finite, computable
property drawn directly from the problem's own tags (Sidon sets), rather
than fabricating or borrowing an OEIS value. No OEIS id is used.

Classical property tested
--------------------------
A Sidon set (B2 set) is a set of integers in which all pairwise sums
a+b (a<=b) are distinct. Take the ground set {0,1,2,3,4,5} and all
C(6,3) = 20 three-element subsets. Exactly one classical, fully specified
question is asked: "which of these 20 subsets are Sidon sets?" This is
computed from first principles by brute force in this script (function
`is_sidon`), independent of any external table.

The 20 subsets are indexed 0..19 (in the order produced by
itertools.combinations); indices 20..31 are padding (5 qubits address 32
states, only 20 of which correspond to a real subset).

Quantum circuit
----------------
A Grover search over the 5-qubit index register is built whose oracle
marks exactly the indices of the classically-computed Sidon subsets. The
oracle is constructed directly from that marked list (multi-controlled-Z
per marked bitstring), the diffusion operator is the standard
Grover diffuser, and the optimal number of iterations is computed from
the true counts N=32, M=|marked|. The circuit is run on the ideal
AerSimulator (statevector-derived sampling), and we check that the most
frequently measured index is one of the classically marked Sidon-set
indices.

PASS/FAIL
---------
PASS if the most-sampled basis state decodes to an index whose subset
is independently verified (again, classically, in this script) to be a
Sidon set. This is a real amplitude-amplification search, not a lookup.
"""

import itertools
import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no external data).
# ---------------------------------------------------------------------------

def is_sidon(subset):
    """A finite set of ints is Sidon iff all pairwise sums a+b (a<=b) differ."""
    sums = []
    items = sorted(subset)
    for i in range(len(items)):
        for j in range(i, len(items)):
            sums.append(items[i] + items[j])
    return len(sums) == len(set(sums))


GROUND_SET = [0, 1, 2, 3, 4, 5]
SUBSETS = list(itertools.combinations(GROUND_SET, 3))  # 20 subsets, index 0..19
assert len(SUBSETS) == 20

sidon_flags = [is_sidon(s) for s in SUBSETS]
MARKED_INDICES = [i for i, ok in enumerate(sidon_flags) if ok]

print("Ground set:", GROUND_SET)
print("Number of 3-subsets:", len(SUBSETS))
print("Sidon subsets (classical, brute force):")
for i in MARKED_INDICES:
    print(f"  index {i}: {SUBSETS[i]}")
print(f"Total Sidon subsets: {len(MARKED_INDICES)} out of {len(SUBSETS)}")

N_QUBITS = 6           # addresses 64 states, covers indices 0..63 (only 0..19 used);
                        # the larger-than-needed address space keeps the marked
                        # fraction M/N small enough for clean Grover amplification
N_STATES = 2 ** N_QUBITS
M_MARKED = len(MARKED_INDICES)
assert 0 < M_MARKED < N_STATES


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffuser built directly from MARKED_INDICES.
# ---------------------------------------------------------------------------

def bits_for(index, n):
    return format(index, f"0{n}b")[::-1]  # little-endian, qubit0 = LSB


def apply_marking(qc, index, n):
    """Flip qubits that are 0 in `index`'s bit pattern, multi-controlled-Z, undo."""
    pattern = bits_for(index, n)
    zero_qubits = [q for q, b in enumerate(pattern) if b == "0"]
    for q in zero_qubits:
        qc.x(q)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for q in zero_qubits:
        qc.x(q)


def oracle(n, marked):
    qc = QuantumCircuit(n, name="oracle")
    for idx in marked:
        apply_marking(qc, idx, n)
    return qc


def diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


# Optimal number of Grover iterations for N states, M marked.
theta = math.asin(math.sqrt(M_MARKED / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"\nGrover iterations used: {iterations} (N={N_STATES}, M={M_MARKED})")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
orc = oracle(N_QUBITS, MARKED_INDICES)
dif = diffuser(N_QUBITS)
for _ in range(iterations):
    qc.append(orc.to_gate(), range(N_QUBITS))
    qc.append(dif.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
result = backend.run(compiled, shots=4096).result()
counts = result.get_counts()

# Qiskit's counts keys are standard big-endian strings (c[n-1]..c[0], MSB
# first) of the integer basis-state index, matching int(bitstring, 2)
# directly -- verified against Statevector amplitudes during development.
decoded_counts = Counter()
for bitstring, c in counts.items():
    idx = int(bitstring, 2)
    decoded_counts[idx] += c

top_index, top_count = decoded_counts.most_common(1)[0]
print(f"\nMost frequently measured index: {top_index} "
      f"({top_count}/{4096} shots)")

if top_index < len(SUBSETS):
    print(f"Decoded subset: {SUBSETS[top_index]}")

# Fraction of shots landing on ANY classically-marked Sidon index.
hit_shots = sum(c for i, c in decoded_counts.items() if i in MARKED_INDICES)
print(f"Shots landing on a Sidon-set index: {hit_shots}/4096 "
      f"({100 * hit_shots / 4096:.1f}%)")


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

quantum_found_sidon = (
    top_index in MARKED_INDICES
    and is_sidon(SUBSETS[top_index])          # re-verify classically, standalone
    and hit_shots / 4096 > 0.5                # amplitude amplification worked
)

if quantum_found_sidon:
    print("\nPASS: Grover search's most-likely outcome is a classically "
          "verified Sidon set, and amplified amplitude confirms genuine "
          "search behavior.")
else:
    print("\nFAIL: quantum result did not match the classical Sidon-set "
          "computation.")

assert quantum_found_sidon, "quantum result disagrees with classical answer"
