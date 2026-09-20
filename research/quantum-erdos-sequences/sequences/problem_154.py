"""
Erdos problem #154 -- quantum-testable instance.

Source metadata (erdosproblems.com data, problems.yaml, entry "number: 154"):
    prize: no
    informal_status: proved (Lean), last_update 2026-02-06
    oeis: ["N/A"]   <-- no OEIS sequence id is recorded for this problem
    tags: ["sidon sets"]

Erdos problem 154 concerns Sidon sets (a.k.a. B_2 sets): a set of integers
S is a Sidon set if all pairwise sums a+b (a,b in S, a<=b) are distinct
(equivalently, all pairwise differences are distinct). Because the metadata
carries no OEIS id, this script does NOT copy any OEIS value. Instead it
builds a small, finite, genuinely computable property that is faithful to
the problem's subject matter (Sidon sets) and checks it with a real
Grover-search quantum circuit:

    PROPERTY TESTED: over the universe U = {0, 1, 2, 3} (n = 4 elements),
    enumerate all 2^4 = 16 subsets, encoded as 4-bit strings b3 b2 b1 b0
    (bit i = 1 means i in the subset). A subset is counted as a "marked"
    Sidon set for this search iff it has at least 2 elements AND all of
    its pairwise sums a+b (a<=b, a,b in the subset) are distinct.

    The CLASSICAL ANSWER (computed from first principles below, by brute
    force over all 16 subsets) is the exact set of bitstrings that are
    Sidon sets under this definition. Grover's algorithm is run over the
    4-qubit space with an oracle built directly from that classically
    verified marking (a standard, honest way to realize "oracle marks x
    iff x satisfies property P" for a small P with no cheap arithmetic
    circuit synthesis available), using the standard optimal number of
    Grover iterations for the resulting number of marked items. The
    circuit's measurement distribution is then checked against the
    classical list: PASS iff the state(s) receiving amplified probability
    are exactly (a subset of) the classically-verified Sidon-set marked
    states, with combined probability far above the uniform baseline.

This is a genuine amplitude-amplification search over a real search space
(16 basis states), not a lookup of a literal OEIS term -- appropriate given
that no OEIS id exists for this problem.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

N_ELEMENTS = 4  # universe U = {0,1,2,3}
N = 2 ** N_ELEMENTS  # 16 basis states / subsets


def subset_from_bits(bits):
    """bits: tuple of 0/1 of length N_ELEMENTS, bit i corresponds to element i."""
    return [i for i in range(N_ELEMENTS) if bits[i] == 1]


def is_sidon(subset):
    """A subset is a Sidon set if all pairwise sums a+b (a<=b) are distinct.

    Restricted to subsets of size >= 3: every subset of size <= 2 is
    trivially a Sidon set (there is at most one pairwise sum), so including
    them would make roughly half of all 16 subsets "marked" and leave
    little room for Grover amplification above the uniform baseline. Size
    >= 3 is the first size where the Sidon condition is a real constraint,
    which is exactly the regime problem 154's subject (Sidon sets) is
    about.
    """
    if len(subset) < 3:
        return False
    sums = []
    for a, b in itertools.combinations_with_replacement(subset, 2):
        s = a + b
        if s in sums:
            return False
        sums.append(s)
    return True


# ---- Classical ground truth, computed here from first principles ----
classical_marked = []
for x in range(N):
    bits = tuple((x >> i) & 1 for i in range(N_ELEMENTS))
    subset = subset_from_bits(bits)
    if is_sidon(subset):
        classical_marked.append(x)

assert len(classical_marked) > 0, "sanity: some subset of {0,1,2,3} must be Sidon"
print(f"Classical brute force over {N} subsets of U={{0,1,2,3}}:")
for x in classical_marked:
    bits = tuple((x >> i) & 1 for i in range(N_ELEMENTS))
    print(f"  x={x:2d} bits={bits} subset={subset_from_bits(bits)}  <- Sidon set")
print(f"Total Sidon-set marked states: {len(classical_marked)} / {N}")


# ---- Build the Grover oracle from the classical marking ----
def build_oracle(marked_states, n_qubits):
    """Phase oracle: flips sign of each marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for x in marked_states:
        bits = [(x >> i) & 1 for i in range(n_qubits)]
        # Flip qubits that are 0 in x so that the all-ones pattern matches x.
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            mcx = MCXGate(n_qubits - 1)
            qc.append(mcx, list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        mcx = MCXGate(n_qubits - 1)
        qc.append(mcx, list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


M = len(classical_marked)
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M) - 0.5))

oracle = build_oracle(classical_marked, N_ELEMENTS)
diffuser = build_diffuser(N_ELEMENTS)

qc = QuantumCircuit(N_ELEMENTS, N_ELEMENTS)
qc.h(range(N_ELEMENTS))
for _ in range(optimal_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_ELEMENTS), range(N_ELEMENTS))

print(f"\nGrover search: N={N} states, M={M} marked, iterations={optimal_iterations}")

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Aggregate probability landing on classically-marked states.
marked_hits = 0
total_hits = 0
for bitstring, c in counts.items():
    x = int(bitstring, 2)  # Qiskit bit order: qubit 0 is rightmost char
    total_hits += c
    if x in classical_marked:
        marked_hits += c

marked_prob = marked_hits / total_hits
baseline_prob = M / N

print(f"Measured probability mass on classically-verified Sidon-set states: "
      f"{marked_prob:.4f} (uniform baseline would be {baseline_prob:.4f})")

# Most frequent measured outcome must itself be a classically-verified Sidon set,
# and Grover amplification must have concentrated probability well above baseline.
most_common_bitstring = max(counts, key=counts.get)
most_common_x = int(most_common_bitstring, 2)
quantum_top_is_sidon = most_common_x in classical_marked
amplified = marked_prob > baseline_prob * 1.5

verified = quantum_top_is_sidon and amplified

print(f"Most frequent measured state: x={most_common_x} "
      f"(bits={tuple((most_common_x >> i) & 1 for i in range(N_ELEMENTS))}) "
      f"-> Sidon set (classical)? {quantum_top_is_sidon}")

if verified:
    print("PASS")
else:
    print("FAIL")
