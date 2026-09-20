"""
Erdos problem #895 -- quantum-testable lane.

Source metadata (erdosproblems.com data, data/problems.yaml, number: "895"):
  prize: no
  status: proved (last_update 2025-08-31)
  oeis: ["N/A"]
  tags: ["additive combinatorics", "graph theory"]

LIMITATION, stated honestly up front: problem #895 has NO OEIS id attached
(oeis: ["N/A"] in the source record). There is therefore no integer sequence
to pull a term or membership test from, and the task's instruction to derive
a property "from its OEIS sequence id(s)" cannot literally be followed --
there is no id. Per the fallback instruction ("write the script anyway with
your best honest attempt, note the limitation clearly"), this script instead
builds a genuine, finite, computable instance of the problem's actual
mathematical content, taken from its tags ("additive combinatorics"): the
existence/identification of SUM-FREE SUBSETS of a finite integer set. A
sum-free subset S of {1,...,n} is one with no a, b, c in S (a, b distinct)
such that a + b = c. This is a standard, well-defined finite combinatorial
search problem squarely in "additive combinatorics" -- the same area problem
895 is tagged with -- and is NOT a value copied from any OEIS entry; it is
derived and checked classically below, from first principles, in this script.

Classical instance chosen: n = 4, universe U = {1, 2, 3, 4}.
There are 2^4 = 16 subsets of U. We classically enumerate all 16 and mark
which are sum-free. This is the ground truth the quantum circuit is checked
against.

Quantum circuit: a genuine Grover search over the 16 subsets (4 qubits,
one qubit per element of U, |1> at position i meaning "i+1 is in the
subset"). For {1,2,3,4}, 13 of the 16 subsets turn out to be sum-free, so
marking "sum-free" would mark a majority of the search space, which is a
degenerate (uninteresting, and numerically unstable for the standard Grover
iteration-count formula) instance for amplitude amplification. Grover's
algorithm is naturally suited to searching for the minority "special" items,
so the oracle instead marks the NOT-sum-free subsets -- i.e. subsets S
containing some a, b, c in S (a != b) with a + b = c -- of which there are
exactly 3 for {1,2,3,4}. The oracle is built directly from that
classically-computed set (a multi-controlled-Z marking exactly those
computational basis states), followed by the standard Grover diffusion
operator, iterated the optimal number of times for the known number of
marked states. The circuit is run on the ideal AerSimulator (statevector-
based sampling), and the script checks that the highest-probability measured
outcomes are exactly the classically-computed non-sum-free subsets.

PASS/FAIL: the script prints PASS if Grover search recovers (as the set of
outcomes with the highest measured probability) precisely the classical set
of non-sum-free subsets of {1,2,3,4}, and FAIL otherwise.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: enumerate sum-free subsets of {1,2,3,4}.
# ---------------------------------------------------------------------------

UNIVERSE = [1, 2, 3, 4]
N = len(UNIVERSE)  # number of qubits = number of elements


def is_sum_free(subset):
    """True iff no a,b,c in subset (a != b) satisfy a + b == c."""
    s = set(subset)
    for a, b in itertools.permutations(subset, 2):
        if (a + b) in s:
            return False
    return True


def bits_to_subset(bits):
    """bits: tuple of 0/1 of length N, bit i corresponds to UNIVERSE[i]."""
    return [UNIVERSE[i] for i in range(N) if bits[i] == 1]


sum_free_subsets = []
not_sum_free_subsets = []
for bits in itertools.product([0, 1], repeat=N):
    subset = bits_to_subset(bits)
    if is_sum_free(subset):
        sum_free_subsets.append(bits)
    else:
        not_sum_free_subsets.append(bits)

# Grover marks the minority class: the NOT-sum-free subsets.
classical_marked = not_sum_free_subsets

num_marked = len(classical_marked)
total_states = 2 ** N

print(f"Universe: {UNIVERSE}")
print(f"Total subsets: {total_states}")
print(f"Sum-free subsets (classical, first-principles enumeration): {len(sum_free_subsets)}")
print(f"NOT-sum-free subsets (classical, first-principles enumeration, Grover target): {num_marked}")
for bits in classical_marked:
    print(f"  subset={bits_to_subset(bits)}  bits={''.join(map(str, bits))}")

assert 0 < num_marked < total_states, (
    "Grover search is not meaningful if 0 or all states are marked; "
    "instance choice must guarantee a nontrivial marked set."
)

# Qiskit bit ordering: qubit 0 is the least-significant bit of the bitstring
# Qiskit reports (rightmost character). We define bit i (element UNIVERSE[i])
# to live on qubit i, so the classical bitstring "b0 b1 b2 b3" (b0=UNIVERSE[0]
# membership) is reversed when compared to Qiskit's printed bitstrings.


def bits_to_qiskit_bitstring(bits):
    # Qiskit prints qubit (N-1) ... qubit 0, i.e. reversed relative to our
    # (bit for qubit 0, bit for qubit 1, ...) tuple.
    return "".join(str(b) for b in reversed(bits))


marked_qiskit_strings = {bits_to_qiskit_bitstring(b) for b in classical_marked}

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search with an oracle built from the marked set.
# ---------------------------------------------------------------------------


def build_oracle(n_qubits, marked_bitstrings):
    """Phase-flip oracle: apply -1 phase to each marked computational basis
    state (marked_bitstrings: iterable of tuples, bit i = qubit i value)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
    for bits in marked_bitstrings:
        zero_positions = [i for i in range(n_qubits) if bits[i] == 0]
        for i in zero_positions:
            qc.x(i)
        # Multi-controlled Z across all qubits (control = qubits 0..n-2,
        # target = qubit n-1) flips the phase of |11...1> only, i.e. exactly
        # the state we've mapped this marked bitstring onto via the X gates.
        qc.append(mcz, list(range(n_qubits - 1)) + [n_qubits - 1])
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
    qc.append(mcz, list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(N, classical_marked)
diffuser = build_diffuser(N)

# Optimal number of Grover iterations for M marked out of 2^N states.
theta = np.arcsin(np.sqrt(num_marked / total_states))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N))
    qc.append(diffuser.to_instruction(), range(N))
qc.measure(range(N), range(N))

print(f"\nGrover iterations used: {iterations}")

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 4096
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
print("\nTop measured outcomes:")
for bitstring, count in sorted_counts[:max(num_marked, 5)]:
    print(f"  {bitstring}: {count}/{shots}")

# The outcomes with the highest measured counts should be exactly the
# classically marked (sum-free) bitstrings.
top_outcomes = {bitstring for bitstring, _ in sorted_counts[:num_marked]}

quantum_matches_classical = top_outcomes == marked_qiskit_strings

print(f"\nClassical marked set (Qiskit bit order): {sorted(marked_qiskit_strings)}")
print(f"Quantum top-{num_marked} outcomes:        {sorted(top_outcomes)}")

if quantum_matches_classical:
    print("\nPASS")
    sys.exit(0)
else:
    print("\nFAIL")
    sys.exit(1)
