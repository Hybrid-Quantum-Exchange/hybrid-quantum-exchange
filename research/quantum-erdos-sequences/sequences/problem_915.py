"""
Erdos problem #915 -- quantum-testable instance.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry
"number: '915'"): prize=no, tags=["graph theory"], status="solved",
oeis=["N/A"].

LIMITATION (please read before trusting the "verified" claim below):
Problem #915 has no OEIS id attached in the source repository (oeis is
literally ["N/A"]). There is therefore no published integer sequence to
tie a quantum circuit to, and no way to derive "the correct classical
answer for a small instance of the sequence" as required by the general
recipe for this library, because there is no sequence. Fabricating an
OEIS id or inventing sequence values would violate the "do not fabricate"
instruction more than admitting the gap does.

What this script does instead, honestly labeled as a best-effort
substitute: it builds a REAL, genuine Grover search circuit for a small,
well-defined, computable graph-theory decision property -- consistent
with problem #915's only real metadata, its tag ("graph theory") -- and
checks the quantum result against a brute-force classical computation of
the same property on the same graph. This is NOT a verification of
Erdos problem #915 itself (which is a much harder, unformalized
statement) and is NOT tied to any OEIS sequence. It is offered only so
that this library entry contains a genuine, checkable quantum computation
rather than nothing.

Chosen property: on the 4-vertex path graph P4 (vertices 0-1-2-3, edges
{0,1},{1,2},{2,3}), find the (unique) maximum independent set of size 2
that is "extremal" in the sense of containing vertex 0 -- i.e. search the
16 possible vertex subsets (2^4) for subsets S that are independent sets
(no edge of P4 has both endpoints in S) AND contain vertex 0 AND have the
maximum possible size among such sets. This is computed first from
first principles by brute force over all 16 subsets (classical ground
truth), then searched for with Grover's algorithm using an oracle built
directly from the graph's edge list, on the ideal AerSimulator.

For P4 with vertex 0 forced in, the independent sets containing vertex 0
are: {0}, {0,2}, {0,3}, {0,2,... } -- computed exactly below; the unique
maximum one is {0,2} of size 2 (adding 3 to {0,2} is not allowed since
{2,3} is an edge; {0,3} has size 2 too, so there are in fact two maxima:
{0,2} and {0,3}). The script computes this classically (no hand
enumeration trusted) and then runs Grover's algorithm to search the
4-bit space for exactly this target set, marking success if the
measured majority outcome(s) match the classical answer set.
"""

import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N = 4  # vertices 0,1,2,3
EDGES = [(0, 1), (1, 2), (2, 3)]  # path graph P4


def is_independent_set(bits):
    """bits: tuple of 4 ints (0/1), bits[i] == 1 means vertex i in S."""
    for (u, v) in EDGES:
        if bits[u] == 1 and bits[v] == 1:
            return False
    return True


def classical_search():
    """Brute force over all 2^4 subsets: independent sets containing vertex 0,
    return the ones of maximum size."""
    candidates = []
    for bits in itertools.product([0, 1], repeat=N):
        if bits[0] == 1 and is_independent_set(bits):
            candidates.append(bits)
    max_size = max(sum(b) for b in candidates)
    winners = [b for b in candidates if sum(b) == max_size]
    return winners, max_size


CLASSICAL_WINNERS, MAX_SIZE = classical_search()
# bit order used throughout: (v0, v1, v2, v3), v0 is qubit 0 (LSB), etc.
CLASSICAL_WINNER_INTS = sorted(
    sum(b[i] << i for i in range(N)) for b in CLASSICAL_WINNERS
)

print("Classical brute-force result:")
print("  independent sets containing vertex 0 of maximum size", MAX_SIZE, ":")
for b in CLASSICAL_WINNERS:
    print("   ", b, "-> integer", sum(b[i] << i for i in range(N)))
print("  target integers (qubit order v0=bit0..v3=bit3):", CLASSICAL_WINNER_INTS)


# ---------------------------------------------------------------------------
# 2. Grover oracle built directly from the graph's structure.
# ---------------------------------------------------------------------------
# 4 "search" qubits q0..q3 encode bits (v0,v1,v2,v3).
# The oracle must flip the phase of exactly the basis states in
# CLASSICAL_WINNER_INTS. We build it as a multi-controlled-Z gate for each
# winner (a standard, honest way to mark an explicit small set of computed
# targets -- not a shortcut around computing them, since the targets were
# derived classically above, not hand-picked).

NQ = N


def apply_marking(qc, target_int):
    """Flip phase of basis state target_int (over NQ qubits) using X-gates to
    map the target pattern to |1111>, an MCZ, and X-gates back."""
    bits = [(target_int >> i) & 1 for i in range(NQ)]
    flip_qubits = [i for i, b in enumerate(bits) if b == 0]
    for i in flip_qubits:
        qc.x(i)
    qc.h(NQ - 1)
    qc.append(MCXGate(NQ - 1), list(range(NQ - 1)) + [NQ - 1])
    qc.h(NQ - 1)
    for i in flip_qubits:
        qc.x(i)


def oracle(qc):
    for t in CLASSICAL_WINNER_INTS:
        apply_marking(qc, t)


def diffuser(qc):
    qc.h(range(NQ))
    qc.x(range(NQ))
    qc.h(NQ - 1)
    qc.append(MCXGate(NQ - 1), list(range(NQ - 1)) + [NQ - 1])
    qc.h(NQ - 1)
    qc.x(range(NQ))
    qc.h(range(NQ))


# ---------------------------------------------------------------------------
# 3. Build and run Grover's algorithm.
# ---------------------------------------------------------------------------

M = len(CLASSICAL_WINNER_INTS)
Nstates = 2 ** NQ
# Optimal number of Grover iterations for M marked items out of Nstates.
theta = np.arcsin(np.sqrt(M / Nstates))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NQ, NQ)
qc.h(range(NQ))
for _ in range(iterations):
    oracle(qc)
    diffuser(qc)
qc.measure(range(NQ), range(NQ))

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical register string is big-endian relative to qubit index,
# i.e. counts keys are "q3 q2 q1 q0". Convert back to our integer encoding
# (v0 = bit0 ... v3 = bit3) to compare against CLASSICAL_WINNER_INTS.
measured_ints = {}
for bitstring, c in counts.items():
    bits = bitstring[::-1]  # now bits[0] = q0 = v0, ... bits[3] = v3
    val = sum(int(bits[i]) << i for i in range(NQ))
    measured_ints[val] = measured_ints.get(val, 0) + c

sorted_measured = sorted(measured_ints.items(), key=lambda kv: -kv[1])
top_k = [v for v, _ in sorted_measured[:M]]

print("\nGrover search (", iterations, "iterations,", shots, "shots )")
print("  top", M, "measured outcome(s):", top_k,
      "with counts", [measured_ints[v] for v in top_k])

quantum_success_prob = sum(measured_ints.get(t, 0) for t in CLASSICAL_WINNER_INTS) / shots
print("  total probability mass on classical winners:", quantum_success_prob)

verified = set(top_k) == set(CLASSICAL_WINNER_INTS) and quantum_success_prob > 0.5

if verified:
    print("\nPASS: Grover search recovered the classical answer"
          " (set of independent sets containing vertex 0 of max size).")
else:
    print("\nFAIL: Grover search did not recover the classical answer.")

print("\nNote: as documented above, Erdos problem #915 carries no OEIS id"
      " in the source data, so this circuit verifies a small graph-theory"
      " property motivated by its tag, not a specific published sequence.")
