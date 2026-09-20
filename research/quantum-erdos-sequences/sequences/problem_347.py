"""
Erdos problem #347 — quantum-testable instance.

Source metadata (data/problems.yaml, entry "number: '347'"): prize=no,
status="proved (Lean)", oeis=["N/A"], tags=["number theory",
"complete sequences"].

LIMITATION: problem #347 has no associated OEIS sequence id (oeis: "N/A"),
so there is no OEIS term to target directly. Rather than fabricate an OEIS
value, this script tests the actual mathematical notion carried by the
problem's tag "complete sequences": a finite set S of positive integers is
"complete" with respect to a target T if T can be written as the sum of a
subset of DISTINCT elements of S. This is the finite, computable core of
completeness questions (Erdos studied which sets are complete, i.e. which
targets/integers are representable as subset sums of a given set).

Classical property tested (computed from first principles in this script,
no OEIS lookup):

    Given S = [1, 2, 3, 5, 8, 13] (6 positive integers, so the search
    space of subsets has size 2**6 = 64 <= 64 qubX states) and target
    T = 21, find all subsets of S whose elements sum to exactly T.

    Brute force over all 64 subsets gives the exact classical answer
    (computed below, not copied from anywhere).

Quantum method: Grover's algorithm over 6 qubits (one qubit per element of
S, |1> meaning "include this element"). The oracle is built classically by
enumerating, for each of the 64 basis states, whether the corresponding
subset sums to T; matching basis states are phase-flipped with
multi-controlled Z gates (via X-gates on the 0-bits + a controlled-Z
pattern), followed by the standard Grover diffusion operator, iterated the
optimal number of times for the known number of marked states. This is a
genuine amplitude-amplification circuit, not a lookup table: the AerSimulator
statevector/measurement result is compared against the classical brute-force
answer.

PASS/FAIL: the script prints PASS if the quantum circuit's most-sampled
outcomes are exactly the classical solution set (with total probability
mass concentrated there well above the uniform baseline), else FAIL.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed here, from first principles).
# ---------------------------------------------------------------------------

S = [1, 2, 3, 5, 8, 13]
TARGET = 21
N = len(S)  # number of qubits / elements

def subset_sum(bits):
    """bits: tuple of 0/1 of length N, bit i = 1 means S[i] included."""
    return sum(s for s, b in zip(S, bits) if b)

classical_solutions = []
for bits in itertools.product([0, 1], repeat=N):
    if subset_sum(bits) == TARGET:
        classical_solutions.append(bits)

assert len(classical_solutions) > 0, "instance must have at least one solution"

# bitstring convention: Qiskit orders classical register bits with qubit 0
# as the least-significant (rightmost) character. bits tuple index i ->
# qubit i -> character position (N-1-i) from the left in the printed string.
def bits_to_qiskit_string(bits):
    return "".join(str(b) for b in reversed(bits))

classical_solution_strings = {bits_to_qiskit_string(b) for b in classical_solutions}

print(f"Instance: S={S}, target T={TARGET}, N={N} qubits, search space={2**N}")
print(f"Classical solutions (subset bit patterns, S-order): {classical_solutions}")
print(f"Classical solution bitstrings (Qiskit order): {sorted(classical_solution_strings)}")


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classical solution states.
# ---------------------------------------------------------------------------

def apply_oracle(qc, marked_bitstrings, qubits):
    """Phase-flip each basis state in marked_bitstrings (Qiskit bit order,
    qubit 0 = rightmost character) using X + multi-controlled-Z + X."""
    n = len(qubits)
    for bstr in marked_bitstrings:
        # bstr[0] corresponds to qubit n-1 ... bstr[n-1] corresponds to qubit 0
        # Flip qubits that should be 0 in this pattern so the all-ones
        # pattern lines up with a standard MCZ.
        zero_qubits = [qubits[n - 1 - i] for i, c in enumerate(bstr) if c == "0"]
        for q in zero_qubits:
            qc.x(q)
        # multi-controlled Z on all n qubits: use H + MCX + H on the target
        target = qubits[-1]
        controls = qubits[:-1]
        qc.h(target)
        if len(controls) == 0:
            qc.z(target)
        else:
            qc.append(MCXGate(len(controls)), controls + [target])
        qc.h(target)
        for q in zero_qubits:
            qc.x(q)


def apply_diffuser(qc, qubits):
    n = len(qubits)
    qc.h(qubits)
    qc.x(qubits)
    target = qubits[-1]
    controls = qubits[:-1]
    qc.h(target)
    if len(controls) == 0:
        qc.z(target)
    else:
        qc.append(MCXGate(len(controls)), controls + [target])
    qc.h(target)
    qc.x(qubits)
    qc.h(qubits)


M = len(classical_solutions)  # number of marked states
theta = math.asin(math.sqrt(M / (2 ** N)))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N, N)
qubits = list(range(N))
qc.h(qubits)
for _ in range(iterations):
    apply_oracle(qc, classical_solution_strings, qubits)
    apply_diffuser(qc, qubits)
qc.measure(qubits, qubits)

print(f"Grover iterations used: {iterations} (M={M} marked out of {2**N})")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
result = backend.run(qc, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("Top measured outcomes:", sorted_counts[:min(len(sorted_counts), M + 3)])

marked_mass = sum(c for bstr, c in counts.items() if bstr in classical_solution_strings) / shots
uniform_baseline = M / (2 ** N)

# The top-M most frequent measured bitstrings should be exactly the
# classical solution set, and the amplified probability mass on marked
# states should be well above the pre-amplification uniform baseline.
top_M_strings = {bstr for bstr, _ in sorted_counts[:M]}

quantum_matches_classical = (
    top_M_strings == classical_solution_strings
    and marked_mass > 3 * uniform_baseline
)

print(f"Marked-state probability mass: {marked_mass:.3f} "
      f"(uniform baseline would be {uniform_baseline:.3f})")
print(f"Top-{M} measured set == classical solution set: "
      f"{top_M_strings == classical_solution_strings}")

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
