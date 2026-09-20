"""
Erdos problem #115 -- quantum-testable lane.

Source: /home/user/manman4/erdosproblems/data/problems.yaml, entry
    - number: "115"
      oeis: ["N/A"]
      tags: ["polynomials", "analysis"]

Erdos problem 115 (proved, formalized in Lean as of 2026-03-03/2026-08-03) is
a statement about real polynomials / analysis and carries NO OEIS sequence id
in the source data (oeis: ["N/A"]). There is therefore no integer sequence
attached to this problem to build a genuine "membership of an integer in the
sequence" or "counting an early term" quantum test from -- the task's own
instructions for this case are: attempt something honest, note the
limitation clearly, and report ran_ok / verified_against_classical
accurately rather than faking a pass.

Best honest attempt made here:

Since the problem's only structure available is its "polynomials" tag, this
script tests a small, finite, genuinely computable *polynomial* property
that has real mathematical content (it is unrelated in content to the
specific claim of problem 115 itself, because problem 115 has no attached
finite decision problem to quantize):

    Property P(x): for the fixed polynomial f(x) = x^2 mod 15,
    is f(x) == 1 ?

    i.e. "is x a square root of 1 modulo 15", searched over the finite
    domain x in {0, 1, ..., 15} (4 qubits).

This is computed first from first principles in plain Python (the classical
ground truth), then the same finite search is performed with a real Grover
search circuit built and executed on Qiskit's ideal AerSimulator. The
quantum result (highest-probability measured x values) is compared against
the classical solution set and the script prints PASS/FAIL accordingly.

LIMITATION (stated plainly, per instructions): this circuit does NOT test
any OEIS sequence tied to Erdos problem 115, because none exists in the
source data for that problem. It is a small arithmetic/polynomial oracle
built in the same spirit (finite, computable, genuinely quantum-searched),
offered as the best honest substitute for a problem with no attached
sequence.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N_QUBITS = 4
DOMAIN_SIZE = 2 ** N_QUBITS  # 16, x in [0, 15]
MODULUS = 15
TARGET = 1


def classical_property(x: int) -> bool:
    """P(x): x^2 mod MODULUS == TARGET."""
    return (x * x) % MODULUS == TARGET


classical_solutions = sorted(x for x in range(DOMAIN_SIZE) if classical_property(x))
print(f"Classical search over x in [0, {DOMAIN_SIZE - 1}]: "
      f"x^2 mod {MODULUS} == {TARGET}")
print(f"Classical solutions: {classical_solutions}")
assert classical_solutions, "expected at least one classical solution"


# ---------------------------------------------------------------------------
# 2. Build a Grover oracle that marks exactly the classical solutions.
# ---------------------------------------------------------------------------
#
# The oracle is built directly from the classical solution set (a standard,
# legitimate way to construct a Grover oracle for an arbitrary finite
# boolean function of a small register: phase-flip every basis state whose
# index is a solution). This is not "faking" the property -- the property
# itself (x^2 mod 15 == 1) was checked classically above, and the oracle is
# mechanically derived from that check, term by term, the same way a
# black-box oracle for f would be compiled from f's truth table.

def build_oracle(qc: QuantumCircuit, qubits, solutions):
    """Phase-flip (multi-controlled Z, via X-sandwiched MCZ) each solution index."""
    for sol in solutions:
        bits = format(sol, f"0{len(qubits)}b")  # MSB first
        # Open-control on the 0-bits of this solution by surrounding with X gates.
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(qubits[i])
        # Multi-controlled Z on all qubits (controls = all but last, target = last).
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(qubits[i])


def build_diffuser(qc: QuantumCircuit, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


num_solutions = len(classical_solutions)
# Optimal number of Grover iterations for this domain/solution-count.
theta = np.arcsin(np.sqrt(num_solutions / DOMAIN_SIZE))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))

qc.h(qubits)  # uniform superposition over all 16 basis states
for _ in range(iterations):
    build_oracle(qc, qubits, classical_solutions)
    build_diffuser(qc, qubits)
qc.measure(qubits, qubits)

print(f"Grover iterations used: {iterations}")

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings with qubit 0 as the rightmost character; convert
# back to integers consistent with the oracle's qubit ordering (qubits[0] is
# the MSB position used in build_oracle's `bits` string, so we must match
# that convention when decoding).
measured_counts = {}
for bitstring, c in counts.items():
    # bitstring is c[q3]c[q2]c[q1]c[q0] as Qiskit prints (classical bit 0 rightmost)
    # our classical register bit i holds qubits[i]; qubits[0] was treated as MSB
    # in build_oracle's `format(sol, ...)` (bits[0] -> qubits[0]). Qiskit's
    # printed string has classical bit (N-1) leftmost, bit 0 rightmost, i.e.
    # leftmost char corresponds to qubits[N-1], rightmost to qubits[0].
    # To recover `sol` consistent with build_oracle's bit-to-qubit mapping,
    # reverse the printed string so index 0 (qubits[0]) is first, then read
    # as the same MSB-first binary string used to build the oracle.
    reordered = bitstring[::-1]  # reordered[i] == outcome of qubits[i]
    x = int(reordered, 2)
    measured_counts[x] = measured_counts.get(x, 0) + c

sorted_counts = sorted(measured_counts.items(), key=lambda kv: -kv[1])
print(f"Top measured outcomes: {sorted_counts[:6]}")

top_k = sorted_counts[: len(classical_solutions)]
measured_top_solutions = sorted(x for x, _ in top_k)

quantum_matches_classical = measured_top_solutions == classical_solutions

# Also require the marked solutions collectively dominate the distribution
# (amplitude was genuinely amplified, not just accidentally top-ranked).
solution_shots = sum(measured_counts.get(x, 0) for x in classical_solutions)
amplification_ok = solution_shots / shots > 0.5

verified = quantum_matches_classical and amplification_ok

print(f"Measured top-{len(classical_solutions)} solution set: {measured_top_solutions}")
print(f"Fraction of shots landing on a true solution: {solution_shots / shots:.3f}")

if verified:
    print("PASS")
else:
    print("FAIL")
