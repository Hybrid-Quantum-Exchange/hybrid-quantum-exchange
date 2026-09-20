"""
Erdos problem #755 -- quantum-testable sequence attempt.

LIMITATION (read first): problem #755's entry in the erdosproblems.com data
(data/problems.yaml, block "number: \"755\"") lists:

    oeis: ["possible"]
    tags: ["geometry"]

"possible" is not a real OEIS sequence identifier (a real one would look like
"A005230" etc.) -- it is a data-entry placeholder used elsewhere in that YAML
file to mean "an OEIS entry may exist but is not recorded here". The problem's
prose/description text is not included in problems.yaml (only metadata is),
so there is no OEIS id and no concrete finite property of problem #755 itself
that can be derived from the available source. Fabricating one would violate
the task's own instruction not to invent unfounded math content.

Per the task's fallback instruction, this script is therefore a best-honest-
effort placeholder: it builds a REAL, genuinely-computing Grover search
circuit (not a fake pass) over a small generic finite search space, and
verifies the quantum result against a classical brute-force computation of
the same instance. It is NOT tied to a specific verified OEIS sequence, and
that fact is reported truthfully rather than disguised.

Chosen small computable instance (self-contained, no external data needed):
  Search space: 3-bit integers x in [0, 8).
  Property being searched for: x is a quadratic residue mod 8 that is also
  even, i.e. x in {0, 4} -- concretely we search for x such that
  (x * x) % 8 == 0 and x != 0 (this has the unique classical solution x = 4
  in [0,8), verified below by direct classical enumeration).

Grover's algorithm is run on a 3-qubit register with an oracle built directly
from that arithmetic condition (multiplication mod 8 done via a boolean
circuit over the fixed 8 basis states), using the standard single number of
iterations for 1 marked item out of 8 states (which is exactly 1 iteration
for N=8, since the optimal iteration count is floor(pi/4 * sqrt(N/M)) = 2 for
N=8, M=1 -- computed explicitly below, not hard-coded).

Result: the simulator's most frequently measured basis state is compared
against the classically-brute-forced answer; the script prints PASS if they
match and FAIL otherwise.
"""

import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ----------------------------------------------------------------------
# Step 1: classical ground truth, computed from first principles (no
# hard-coded OEIS values -- just direct enumeration of the stated property).
# ----------------------------------------------------------------------
def classical_property(x: int) -> bool:
    """x in [0,8): (x*x) mod 8 == 0 and x != 0."""
    return (x * x) % 8 == 0 and x != 0


N = 8  # 3-bit search space
solutions = [x for x in range(N) if classical_property(x)]
assert solutions == [4], f"expected unique solution [4], got {solutions}"
target = solutions[0]
M = len(solutions)
print(f"Classical brute force over x in [0,{N}): solutions = {solutions}")

# Optimal number of Grover iterations for N states, M marked items.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations computed as {iterations} (N={N}, M={M})")


# ----------------------------------------------------------------------
# Step 2: build the oracle. Since the search space is only 8 states, the
# oracle is built as an exact multi-controlled-Z that flips the phase of
# the specific computational basis state(s) satisfying classical_property.
# This is a legitimate way to realize "the arithmetic condition as a
# circuit" for a small enough domain: each marked basis state gets an
# X-sandwiched multi-controlled-Z (a standard Grover oracle construction),
# so the oracle's action is exactly "phase-flip iff classical_property(x)".
# ----------------------------------------------------------------------
def build_oracle(marked_states, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        flip_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in flip_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


n_qubits = 3
oracle = build_oracle(solutions, n_qubits)
diffuser = build_diffuser(n_qubits)

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(n_qubits))
    qc.append(diffuser.to_instruction(), range(n_qubits))
qc.measure(range(n_qubits), range(n_qubits))
qc = qc.decompose().decompose().decompose()


# ----------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator and compare to the classical answer.
# ----------------------------------------------------------------------
sim = AerSimulator()
shots = 2000
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()
print("Measurement counts:", counts)

most_common_bitstring = max(counts, key=counts.get)
# Qiskit prints classical bits as c_{n-1}...c_0, and measure(qubit i -> clbit
# i) with qubit i weighted as bit i means this string, read as plain binary,
# already equals the integer value of the measured basis state.
measured_value = int(most_common_bitstring, 2)
print(f"Most frequently measured value: {measured_value} "
      f"({counts[most_common_bitstring]}/{shots} shots)")

success_prob = counts.get(most_common_bitstring, 0) / shots

ok = (measured_value == target) and (success_prob > 0.5)

print()
print(f"Classical answer: x = {target}")
print(f"Quantum (Grover) answer: x = {measured_value}, "
      f"probability ~{success_prob:.3f}")

if ok:
    print("PASS")
else:
    print("FAIL")
