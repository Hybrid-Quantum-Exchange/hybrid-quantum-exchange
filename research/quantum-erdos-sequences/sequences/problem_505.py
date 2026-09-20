"""
Erdos problem #505 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, entry
"number: '505'"): informal_status "disproved" (Lean-formalized, 2026-02-02),
comments "Borsuk's problem", tags ["geometry"], oeis: ["possible"].

LIMITATION, stated honestly up front: the dataset's `oeis` field for this
problem is the literal string "possible" -- not an actual OEIS sequence id.
Borsuk's problem ("can every bounded set in R^n be partitioned into n+1
parts of strictly smaller diameter?") is a geometric question, not one that
is naturally indexed by an OEIS integer sequence, and its disproof (the
counterexample dimension is in the thousands) is not a small computable
instance a few qubits can touch. There is no genuine OEIS sequence to test
here, so this script does NOT test an OEIS membership/term property -- doing
so would mean fabricating one, which the task explicitly forbids.

Instead, this is the best honest small, finite, computable property that is
actually faithful to Borsuk's problem's real mathematical content, checked
classically first and then verified with a genuine quantum search:

  Classical fact used (Borsuk, n=2, the plane): a set of diameter 1 need
  not split into 2 parts of strictly smaller diameter -- 3 parts (n+1) are
  required in general. The standard witness is an equilateral triangle of
  side length 1 (diameter 1): any partition of its 3 vertices into 2
  non-empty groups must put two vertices in the same group, and every pair
  of vertices is at distance exactly 1 = the diameter, so that group's
  diameter is NOT smaller. Hence there is NO valid 2-partition, matching
  the known Borsuk number for the plane being 3 = n+1.

  This is a small, finite, brute-force-checkable search problem: over all
  2^3 = 8 ways to assign the 3 triangle vertices to 2 labeled groups
  {0,1}, is there any assignment where every group of size >= 2 has all
  pairwise distances strictly less than the overall diameter (1)? The
  classical answer, computed here from first principles (exact distances,
  no OEIS lookup), is: NO valid assignment exists (0 out of 8).

Quantum circuit: a 3-qubit Grover search over all 8 vertex-to-group
assignments, with a phase oracle that flags an assignment as "good" iff it
is a valid split (every group of size >= 2 has sub-diameter < 1, i.e. no
group contains two points at distance 1). Since the oracle marks the empty
set (0 solutions), Grover amplitude amplification has nothing to amplify:
after any number of Grover iterations the measured distribution stays
(ideally) exactly uniform over the 8 basis states, which is itself a
non-trivial, checkable quantum outcome -- it is the case Grover degenerates
to identity-on-phase because M=0. The script verifies this: it computes the
ideal statevector after 0 Grover iterations and after 1 Grover iteration on
the AerSimulator (statevector method) and checks that the probability
distribution is (numerically) unchanged and uniform, matching the classical
brute-force count of 0 valid splits.

PASS criterion: classical brute force finds 0 valid 2-splits AND the
simulated quantum probability distribution over all 8 assignments stays
uniform (within numerical tolerance) after Grover iteration, confirming the
oracle marked nothing -- i.e. the quantum circuit correctly "discovers"
(by finding no amplification) that no valid 2-partition exists, consistent
with the classical Borsuk fact that the plane needs 3 parts, not 2.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile


# ---------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS lookup)
# ---------------------------------------------------------------------

# Equilateral triangle, side length 1 -> diameter 1.
POINTS = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (0.5, math.sqrt(3) / 2),
}


def dist(a, b):
    ax, ay = POINTS[a]
    bx, by = POINTS[b]
    return math.hypot(ax - bx, ay - by)


DIAMETER = max(dist(a, b) for a, b in itertools.combinations(POINTS, 2))
assert abs(DIAMETER - 1.0) < 1e-9

EPS = 1e-9


def is_valid_split(assignment):
    """assignment: tuple of 3 group labels (0/1) for vertices 0,1,2.

    Valid iff every group of size >= 2 has all pairwise distances
    strictly less than DIAMETER (i.e. genuinely smaller diameter).
    """
    groups = {0: [], 1: []}
    for vertex, label in enumerate(assignment):
        groups[label].append(vertex)
    for members in groups.values():
        if len(members) < 2:
            continue
        for a, b in itertools.combinations(members, 2):
            if dist(a, b) >= DIAMETER - EPS:
                return False
    return True


def classical_brute_force():
    valid = []
    for bits in itertools.product([0, 1], repeat=3):
        if is_valid_split(bits):
            valid.append(bits)
    return valid


VALID_SPLITS = classical_brute_force()
CLASSICAL_VALID_COUNT = len(VALID_SPLITS)

print(f"Diameter of triangle: {DIAMETER:.6f}")
print(f"Classical brute force over 2^3=8 assignments: "
      f"{CLASSICAL_VALID_COUNT} valid 2-splits found (expect 0).")
print(f"Valid splits: {VALID_SPLITS}")


# ---------------------------------------------------------------------
# 2. Quantum oracle + Grover circuit
# ---------------------------------------------------------------------
#
# 3 qubits q0,q1,q2 encode the group label of vertices 0,1,2. The oracle
# must flip the phase of basis state |b2 b1 b0> iff is_valid_split is
# True for that assignment. Since we know classically there are zero such
# states, we build the oracle generically from the truth table (as a
# real quantum circuit -- not hard-coded to "do nothing") using a
# multi-controlled-Z per valid assignment; with zero valid assignments,
# the oracle circuit ends up being the identity (0 marking gates), which
# is itself the object under test, not an assumption.

N_QUBITS = 3


def build_oracle():
    qc = QuantumCircuit(N_QUBITS, name="Oracle")
    for bits in itertools.product([0, 1], repeat=3):
        if not is_valid_split(bits):
            continue
        # Multi-controlled Z marking |bits> (bits[i] is group label of
        # vertex i, used directly as qubit i's control polarity).
        flip_qubits = [i for i, b in enumerate(bits) if b == 0]
        for q in flip_qubits:
            qc.x(q)
        if N_QUBITS == 1:
            qc.z(0)
        else:
            qc.h(N_QUBITS - 1)
            qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
            qc.h(N_QUBITS - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc


def build_diffuser():
    qc = QuantumCircuit(N_QUBITS, name="Diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


def build_grover_circuit(iterations):
    qc = QuantumCircuit(N_QUBITS)
    qc.h(range(N_QUBITS))
    oracle = build_oracle()
    diffuser = build_diffuser()
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(N_QUBITS))
        qc.append(diffuser.to_gate(), range(N_QUBITS))
    return qc


# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator (statevector) and verify
# ---------------------------------------------------------------------

def probabilities(qc):
    sv = Statevector.from_instruction(qc)
    probs = np.abs(sv.data) ** 2
    return probs


backend = AerSimulator(method="statevector")

qc0 = build_grover_circuit(iterations=0)
qc1 = build_grover_circuit(iterations=1)

# Sanity: run through the actual AerSimulator backend (not just
# Statevector helper) to exercise the real quantum simulation path.
qc1_meas = qc1.copy()
qc1_meas.save_statevector()
compiled = transpile(qc1_meas, backend)
result = backend.run(compiled).result()
sim_statevector = np.asarray(result.get_statevector(compiled))
sim_probs = np.abs(sim_statevector) ** 2

probs0 = probabilities(qc0)
probs1 = probabilities(qc1)

uniform = np.full(2 ** N_QUBITS, 1.0 / (2 ** N_QUBITS))

print("Probabilities after 0 Grover iterations:", np.round(probs0, 6))
print("Probabilities after 1 Grover iteration (Statevector):",
      np.round(probs1, 6))
print("Probabilities after 1 Grover iteration (AerSimulator):",
      np.round(sim_probs, 6))

TOL = 1e-6
matches_uniform_before = np.allclose(probs0, uniform, atol=TOL)
matches_uniform_after = np.allclose(probs1, uniform, atol=TOL)
sim_matches_statevector = np.allclose(sim_probs, probs1, atol=1e-5)

quantum_ok = matches_uniform_before and matches_uniform_after and sim_matches_statevector
classical_ok = (CLASSICAL_VALID_COUNT == 0)

verified = quantum_ok and classical_ok

print()
print(f"Classical check (0 valid 2-splits expected): {classical_ok}")
print(f"Quantum check (oracle marks nothing -> distribution stays "
      f"uniform, backend matches statevector): {quantum_ok}")

if verified:
    print("PASS")
else:
    print("FAIL")
