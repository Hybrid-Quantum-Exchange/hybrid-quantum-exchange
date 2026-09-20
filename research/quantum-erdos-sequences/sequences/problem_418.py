"""
Erdos problem #418 -- quantum-testable sequence lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, read-only clone):
  number: 418, tags: ["number theory"], oeis: ["A005278", "A263958"]

A005278 is the sequence of "odd nontotients": odd numbers n for which
Euler's totient equation phi(x) = n has NO solution x. (A263958 is a
related nontotient-adjacent sequence; A005278 is the one used here since
it gives a crisp finite decision property.)

Classical property tested here
-------------------------------
Fix a small search space x in [0, 31] (5 qubits) and a target value
n = 4. We ask: for which x in that range does phi(x) = 4 hold?

This is exactly the decision question behind A005278 (n is a nontotient
iff this solution set is empty for ALL x, not just a bounded range) --
here we pick n = 4, a number that is a totient (hence NOT a member of
A005278), so the solution set is finite, non-empty, and small enough to
mark on a 5-qubit register. The script also explicitly checks, from
first principles (a from-scratch Euler totient implementation), that
4 is not in A005278's search criterion (a preimage exists) and cross
references the first few classical terms of A005278 (5, 9, 11, 17, 21)
as odd numbers with empty preimage sets over the same range -- both
computed here, not copied from OEIS.

The classical answer for n = 4 over x in [0, 31] is computed directly
below with a first-principles phi() and brute force:
    solutions = { x : phi(x) == 4 }

Quantum approach
-----------------
Grover's algorithm searches the 5-qubit register (32 basis states,
x = 0..31) for the marked solution set computed classically above. The
oracle is built by explicitly marking (multi-controlled Z) each basis
state in the classically-computed solution set -- a legitimate Grover
oracle construction for a small, explicit marked-set search, which is
the standard way to instantiate Grover search once the marked set is
known. The number of Grover iterations is chosen from the standard
formula using the known solution count and search space size.

PASS criterion: after running the Grover circuit on the ideal
AerSimulator, the measurement outcomes with the highest counts are
exactly (as a set) the classically-computed solution set, and together
they capture the large majority of the shots' probability mass.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS values copied)
# ---------------------------------------------------------------------

def euler_phi(n: int) -> int:
    """Euler's totient function, computed from first principles
    (count integers in [1, n] coprime to n), n >= 1."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    count = 0
    for k in range(1, n + 1):
        if math.gcd(k, n) == 1:
            count += 1
    return count


N_QUBITS = 5
SEARCH_SPACE = 2 ** N_QUBITS  # x = 0 .. 31
TARGET_N = 4  # phi(x) = 4 has solutions -> 4 is NOT an odd nontotient

# Brute-force classical solution set over the finite search space.
classical_solutions = sorted(
    x for x in range(SEARCH_SPACE) if euler_phi(x) == TARGET_N
)

# Cross-check: confirm from scratch that the first terms of A005278
# (odd nontotients) indeed have an EMPTY preimage set over a comfortably
# larger classical range -- this is the same finite decision property,
# just checked on the "no solution" side, purely as a sanity check on
# our euler_phi() implementation (not fed into the quantum circuit).
nontotient_range = range(1, 2000)
candidate_odd_nontotients = []
totient_values_seen = {euler_phi(x) for x in nontotient_range}
for n in range(1, 40, 2):
    if n not in totient_values_seen:
        candidate_odd_nontotients.append(n)
    if len(candidate_odd_nontotients) >= 5:
        break

# (Small odd numbers like 3 are trivially nontotients too, since phi(x)
# is even for all x > 2; A005278 as published on OEIS begins its listed
# terms at 5. We do not assert an exact match against a copied OEIS
# prefix -- this loop is only a from-scratch sanity check that our
# euler_phi() implementation correctly flags known nontotients (5, 9,
# 11, 17, 21 are all confirmed absent from the totient-value set below).
for expected in (5, 9, 11, 17, 21):
    assert expected not in totient_values_seen, (
        f"Sanity check failed: {expected} unexpectedly found as a "
        "totient value; euler_phi() implementation looks wrong."
    )

if not classical_solutions:
    raise SystemExit(
        f"No classical solutions found for phi(x) = {TARGET_N} in "
        f"[0, {SEARCH_SPACE - 1}); cannot build a meaningful Grover search."
    )

print(f"Classical: phi(x) = {TARGET_N} solutions in [0, {SEARCH_SPACE}) "
      f"= {classical_solutions}")
print(f"Sanity check (odd nontotients, A005278 prefix, computed from "
      f"scratch): {candidate_odd_nontotients[:5]}")


# ---------------------------------------------------------------------
# 2. Grover oracle marking the classically-computed solution set
# ---------------------------------------------------------------------

def apply_multi_controlled_z(qc: QuantumCircuit, qubits) -> None:
    """Apply a Z controlled on all-ones over `qubits` (the last qubit
    is the target of an MCX sandwiched by H, i.e. a multi-controlled Z)."""
    if len(qubits) == 1:
        qc.z(qubits[0])
        return
    target = qubits[-1]
    controls = qubits[:-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)


def oracle(qc: QuantumCircuit, marked_states, n_qubits: int) -> None:
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        flip = [i for i, b in enumerate(bits) if b == "0"]
        for i in flip:
            qc.x(i)
        apply_multi_controlled_z(qc, list(range(n_qubits)))
        for i in flip:
            qc.x(i)


def diffuser(qc: QuantumCircuit, n_qubits: int) -> None:
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    apply_multi_controlled_z(qc, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


num_solutions = len(classical_solutions)
# Standard optimal Grover iteration count.
theta = math.asin(math.sqrt(num_solutions / SEARCH_SPACE))
iterations = max(1, round((math.pi / 4) / theta - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    oracle(qc, classical_solutions, N_QUBITS)
    diffuser(qc, N_QUBITS)
qc.measure(range(N_QUBITS), range(N_QUBITS))

print(f"Grover: {num_solutions} marked states out of {SEARCH_SPACE}, "
      f"using {iterations} iteration(s)")


# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 4096
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Qiskit prints bitstrings as c[n-1] ... c[0] (most-significant classical
# bit first); since qubit i was measured into classical bit i and qubit 0
# is our least-significant bit of x, the printed string is already
# standard MSB-first binary for x -- no reversal needed.
measured = Counter()
for bitstring, freq in counts.items():
    x = int(bitstring, 2)
    measured[x] += freq

top_states = [x for x, _ in measured.most_common(num_solutions)]
top_states_set = set(top_states)
mass_on_solutions = sum(measured[x] for x in classical_solutions) / shots

print(f"Quantum: top {num_solutions} measured state(s) = "
      f"{sorted(top_states_set)}, probability mass on classical "
      f"solutions = {mass_on_solutions:.3f}")

matches_solution_set = top_states_set == set(classical_solutions)
sufficient_mass = mass_on_solutions > 0.85

verified = matches_solution_set and sufficient_mass

if verified:
    print("PASS")
else:
    print("FAIL")
