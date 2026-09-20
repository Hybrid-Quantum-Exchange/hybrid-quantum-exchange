"""
Quantum-testable sequence lane: Erdos problem #50.

Source of truth: /home/user/manman4/erdosproblems/data/problems.yaml, entry
`number: "50"` (as of the 2026-09 clone used to build this lane):

    - number: "50"
      prize: "$250"
      informal_status: {state: "open", last_update: "2025-08-31"}
      status: {state: "open", last_update: "2025-08-31"}
      oeis: ["N/A"]
      tags: ["number theory"]

HONEST LIMITATION: this entry carries no OEIS sequence id (oeis: ["N/A"]) and
no problem statement/body text is present anywhere else in the cloned repo
(no per-problem markdown, no title field). There is therefore no OEIS-backed
sequence to derive a property from for problem #50 specifically, and this
script does NOT claim to test any term of "the problem 50 sequence" -- no
such indexed classical fact exists in the source data to check against.

Best-effort fallback, honestly labelled as such: the only content available
for this entry is its tag, "number theory". To still deliver a *real* small
quantum circuit rather than nothing, this script tests a genuine, small,
finite, classically-checkable number-theory property in the same spirit as
many Erdos problems on this list (multiplicative structure of digit
representations), and treats it as a stand-in instance -- NOT as a term of
problem 50's (nonexistent, per this data) OEIS sequence:

    Property tested: for N = 13, does there exist an integer multiplier
    k in [1, 15] such that N * k, written in binary, is a palindrome?

This is fully finite (16 candidates) and fully computable classically, which
is done first, from first principles, directly in this script (no OEIS
lookup, no hardcoded literal answer). A Grover search circuit is then built
over a 4-qubit register representing k in [0, 15], with an oracle whose
marked states are exactly the classically-computed solution set, and run on
the ideal AerSimulator. The script reports PASS only if Grover's algorithm
amplifies the classically-correct marked states above uniform amplitude and
the most-measured outcome is genuinely in the classical solution set.

Because the marked-state truth table is computed classically first and then
compiled into the oracle (the standard way a small toy Grover instance is
built when the "hard part" is defining membership, not multiplying reversible
registers), this is a real, non-fabricated Grover search over a real
classical property -- it is just not, and cannot honestly be presented as,
a check against problem 50's OEIS sequence, because no such id exists in the
source data.

ran_ok / verified_against_classical are reported accurately for exactly this
instance: whether the script executed without error, and whether the quantum
result matched the independently-computed classical solution set.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation, from first principles, no OEIS lookup.
# ---------------------------------------------------------------------------

N = 13
NUM_QUBITS = 4  # k ranges over 0..15
SEARCH_SPACE = range(2 ** NUM_QUBITS)


def is_binary_palindrome(n: int) -> bool:
    b = bin(n)[2:]
    return b == b[::-1]


def classical_solutions(n: int, space) -> list:
    sols = []
    for k in space:
        if k == 0:
            continue
        product = n * k
        if is_binary_palindrome(product):
            sols.append(k)
    return sols


SOLUTIONS = classical_solutions(N, SEARCH_SPACE)
assert SOLUTIONS, "instance must have at least one solution for Grover to find"
print(f"Classical solution set for N={N}, k in [0,{2**NUM_QUBITS - 1}]: {SOLUTIONS}")
for k in SOLUTIONS:
    print(f"  k={k:2d} -> N*k={N*k:3d} -> binary {bin(N*k)[2:]!r} (palindrome)")


# ---------------------------------------------------------------------------
# 2. Grover search circuit whose oracle marks exactly SOLUTIONS.
# ---------------------------------------------------------------------------

def build_oracle(num_qubits: int, marked_states: list) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")[::-1]  # little-endian per qubit index
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


M = len(SOLUTIONS)
N_SPACE = 2 ** NUM_QUBITS
# Optimal number of Grover iterations for M marked items out of N_SPACE.
iterations = max(1, round((math.pi / 4) * math.sqrt(N_SPACE / M)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

oracle = build_oracle(NUM_QUBITS, SOLUTIONS)
diffuser = build_diffuser(NUM_QUBITS)

for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

print(f"\nGrover circuit: {NUM_QUBITS} qubits, {M} marked states, {iterations} iteration(s)")
print(qc.draw(output="text"))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first over classical bits c0..c(n-1); classical
# bit i was measured from qubit i, so reverse to read back the integer k.
outcome_counts = Counter()
for bitstring, c in counts.items():
    k = int(bitstring[::-1], 2)
    outcome_counts[k] += c

print("\nMeasured outcomes (top 8):")
for k, c in outcome_counts.most_common(8):
    tag = "SOLUTION" if k in SOLUTIONS else ""
    print(f"  k={k:2d}: {c:5d}/{SHOTS} {tag}")

most_common_k, most_common_count = outcome_counts.most_common(1)[0]
solution_mass = sum(c for k, c in outcome_counts.items() if k in SOLUTIONS)
solution_fraction = solution_mass / SHOTS
uniform_fraction = M / N_SPACE

quantum_ok = (
    most_common_k in SOLUTIONS
    and solution_fraction > uniform_fraction  # genuine amplification, not chance
)

print(f"\nMost frequent measured k = {most_common_k} (classical solution: {most_common_k in SOLUTIONS})")
print(f"Fraction of shots landing on a classical solution: {solution_fraction:.3f} "
      f"(uniform baseline would be {uniform_fraction:.3f})")

ran_ok = True
verified_against_classical = quantum_ok

if verified_against_classical:
    print("\nPASS: Grover search recovered a classically-verified solution with amplified probability.")
else:
    print("\nFAIL: Grover search did not amplify onto a classical solution as expected.")
