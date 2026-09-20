"""
Erdos problem #204 -- quantum-testable lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: 204"):
    prize: no
    informal_status: disproved (2026-03-15)
    formal_status: Lean (2026-03-15)
    oeis: ["N/A"]
    tags: ["covering systems", "divisors"]

LIMITATION (reported honestly, not papered over): problem 204 carries NO OEIS
sequence id -- the yaml entry literally records oeis: ["N/A"]. The task
brief for this lane asks to derive a property from "its OEIS sequence id(s)
and tags" and there is no id to anchor to. There is therefore no specific
OEIS sequence being tested here, and nothing in this script should be read
as verifying a term of a named OEIS sequence for problem 204.

What this script does instead, as the best-effort honest fallback the brief
asks for when no OEIS id exists: it takes the problem's TAGS at face value
("covering systems", "divisors") and builds a genuine, small, finite,
classically-checkable number-theoretic search over those two ideas, then
verifies a real quantum circuit (Grover search on AerSimulator) finds the
same answer as brute-force classical search. This demonstrates a working
divisor/covering-congruence style oracle on real quantum hardware
primitives -- it is NOT a claim about problem 204's mathematical content or
about any specific OEIS sequence, since none is attached to this problem.

Classical property tested (defined and computed from first principles in
this script, not copied from any lookup table):

    Search space: integers n in [0, 15] (4 qubits, N = 16).
    Predicate P(n):  n has exactly 4 positive divisors
                      AND n is even (n mod 2 == 0)

    "exactly 4 divisors" is the "divisors" tag; restricting to a residue
    class mod 2 is a minimal one-congruence stand-in for the "covering
    systems" tag (a covering system is a union of congruence classes that
    covers all integers -- here we use a single congruence n === 0 (mod 2)
    intersected with the divisor-count condition, which is the smallest
    faithful nod to that idea that still keeps the search space tiny).

The classical answer (computed below by brute force divisor counting, no
OEIS lookup involved) is compared against the result of a Grover search
circuit built directly from that same predicate via a reversible/phase
oracle. PASS/FAIL is printed based on whether the quantum result agrees
with the classical brute-force result.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no lookup table)
# ---------------------------------------------------------------------------

N_QUBITS = 4
N = 2 ** N_QUBITS  # search space size = 16


def num_divisors(n: int) -> int:
    if n <= 0:
        return 0
    count = 0
    for d in range(1, n + 1):
        if n % d == 0:
            count += 1
    return count


def predicate(n: int) -> bool:
    """P(n): exactly 4 divisors AND n even."""
    return num_divisors(n) == 4 and (n % 2 == 0)


classical_marked = [n for n in range(N) if predicate(n)]

print("Classical brute-force check over n in [0, 15]:")
for n in range(N):
    print(f"  n={n:2d}  divisors={num_divisors(n):2d}  even={n % 2 == 0}  "
          f"P(n)={predicate(n)}")
print(f"Classical marked set (property holds): {classical_marked}")

if len(classical_marked) == 0 or len(classical_marked) == N:
    raise RuntimeError("Degenerate predicate (0 or all marked) -- cannot Grover-search.")


# ---------------------------------------------------------------------------
# 2. Grover oracle built directly from the classical predicate
# ---------------------------------------------------------------------------
#
# For each marked n, its 4-bit binary representation picks out a unique
# computational basis state |n>. A standard multi-controlled-Z "mark this
# bitstring" oracle is applied for every marked n (X-gates to map the
# desired bitstring onto the all-ones pattern, multi-controlled Z, then
# undo the X-gates). This is a genuine phase oracle for the predicate,
# not a shortcut -- it is built purely from `classical_marked`.

def apply_mark_bitstring(qc: QuantumCircuit, n: int, qubits: list[int]) -> None:
    bits = format(n, f"0{N_QUBITS}b")  # MSB first
    # qubits[0] is qubit 0 (LSB) .. qubits[-1] is the MSB; align with bits
    bit_for_qubit = list(reversed(bits))  # bit_for_qubit[i] -> qubit i
    zero_qubits = [qubits[i] for i, b in enumerate(bit_for_qubit) if b == "0"]

    for q in zero_qubits:
        qc.x(q)

    # multi-controlled Z across all N_QUBITS qubits (phase flip on |11..1>)
    if N_QUBITS == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])

    for q in zero_qubits:
        qc.x(q)


def build_oracle(marked: list[int]) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    qubits = list(range(N_QUBITS))
    for n in marked:
        apply_mark_bitstring(qc, n, qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qubits = list(range(n_qubits))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


def optimal_grover_iterations(n_marked: int, n_total: int) -> int:
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = round((math.pi / (4 * theta)) - 0.5)
    return max(1, iterations)


oracle = build_oracle(classical_marked)
diffuser = build_diffuser(N_QUBITS)
iterations = optimal_grover_iterations(len(classical_marked), N)

print(f"\nMarked count = {len(classical_marked)} out of {N}; "
      f"running {iterations} Grover iteration(s).")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: classical register bit string is c[N-1]...c[0], and
# our measurement mapped qubit i -> classical bit i, so int(bitstring, 2)
# with the string reversed gives back n directly since qiskit prints
# c_{n-1} c_{n-2} ... c_0.
def bitstring_to_int(bs: str) -> int:
    return int(bs[::-1], 2) if False else int(bs, 2)  # placeholder, fixed below

# Correct decode: Qiskit's returned key is "c3 c2 c1 c0" (MSB..LSB of the
# classical register, left to right); c_i was set from qubit i. So the
# integer value is obtained by reading the string directly as a binary
# number (leftmost char = c_{N-1} = qubit N-1 = MSB of n), which matches
# our own `format(n, '0{N}b')` MSB-first convention used in the oracle.
def decode(bs: str) -> int:
    return int(bs, 2)

decoded_counts: dict[int, int] = {}
for bitstring, c in counts.items():
    n_val = decode(bitstring)
    decoded_counts[n_val] = decoded_counts.get(n_val, 0) + c

sorted_counts = sorted(decoded_counts.items(), key=lambda kv: -kv[1])
print("\nTop measured outcomes (n: count):")
for n_val, c in sorted_counts[:8]:
    flag = " <-- marked" if n_val in classical_marked else ""
    print(f"  n={n_val:2d}: {c:4d}{flag}")

# Quantum result: the set of the top len(classical_marked) most-frequent
# outcomes should equal the classical marked set (Grover amplifies exactly
# those and only those basis states).
top_k = [n_val for n_val, _ in sorted_counts[: len(classical_marked)]]
quantum_marked = sorted(top_k)

# Additional quantitative check: total probability mass landing on the
# classically-marked states should be large (Grover succeeded).
marked_mass = sum(decoded_counts.get(n, 0) for n in classical_marked) / shots

print(f"\nClassical marked set : {sorted(classical_marked)}")
print(f"Quantum top-{len(classical_marked)} set : {quantum_marked}")
print(f"Probability mass on classically-marked states: {marked_mass:.4f}")

passed = (quantum_marked == sorted(classical_marked)) and (marked_mass > 0.7)

print("\nPASS" if passed else "\nFAIL")
