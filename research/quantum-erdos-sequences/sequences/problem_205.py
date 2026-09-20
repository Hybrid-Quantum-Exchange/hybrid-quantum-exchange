"""
Erdos problem #205 -- quantum-testable-sequence lane.

Source-data check (2026-09-19): in the read-only clone at
/home/user/manman4/erdosproblems/data/problems.yaml, the block for
`number: "205"` reads:

    oeis: ["possible"]
    tags: ["number theory"]
    status: disproved (Lean)

"possible" is a literal placeholder string in the metadata, not a real OEIS
A-number, and no other file in that repository (README, docs/, scripts/)
carries a description, formula, or sequence definition for problem 205.
There is therefore no genuine OEIS sequence to derive a testable classical
property from for this problem. Per the task instructions, this is reported
honestly rather than fabricated: **no OEIS id is available for Erdos problem
205**, so the "OEIS id(s) used" for this lane is NONE.

Best honest attempt in lieu of an OEIS-derived property
--------------------------------------------------------
To still deliver a genuine, non-fabricated quantum computation for this
lane (rather than an empty file), this script falls back to the one piece
of real mathematical content problem 205's metadata does carry: its tag,
"number theory". It defines a small, finite, computable number-theoretic
property that is unambiguously checkable classically -- primality over the
4-bit range [0, 15] -- and solves it with a real unstructured (Grover)
search circuit on the ideal AerSimulator.

Classical property tested
--------------------------
    P(n) := "n is prime", for n in {0, 1, ..., 15} (4 qubits, N = 16).

The classical marked set M = {n in [0,15] : n is prime} is computed in this
script from first principles (trial division), not copied from any table:
    M = {2, 3, 5, 7, 11, 13}   (|M| = 6)

Quantum circuit
----------------
A standard Grover search circuit over 4 qubits:
  1. Uniform superposition (H on all qubits).
  2. Oracle: phase-flips exactly the computational basis states in M, built
     as a multi-controlled-Z conjugated by X gates selecting each marked
     bitstring (a direct "marked-state" oracle -- genuine circuit
     construction, not a lookup table smuggled into a classical shortcut).
  3. Diffuser (inversion about the mean).
  4. Repeat for the Grover-optimal number of iterations for
     N = 16, M = 6.
  5. Measure all 4 qubits, run on AerSimulator, and check that the states
     with the highest measured probability are exactly the classically
     computed prime set M.

PASS/FAIL
----------
The script prints PASS iff the set of the |M| most-frequently measured
4-bit strings (converted to integers) equals M exactly, and the total
measured probability mass landing in M exceeds a fixed threshold
(demonstrating genuine amplitude amplification rather than a lucky
coincidence of measurement noise).
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16
MARKED = [n for n in range(N) if is_prime(n)]
assert MARKED == [2, 3, 5, 7, 11, 13], f"unexpected classical prime set: {MARKED}"
M = len(MARKED)


# ---------------------------------------------------------------------------
# 2. Grover oracle and diffuser, built as real circuits (no shortcuts).
# ---------------------------------------------------------------------------

def mark_state(qc: QuantumCircuit, value: int, n_qubits: int) -> None:
    """Phase-flip the |value> basis state via X-conjugated multi-controlled-Z."""
    bits = format(value, f"0{n_qubits}b")[::-1]  # qubit 0 = LSB
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]

    for i in zero_positions:
        qc.x(i)

    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    for i in zero_positions:
        qc.x(i)


def build_oracle(marked_values, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for v in marked_values:
        mark_state(qc, v, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# Grover-optimal iteration count for N states, M marked states.
theta = np.arcsin(np.sqrt(M / N))
n_iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(MARKED, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

for _ in range(n_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

simulator = AerSimulator()
compiled = transpile(qc, simulator)
shots = 20000
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-left over the classical register, in the
# same qubit ordering used to build the marked-state bitstrings above.
int_counts = {}
for bitstring, freq in counts.items():
    value = int(bitstring, 2)
    int_counts[value] = int_counts.get(value, 0) + freq

top_states = sorted(int_counts.items(), key=lambda kv: -kv[1])[:M]
measured_top_set = sorted(v for v, _ in top_states)

mass_in_marked = sum(int_counts.get(v, 0) for v in MARKED) / shots


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

print(f"Classical prime set M (n in [0,{N-1}]): {MARKED}")
print(f"Grover iterations used: {n_iterations}")
print(f"Top-{M} measured states (by frequency): {measured_top_set}")
print(f"Measured probability mass landing in M: {mass_in_marked:.3f}")

MASS_THRESHOLD = 0.80  # amplitude amplification should concentrate well above 6/16 = 0.375

ok = (measured_top_set == MARKED) and (mass_in_marked >= MASS_THRESHOLD)

if ok:
    print("PASS")
else:
    print("FAIL")
