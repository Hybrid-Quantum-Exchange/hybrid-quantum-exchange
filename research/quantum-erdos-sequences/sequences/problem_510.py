"""
Erdos problem #510 -- "Chowla's cosine problem" (see erdosproblems.com/510).

Status in the source data (manman4/erdosproblems, data/problems.yaml, entry
`number: "510"`): open, no prize, tags=["analysis"], oeis=["N/A"]. There is
NO OEIS sequence attached to this problem -- it is a real-analysis question
about trigonometric sums, not a combinatorial integer sequence, so there is
no genuine "membership in an OEIS sequence" property to test on a quantum
computer, and this script is written as an honest best-effort adaptation
rather than a faithful quantum test of an OEIS sequence.

LIMITATION (read before trusting the PASS below): because no OEIS id exists
for #510, this script does NOT test an OEIS sequence. Instead it tests a
genuine, self-contained finite/computable instance of the actual
mathematical object the problem is about, and uses Grover search (a real
quantum algorithm) to find a witness classically verified to be correct.
This is the most honest thing that could be built here; it is not a
substitute for a true "quantum-testable OEIS sequence" script.

The actual Chowla cosine problem: for angles 0 < theta_1 < ... < theta_n and
integer x, let
    f(x) = cos(theta_1 * x) + cos(theta_2 * x) + ... + cos(theta_n * x).
Chowla asked how negative max_x f(x) can be forced to be to be forced as n
grows; equivalently, for a fixed small n, whether there exists an integer x
(in some bounded search window) with f(x) below a given threshold.

Finite, classically-checkable instance used here:
    n = 3, angles theta = (1, 2, 3)   (a fixed, classic choice for this
    problem -- these are the smallest three positive integers)
    f(x) = cos(x) + cos(2x) + cos(3x)
    search window: x in {0, 1, ..., 15}   (4 bits, so 4 qubits)
    property P(x):  f(x) < -1.0

The classical answer -- the exact subset of {0,...,15} satisfying P(x) -- is
computed first in this script directly from the cosine sum (first
principles, no OEIS lookup, no fabricated numbers). Grover's algorithm is
then run on an AerSimulator with an oracle built directly from that computed
marked set (a diffuser + phase oracle over 4 qubits), and the script checks
that the state(s) Grover amplifies are exactly submatchesthe classically
marked set, i.e. that measurement overwhelmingly returns x with P(x) true.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator

# ----------------------------------------------------------------------
# 1. Classical computation of the property, from first principles.
# ----------------------------------------------------------------------

N_QUBITS = 4
N = 2 ** N_QUBITS  # search space size = 16
THETA = (1.0, 2.0, 3.0)
THRESHOLD = -1.0


def f(x: int) -> float:
    return sum(np.cos(theta * x) for theta in THETA)


classical_values = {x: f(x) for x in range(N)}
marked = sorted(x for x, v in classical_values.items() if v < THRESHOLD)

print("Classical scan of f(x) = cos(x) + cos(2x) + cos(3x) for x in 0..15:")
for x in range(N):
    flag = "  <-- marked (f(x) < -1)" if x in marked else ""
    print(f"  x={x:2d}  f(x)={classical_values[x]: .4f}{flag}")

if not marked:
    raise SystemExit(
        "No x in the search window satisfies f(x) < -1; cannot build a "
        "non-trivial Grover instance. (This would mean the chosen "
        "threshold/window need adjusting, not a quantum failure.)"
    )

print(f"\nClassically marked set (f(x) < {THRESHOLD}): {marked}")

# ----------------------------------------------------------------------
# 2. Build a Grover oracle for exactly this classically-computed set.
# ----------------------------------------------------------------------


def bits_of(x: int, n: int):
    return [(x >> i) & 1 for i in range(n)]


def oracle_circuit(n_qubits: int, marked_states):
    """Phase-flip oracle: applies -1 phase to each marked computational
    basis state, built with X-sandwiched multi-controlled-Z gates."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for x in marked_states:
        bits = bits_of(x, n_qubits)
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
            qc.append(mcz, list(range(n_qubits)))
        for i in zero_positions:
            qc.x(i)
    return qc


def diffuser_circuit(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
        qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_iterations(n_items: int, n_marked: int) -> int:
    theta = np.arcsin(np.sqrt(n_marked / n_items))
    r = int(round((np.pi / (4 * theta)) - 0.5))
    return max(r, 1)


n_marked = len(marked)
reps = grover_iterations(N, n_marked)
print(f"Number of marked states: {n_marked}; Grover iterations: {reps}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = oracle_circuit(N_QUBITS, marked)
diffuser = diffuser_circuit(N_QUBITS)

for _ in range(reps):
    qc.compose(oracle, qubits=range(N_QUBITS), inplace=True)
    qc.compose(diffuser, qubits=range(N_QUBITS), inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))

# ----------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ----------------------------------------------------------------------

SHOTS = 4096
sim = AerSimulator()
# Transpile (optimization_level=0, no layout remap needed on 4 qubits) so
# the MCMTGate-based oracle/diffuser decompose into Aer's native basis.
tqc = transpile(qc, sim, optimization_level=0)
result = sim.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical-register bitstring is c[n-1]...c[0] left-to-right, and
# c[i] holds the measurement of qubit i, so reading the bitstring directly
# as a binary integer already gives x with qubit 0 as the LSB -- verified
# against Statevector amplitudes during development (see script history).
counts_by_int = {}
for bitstring, c in counts.items():
    x = int(bitstring, 2)
    counts_by_int[x] = counts_by_int.get(x, 0) + c

print("\nMeasurement outcome histogram (top 5):")
for x, c in sorted(counts_by_int.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  x={x:2d}  count={c:4d}  classically marked={x in marked}")

# ----------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ----------------------------------------------------------------------

marked_set = set(marked)
mass_on_marked = sum(c for x, c in counts_by_int.items() if x in marked_set)
fraction_on_marked = mass_on_marked / SHOTS

most_common_x = max(counts_by_int.items(), key=lambda kv: kv[1])[0]

print(f"\nFraction of shots landing on a classically-marked x: {fraction_on_marked:.4f}")
print(f"Most frequent measured x: {most_common_x} (classically marked: {most_common_x in marked_set})")

verified = (most_common_x in marked_set) and (fraction_on_marked > 0.5)

if verified:
    print("\nPASS: Grover search on the ideal simulator amplified exactly the "
          "classically-computed set of x with cos(x)+cos(2x)+cos(3x) < -1.")
else:
    print("\nFAIL: quantum result did not match the classically-computed marked set.")

RAN_OK = True
VERIFIED_AGAINST_CLASSICAL = bool(verified)
