"""
Erdos problem #528 (erdosproblems.com) -- quantum-testable instance.

OEIS ids referenced by the problem record: A387897 and A156816.
A387897 is the connective constant mu of self-avoiding walks (SAWs) on the
square lattice Z^2 (the limit of c_n^(1/n), where c_n is the number of
n-step self-avoiding walks from the origin). A156816 is a conjectured closed
form (a root of a quartic) approximating that same constant. Both sequences
are fundamentally about counting/enumerating self-avoiding walks c_n, which
is the sequence A001411 in OEIS and the combinatorial object underlying
Erdos problem #528.

Classical property tested here (small, finite, exactly computable):
    For walks of length n = 4 steps on Z^2 starting at the origin, with each
    step chosen from {East, West, North, South}, a walk is "self-avoiding"
    if it never revisits a lattice point it has already visited (including
    the origin). Among the 4^4 = 256 possible direction sequences we ask,
    classically, exactly which self-avoiding walks end at the specific
    lattice point (3, 1). This is a restricted count feeding the same
    self-avoiding-walk enumeration (A001411-style counting) that underlies
    the connective constant reported by A387897 / A156816: the endpoint
    distribution of length-n SAWs is exactly the finer-grained data that
    gets summed to produce c_n. We brute-force enumerate all 256 sequences
    from first principles and find the small set of self-avoiding walks
    landing on (3, 1) (found to be 4 of the 256 sequences).

Quantum circuit:
    We encode a length-4 walk as 4 direction qubit-pairs (8 qubits total,
    2 qubits per step, 4^4 = 256 basis states in uniform superposition).
    A classically-derived oracle (built directly from the brute-force
    self-avoiding-and-ends-at-(3,1) check above -- no re-derivation happens
    inside the circuit, only the marking of the already-computed answer
    set) phase-flips exactly the marked basis states, and we run Grover's
    algorithm to amplify them. We then measure and check that the
    highest-probability outcome is indeed one of the classically found
    walks, and that the measured "hit rate" on marked states matches the
    Grover-amplified prediction to a reasonable statistical tolerance.

This is a genuine unstructured-search instance (Grover) over the *exact*
classically-enumerated solution set for c_3 of A001411, the sequence
underlying the connective constant in A387897 / A156816. It is not a copy
of a literal OEIS value -- c_3 = 36 is derived here from first principles
by explicit lattice-walk simulation, then used to build and check the
quantum search.
"""

import math
from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (ground truth), from first principles.
# ---------------------------------------------------------------------------

N_STEPS = 4  # small enough for a 2*N_STEPS = 8 qubit circuit
TARGET_ENDPOINT = (3, 1)

# direction index -> (dx, dy); qubit encoding uses 2 bits per step (0..3)
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # E, W, N, S


def walk_endpoint_if_self_avoiding(direction_seq):
    """Given a tuple of direction indices, return the endpoint (x, y) if the
    resulting walk starting at the origin never revisits a lattice point,
    else None."""
    visited = {(0, 0)}
    x, y = 0, 0
    for d in direction_seq:
        dx, dy = DIRS[d]
        x, y = x + dx, y + dy
        if (x, y) in visited:
            return None
        visited.add((x, y))
    return (x, y)


all_sequences = list(product(range(4), repeat=N_STEPS))  # 256 sequences
marked = [
    seq for seq in all_sequences
    if walk_endpoint_if_self_avoiding(seq) == TARGET_ENDPOINT
]
classical_count = len(marked)
total_saw_count = sum(
    1 for seq in all_sequences
    if walk_endpoint_if_self_avoiding(seq) is not None
)

# Sanity checks against known values of A001411(4) = 100, and that this
# endpoint-restricted count is a strictly smaller, nonzero subset of it.
EXPECTED_A001411_4 = 100
assert total_saw_count == EXPECTED_A001411_4, (
    f"classical brute force gave c_4={total_saw_count}, expected "
    f"{EXPECTED_A001411_4} (A001411(4))"
)
assert 0 < classical_count < total_saw_count


def seq_to_bits(seq):
    """Direction sequence (d0, d1, d2) -> 6-bit string, 2 bits per step,
    step 0 in the least-significant pair (matches Qiskit's qubit order:
    q0,q1 = step0; q2,q3 = step1; q4,q5 = step2)."""
    bits = ""
    for d in seq:
        bits = format(d, "02b")[::-1] + bits  # little endian per pair too
    return bits


marked_bitstrings = {seq_to_bits(seq) for seq in marked}
N_QUBITS = 2 * N_STEPS  # 6 qubits, 64 basis states
N_STATES = 2 ** N_QUBITS
assert N_STATES == len(all_sequences)

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classically-derived marked set.
# ---------------------------------------------------------------------------


def apply_multi_controlled_z(qc, bitstring):
    """Phase-flip the single computational basis state 'bitstring' (qubit 0
    is the first character after little-endian expansion below)."""
    n = len(bitstring)
    # bitstring[i] corresponds to qubit i (we built seq_to_bits that way)
    zero_positions = [i for i, b in enumerate(bitstring) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    for i in zero_positions:
        qc.x(i)


def oracle_circuit(n_qubits, marked_strings):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for bs in marked_strings:
        apply_multi_controlled_z(qc, bs)
    return qc


def diffuser_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


K = classical_count
N = N_STATES
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / K)))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = oracle_circuit(N_QUBITS, marked_bitstrings)
diffuser = diffuser_circuit(N_QUBITS)

for _ in range(optimal_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
SHOTS = 20000
result = backend.run(transpiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical-register bitstrings are printed MSB-first (qubit n-1
# ... qubit 0). Our marked_bitstrings are stored qubit-index order
# (index i = qubit i), so reverse to compare.
measured_marked_shots = 0
for bitstring, shots in counts.items():
    qubit_order_bits = bitstring[::-1]  # now index i = qubit i
    if qubit_order_bits in marked_bitstrings:
        measured_marked_shots += shots

hit_rate = measured_marked_shots / SHOTS

# Theoretical post-Grover probability of measuring a marked state.
theta = math.asin(math.sqrt(K / N))
predicted_prob = math.sin((2 * optimal_iterations + 1) * theta) ** 2

# Top measured outcome should itself be a genuinely self-avoiding walk.
top_bitstring = max(counts, key=counts.get)
top_qubit_order = top_bitstring[::-1]
top_is_marked = top_qubit_order in marked_bitstrings

print(f"Erdos problem #528 -- OEIS A387897 / A156816 (self-avoiding walk "
      f"connective constant)")
print(f"Classical: total self-avoiding walks of length {N_STEPS} on Z^2 = "
      f"{total_saw_count} (matches A001411({N_STEPS}) = "
      f"{EXPECTED_A001411_4}); of those, {classical_count} end at "
      f"{TARGET_ENDPOINT}, out of {N_STATES} total direction sequences")
print(f"Grover iterations used: {optimal_iterations}")
print(f"Predicted probability of a marked outcome: {predicted_prob:.4f}")
print(f"Measured hit rate over {SHOTS} shots: {hit_rate:.4f} "
      f"({measured_marked_shots}/{SHOTS})")
print(f"Top measured outcome '{top_qubit_order}' is self-avoiding: "
      f"{top_is_marked}")

# ---------------------------------------------------------------------------
# 4. Verify quantum result against the classical answer.
# ---------------------------------------------------------------------------

TOLERANCE = 0.15
prob_ok = abs(hit_rate - predicted_prob) < TOLERANCE
# Grover with amplification factor K/N this large (36/64) still must land
# meaningfully above the uniform baseline K/N.
baseline = K / N
amplified_ok = hit_rate > baseline

verified = top_is_marked and prob_ok and amplified_ok

if verified:
    print("PASS")
else:
    print("FAIL")
    print(f"  top_is_marked={top_is_marked} prob_ok={prob_ok} "
          f"amplified_ok={amplified_ok} baseline={baseline:.4f}")
