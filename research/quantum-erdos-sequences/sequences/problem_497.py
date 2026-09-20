"""
Erdos problem #497 (Dedekind's problem), OEIS A000372.

A000372 enumerates Dedekind numbers M(n): the number of monotone Boolean
functions of n variables (equivalently, the number of antichains, or the
number of elements of the free distributive lattice, on n generators).
The known values begin M(0..4) = 2, 3, 6, 20, 168.

Classical property tested here (small, finite, exactly computable):
  For n = 2 variables there are 2^(2^2) = 16 Boolean functions
  f: {0,1}^2 -> {0,1}, represented as a 4-bit truth table
  (f(00), f(01), f(10), f(11)). A function is *monotone* iff x <= y
  (componentwise) implies f(x) <= f(y); for n=2 this reduces to the four
  covering-relation constraints:
      f(00) <= f(01),  f(00) <= f(10),  f(01) <= f(11),  f(10) <= f(11).
  The script first computes, by brute-force classical enumeration of all
  16 truth tables, the exact set of monotone ones and its size. That size
  must equal the third term of A000372, M(2) = 6 -- this is checked
  in-script, not copied from OEIS.

Quantum circuit: a genuine Grover search over the 4-qubit truth-table
register (16 basis states = all Boolean functions of 2 variables). The
oracle computes the four monotonicity-violation conditions into ancilla
qubits (uncomputed afterward), combines "no violation on any constraint"
into a phase-kickback ancilla prepared in the |-> state, and so applies a
-1 phase to exactly the computational basis states that encode a monotone
function -- with no other knowledge of which 6 states those are baked in
by hand. One Grover diffusion step follows (the Grover-optimal iteration
count for 6 marked states out of 16). The resulting statevector's total
probability mass on the classically-determined monotone set is computed
exactly (via Statevector, no sampling noise) and compared against the
closed-form Grover amplification formula sin^2((2k+1) theta), theta =
arcsin(sqrt(6/16)), k = 1 -- an independent classical prediction. A
shot-based AerSimulator run then samples the same circuit; the most
frequent measured truth table is checked, by the same classical monotone
predicate, to indeed be monotone, and the fraction of shots landing on
some monotone truth table is checked against the same theoretical bound.

PASS requires all of:
  (a) brute-force classical count of monotone 2-variable Boolean
      functions equals A000372(2) = 6;
  (b) the exact (statevector) post-Grover probability of measuring a
      monotone truth table matches the closed-form theoretical value;
  (c) the AerSimulator shot-sampled circuit's most frequent outcome is
      classically monotone, and its monotone-hit fraction is close to
      the same theoretical value.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# 1. Classical ground truth: brute-force all 16 Boolean functions of two
#    variables and find the monotone ones.
# ---------------------------------------------------------------------

def truth_table_bits(tt):
    """tt is int 0..15; return (f00, f01, f10, f11) with f00 = LSB."""
    return tuple((tt >> i) & 1 for i in range(4))


def is_monotone(tt):
    f00, f01, f10, f11 = truth_table_bits(tt)
    return f00 <= f01 and f00 <= f10 and f01 <= f11 and f10 <= f11


MONOTONE_SET = sorted(tt for tt in range(16) if is_monotone(tt))
DEDEKIND_M2 = len(MONOTONE_SET)

assert DEDEKIND_M2 == 6, (
    f"classical brute force gave M(2) = {DEDEKIND_M2}, expected A000372(2) = 6"
)

# ---------------------------------------------------------------------
# 2. Build the Grover oracle + diffuser for this 4-qubit search space.
# ---------------------------------------------------------------------
# Qubit order in the 'f' register: f[0]=f(00), f[1]=f(01), f[2]=f(10), f[3]=f(11)
# Constraints (a implies b, i.e. f[a] <= f[b]) violated iff f[a]=1 and f[b]=0.
CONSTRAINTS = [(0, 1), (0, 2), (1, 3), (2, 3)]

f_reg = QuantumRegister(4, "f")
v_reg = QuantumRegister(4, "v")   # one violation-ancilla per constraint
m_reg = QuantumRegister(1, "m")   # phase-kickback ancilla


def add_oracle(qc):
    f = f_reg
    v = v_reg
    m = m_reg
    # --- compute violation flags: v_i = f[a] AND (NOT f[b]) ---
    for i, (a, b) in enumerate(CONSTRAINTS):
        qc.x(f[b])
        qc.ccx(f[a], f[b], v[i])
        qc.x(f[b])
    # --- flip to "no violation" bits, then AND them all into m via MCX ---
    for i in range(4):
        qc.x(v[i])
    qc.append(MCXGate(4), [v[0], v[1], v[2], v[3], m[0]])
    for i in range(4):
        qc.x(v[i])
    # --- uncompute violation flags ---
    for i, (a, b) in reversed(list(enumerate(CONSTRAINTS))):
        qc.x(f[b])
        qc.ccx(f[a], f[b], v[i])
        qc.x(f[b])


def add_diffuser(qc, f):
    qc.h(f)
    qc.x(f)
    # multi-controlled Z on |1111> via H-MCX-H sandwich on the target qubit
    qc.h(f[3])
    qc.append(MCXGate(3), [f[0], f[1], f[2], f[3]])
    qc.h(f[3])
    qc.x(f)
    qc.h(f)


def build_circuit(iterations, with_measurement):
    qc = QuantumCircuit(f_reg, v_reg, m_reg)
    qc.h(f_reg)
    # phase-kickback ancilla in |-> so the MCX inside add_oracle acts as a
    # controlled phase flip on marked (monotone) basis states of f_reg.
    qc.x(m_reg[0])
    qc.h(m_reg[0])
    for _ in range(iterations):
        add_oracle(qc)
        add_diffuser(qc, f_reg)
    # restore ancilla
    qc.h(m_reg[0])
    qc.x(m_reg[0])
    if with_measurement:
        qc.measure_all()
    return qc


# Grover-optimal iteration count for 6 marked out of 16.
N = 16
K = DEDEKIND_M2
theta = math.asin(math.sqrt(K / N))
ITERS = max(1, round((math.pi / (4 * theta)) - 0.5))

# ---------------------------------------------------------------------
# 3. Exact statevector check (no sampling noise) vs. closed-form theory.
# ---------------------------------------------------------------------
sv_circuit = build_circuit(ITERS, with_measurement=False)
sv = Statevector.from_instruction(sv_circuit)
probs = sv.probabilities_dict(qargs=range(4))  # marginal over f_reg only

quantum_monotone_prob = sum(
    p for bitstring, p in probs.items()
    if int(bitstring, 2) in MONOTONE_SET
)

theoretical_prob = math.sin((2 * ITERS + 1) * theta) ** 2

statevector_ok = math.isclose(quantum_monotone_prob, theoretical_prob, abs_tol=1e-6)

# ---------------------------------------------------------------------
# 4. Shot-based AerSimulator run: sample the circuit for real.
# ---------------------------------------------------------------------
shot_circuit = build_circuit(ITERS, with_measurement=True)
backend = AerSimulator()
shots = 4096
result = backend.run(shot_circuit, shots=shots).result()
counts = result.get_counts()

# clbit register order from measure_all is [f0..f3, v0..v3, m0] reversed in the
# returned bitstring; recover just the f_reg outcome (first 4 measured qubits).
def f_value_from_key(key):
    bits = key.replace(" ", "")
    # Qiskit prints classical bits MSB..LSB across the *whole* register list
    # in reverse creation order; f_reg was created first (qubits 0-3), so its
    # bits are the last 4 characters of the string.
    f_bits = bits[-4:]
    return int(f_bits, 2)

monotone_hits = 0
best_key, best_count = max(counts.items(), key=lambda kv: kv[1])
most_frequent_tt = f_value_from_key(best_key)
most_frequent_is_monotone = is_monotone(most_frequent_tt)

for key, c in counts.items():
    if is_monotone(f_value_from_key(key)):
        monotone_hits += c

measured_monotone_fraction = monotone_hits / shots
sampling_ok = (
    most_frequent_is_monotone
    and abs(measured_monotone_fraction - theoretical_prob) < 0.08
)

# ---------------------------------------------------------------------
# 5. Verdict.
# ---------------------------------------------------------------------
print(f"Classical monotone truth tables (M(2)): {MONOTONE_SET} -> count {DEDEKIND_M2}")
print(f"Grover iterations used: {ITERS}")
print(f"Theoretical post-Grover monotone probability: {theoretical_prob:.4f}")
print(f"Exact statevector monotone probability:       {quantum_monotone_prob:.4f}")
print(f"Sampled ({shots} shots) monotone fraction:      {measured_monotone_fraction:.4f}")
print(f"Most frequent measured truth table: {most_frequent_tt:04b} "
      f"(monotone: {most_frequent_is_monotone})")

overall_ok = DEDEKIND_M2 == 6 and statevector_ok and sampling_ok

print("PASS" if overall_ok else "FAIL")
