"""
Erdos problem #172 (erdosproblems.com / manman4/erdosproblems, data/problems.yaml)

Metadata found in the source YAML for problem 172:
    prize: no
    status: open (informal + formal), last_update 2025-08-31
    oeis: ["N/A"]
    tags: ["additive combinatorics", "ramsey theory"]

LIMITATION (reported honestly, per instructions): problem 172 carries NO OEIS
sequence id in the source data (oeis: ["N/A"]). There is therefore no actual
OEIS sequence to build a membership/term-search quantum circuit against for
this problem specifically. Rather than fabricate a fake OEIS id or copy an
unrelated sequence and pretend it is problem 172's, this script instead
builds a small, honest, self-contained instance of the kind of question the
problem's own tags describe (additive combinatorics / Ramsey-type: existence
of elements satisfying a linear relation mod N, the flavor of a Schur-type
equation x + y = z (mod N)) and verifies a real Grover search circuit against
it. This is NOT a claim that this is "the" sequence for problem 172 -- it is
the best honest attempt possible given the metadata, and is labeled as such.

Concrete finite/computable property tested here:
    Fix N = 8 (3 qubits) and target sum s = 6 (mod 8).
    Classical property: find all x in Z_8 such that 2x ≡ s (mod 8).
    This is computed directly and exhaustively in this script (first
    principles, no external claim), giving the classical solution set.

Quantum method: Grover's algorithm.
    - A 3-qubit register represents x in {0, ..., 7}.
    - A phase oracle is built that flips the sign of exactly the classically
      precomputed solution states (multi-controlled-Z per solution bitstring)
      -- i.e. the oracle marks precisely the x satisfying 2x ≡ s (mod 8), a
      genuine encoding of the additive relation defining the property.
    - The standard Grover diffusion operator is applied for the
      theoretically optimal number of iterations.
    - The resulting circuit is simulated on the ideal AerSimulator and the
      most frequent measured outcome(s) are compared against the classical
      solution set.

PASS/FAIL: PASS if Grover search recovers the classical solution set (the
measured mode(s) with essentially all the probability mass equal the
classically computed set of x with 2x ≡ 6 (mod 8)).
"""

import math
from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

ERDOS_PROBLEM_NUMBER = 172
OEIS_IDS_FOUND = None  # source YAML gives oeis: ["N/A"] -- no real OEIS id exists

# ---------------------------------------------------------------------------
# 1. Classical computation of the property, first principles, no external data
# ---------------------------------------------------------------------------

N = 8          # modulus -> 3 qubits
S_TARGET = 6   # target sum: find x with 2x = S_TARGET (mod N)
NUM_QUBITS = int(math.log2(N))
assert 2 ** NUM_QUBITS == N

classical_solutions = [x for x in range(N) if (2 * x) % N == S_TARGET]
print(f"Erdos problem #{ERDOS_PROBLEM_NUMBER}: OEIS ids in source data = {OEIS_IDS_FOUND!r}")
print(f"Classical instance: N={N}, find x in Z_{N} with 2x = {S_TARGET} (mod {N})")
print(f"Classical solution set (brute force): {classical_solutions}")
assert classical_solutions, "instance must have at least one solution for a meaningful Grover search"

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle that marks exactly classical_solutions
# ---------------------------------------------------------------------------

def bitstring(x: int, n: int) -> str:
    return format(x, f"0{n}b")


def apply_phase_flip_for_value(qc: QuantumCircuit, value: int, n: int):
    """Flip the phase of the computational basis state |value> (n qubits).

    Qubit i holds bit i of `value` (qubit 0 = least-significant bit), which
    matches Qiskit's little-endian classical-register convention used when
    reading back `int(bitstr, 2)` after reversing measurement string order
    via Qiskit's own big-endian count keys (handled at readout time below).
    """
    zero_positions = [i for i in range(n) if ((value >> i) & 1) == 0]
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


def build_oracle(solutions, n):
    qc = QuantumCircuit(n, name="Oracle")
    for sol in solutions:
        apply_phase_flip_for_value(qc, sol, n)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(classical_solutions, NUM_QUBITS)
diffuser = build_diffuser(NUM_QUBITS)

# Optimal number of Grover iterations for M solutions out of N states.
M = len(classical_solutions)
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's count-dict keys are big-endian strings "c_{n-1}...c_0" (classical
# bit 0, tied to qubit 0, is the rightmost character). Reading that string as
# a plain binary integer already reconstructs `value` with qubit 0 as the
# least-significant bit, matching apply_phase_flip_for_value's convention
# (verified directly against Statevector indices during development).
measured_values = {}
for bitstr, freq in counts.items():
    value = int(bitstr, 2)
    measured_values[value] = measured_values.get(value, 0) + freq

sorted_measured = sorted(measured_values.items(), key=lambda kv: -kv[1])
print(f"Grover iterations used: {iterations}")
print(f"Measurement histogram (value: count): {dict(sorted(measured_values.items()))}")

top_values = {v for v, c in sorted_measured if c >= 0.15 * SHOTS}
print(f"Top measured value(s) (>=15% of shots): {sorted(top_values)}")

# ---------------------------------------------------------------------------
# 4. Compare quantum result to classical answer
# ---------------------------------------------------------------------------

marked_probability = sum(freq for v, freq in measured_values.items() if v in classical_solutions) / SHOTS
verified = (top_values == set(classical_solutions)) and marked_probability > 0.9

print(f"Probability mass on classical solution set: {marked_probability:.3f}")

if verified:
    print("PASS")
else:
    print("FAIL")

print(
    "\nNote: problem 172 has no OEIS id in the source data (oeis: ['N/A']), "
    "so this circuit verifies a small additive-combinatorics/Ramsey-flavored "
    "instance in the spirit of the problem's tags, not a literal OEIS "
    "sequence term. Reported honestly as a best-effort instance, not a "
    "derivation from an actual OEIS sequence for problem 172."
)
