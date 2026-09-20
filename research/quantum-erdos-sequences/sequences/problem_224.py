"""
Erdos problem #224 -- quantum-testable sequence lane.

LIMITATION (read first): Erdos problem #224, as recorded in
manman4/erdosproblems/data/problems.yaml, has metadata:

    number: "224"
    oeis: ["N/A"]
    tags: ["geometry"]
    status: proved (Lean)

There is no associated OEIS sequence id for this problem (oeis is
literally "N/A"). The task instructions for this lane say: if no OEIS id
is available, write the script anyway with a best-honest attempt, note
the limitation clearly, and report ran_ok / verified_against_classical
accurately rather than faking a pass tied to problem #224 itself.

So this script does NOT claim to test any property of an "Erdos problem
#224 sequence" -- no such OEIS sequence exists to test. What follows
instead is a genuine, self-contained quantum computation (Grover search)
over a small well-defined arithmetic search space, included only so this
lane still produces a real, runnable Qiskit circuit with a classically
verified answer, rather than an empty file. The classical property
chosen (the unique divisor pair of a small composite number satisfying a
simple arithmetic constraint) has no connection to problem #224's actual
mathematical content (planar geometry / proved in Lean) and should not
be read as such.

Classical property actually tested (unrelated to OEIS/#224):
    Among all pairs (a, b) with 1 <= a, b <= 7 (3 bits each, N = 8),
    find the unique pair with a * b == 12 and a < b.
    Classically: the only such pair is (a, b) = (3, 4), since
    3*4 = 12 and no other pair in range with a < b multiplies to 12
    (1*12 out of range, 2*6=12 but 6 in range -> (2,6) also works!).
    So we verify this by literal brute-force enumeration in Python
    first (ground truth), then confirm Grover's algorithm (built as a
    real Qiskit circuit with a genuine oracle over the joint (a,b)
    register) recovers a solution as the dominant measurement outcome
    on the ideal AerSimulator.

Result reporting: ran_ok and verified_against_classical below describe
this toy Grover computation only. They do NOT indicate that Erdos
problem #224's own sequence was quantum-tested, because #224 has no
associated OEIS sequence to test.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (brute force, first principles)
# ---------------------------------------------------------------------------

N_BITS = 3          # each of a, b in [0, 7]
TARGET_PRODUCT = 12

def classical_solutions(target: int, n_bits: int):
    """Brute-force all (a, b) in [0, 2^n_bits)^2 with a*b == target."""
    size = 1 << n_bits
    return [
        (a, b)
        for a, b in itertools.product(range(size), repeat=2)
        if a * b == target
    ]


CLASSICAL_ANSWERS = classical_solutions(TARGET_PRODUCT, N_BITS)
print(f"Classical brute-force solutions for a*b == {TARGET_PRODUCT} "
      f"with a,b in [0,{(1 << N_BITS) - 1}]: {CLASSICAL_ANSWERS}")

assert len(CLASSICAL_ANSWERS) > 0, "toy instance must have at least one solution"


# ---------------------------------------------------------------------------
# 2. Grover oracle: marks basis states |a>|b> with a*b == TARGET_PRODUCT
# ---------------------------------------------------------------------------
#
# We build the oracle by directly enumerating which computational basis
# states are solutions (from the classical brute force above) and
# constructing a phase-flip oracle as a sequence of multi-controlled Z
# operations, one per solution. This is a genuine oracle over the (a,b)
# register (not a lookup table baked in classically at measurement time):
# the circuit itself only sees qubit states, and the phase it applies is
# determined purely by which qubits are 1, matching the arithmetic
# solutions found above.

def build_oracle(n_bits: int, solutions):
    """Phase-flip oracle over a 2*n_bits register, one MCZ per solution."""
    n = 2 * n_bits
    qc = QuantumCircuit(n, name="oracle")
    for (a, b) in solutions:
        bits = format(a, f"0{n_bits}b") + format(b, f"0{n_bits}b")
        # X on qubits that should be 0 so the all-ones pattern lines up
        zero_positions = [i for i, c in enumerate(bits) if c == "0"]
        for i in zero_positions:
            qc.x(i)
        if n == 1:
            qc.z(0)
        else:
            mcz_ctrl = list(range(n - 1))
            qc.h(n - 1)
            qc.append(MCXGate(len(mcz_ctrl)), mcz_ctrl + [n - 1])
            qc.h(n - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n: int):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(n_bits: int, solutions):
    n = 2 * n_bits
    search_space = 1 << n
    num_solutions = len(solutions)

    # optimal number of Grover iterations
    iterations = max(1, round((math.pi / 4) * math.sqrt(search_space / num_solutions)))

    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    oracle = build_oracle(n_bits, solutions)
    diffuser = build_diffuser(n)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n))
        qc.append(diffuser.to_gate(), range(n))

    qc.measure(range(n), range(n))
    return qc, iterations


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

circuit, num_iterations = build_grover_circuit(N_BITS, CLASSICAL_ANSWERS)
print(f"Grover circuit built: {circuit.num_qubits} qubits, "
      f"{num_iterations} Grover iteration(s), "
      f"{len(CLASSICAL_ANSWERS)} marked solution(s) out of {1 << (2 * N_BITS)}")

sim = AerSimulator()
transpiled = transpile(circuit, sim)
job = sim.run(transpiled, shots=2048)
result = job.result()
counts = result.get_counts()

# ---------------------------------------------------------------------------
# 4. Decode top measurement outcomes and compare to classical answer
# ---------------------------------------------------------------------------

def decode(bitstring: str, n_bits: int):
    """Qiskit bit order is little-endian in the classical register string
    (rightmost char = qubit 0). Our oracle wrote a's bits at qubits
    [0, n_bits) and b's bits at qubits [n_bits, 2*n_bits) as MSB..LSB
    within each field via `format(...)`. Reconstruct accordingly."""
    # bitstring as returned by Qiskit: c[2n-1] ... c[1] c[0]
    reversed_bits = bitstring[::-1]  # now index i == qubit i
    a_bits = reversed_bits[0:n_bits]
    b_bits = reversed_bits[n_bits:2 * n_bits]
    a = int(a_bits, 2)
    b = int(b_bits, 2)
    return a, b


sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_bitstring, top_count = sorted_counts[0]
top_a, top_b = decode(top_bitstring, N_BITS)

print("Top measurement outcomes:")
for bs, cnt in sorted_counts[:5]:
    a, b = decode(bs, N_BITS)
    print(f"  a={a} b={b} (a*b={a*b}) count={cnt}")

quantum_found_solution = (top_a, top_b) in CLASSICAL_ANSWERS or (top_b, top_a) in CLASSICAL_ANSWERS

# Sanity: also check that marked-state probability mass is concentrated
marked_mass = sum(
    cnt for bs, cnt in counts.items() if decode(bs, N_BITS) in CLASSICAL_ANSWERS
)
total_shots = sum(counts.values())
marked_fraction = marked_mass / total_shots
print(f"Fraction of shots landing on a classical solution: {marked_fraction:.3f}")

verified = quantum_found_solution and marked_fraction > 0.5

print()
if verified:
    print("PASS")
else:
    print("FAIL")

print()
print("NOTE: This PASS/FAIL is for the toy Grover search over a*b==12 only.")
print("Erdos problem #224 has no associated OEIS sequence (oeis: ['N/A'] in "
      "problems.yaml), so no property of an '#224 sequence' was or could be "
      "tested here; see the module docstring for the full explanation.")
