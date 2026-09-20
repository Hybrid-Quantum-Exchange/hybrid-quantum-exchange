"""
Erdos problem #829 -- quantum-testable sequence entry.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: 829"):
    oeis: ["A025455", "A025468", "possible"]
    tags: ["number theory"]

OEIS A025455: number of ways to write n as an ordered-by-size sum of two
positive cubes, i.e. the number of pairs (x, y) with x >= y > 0 and
x^3 + y^3 = n. (A025468 is the closely related "number of partitions of n
into 2 cubes" variant; both live in the same sum-of-two-cubes family that
this script tests.)

Classical property tested here
-------------------------------
Fix N = 1729 (the Hardy-Ramanujan taxicab number, famous precisely because
A025455(1729) = 2: 1^3 + 12^3 = 9^3 + 10^3 = 1729) and a small finite search
space x, y in {1, ..., 15} (representable with 4 bits each, 8 qubits total,
256 basis states).

The script first computes, by brute-force classical search over all 256
ordered pairs (x, y) in that space, the exact set of solutions to
    x^3 + y^3 = 1729.
This recovers all four *ordered* pairs -- (1,12), (12,1), (9,10), (10,9) --
which correspond exactly to the two *unordered* representations counted by
A025455(1729) = 2. This is derived from first principles in this script
(brute-force cube enumeration), not copied from OEIS.

Quantum circuit
----------------
A genuine Grover search circuit over the 8-qubit space {0,...,255} (x in the
high 4 bits, y in the low 4 bits) is built:
  - The oracle phase-flips exactly the basis states corresponding to the
    classically-precomputed solution set (standard technique for small,
    explicit marked-element sets: X-gates to map each solution's 0-bits to
    1, a multi-controlled Z, then undo the X-gates).
  - The diffuser is the standard Grover diffusion operator over all 8 qubits.
  - The number of Grover iterations is chosen optimally from the known
    count of marked states (4 marked out of 256).

The circuit is run on the ideal AerSimulator (statevector-based, exact
simulation, no noise). PASS/FAIL is decided by checking that the four
highest-probability measured bitstrings are exactly the four classically
computed solution pairs, and that a majority of measurement shots land on
solution states -- i.e. the quantum search amplifies exactly the classical
answer.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS values copied).
# ---------------------------------------------------------------------------

N = 1729
BITS = 4                       # x, y each range over 1..15 -> 4 bits
DOMAIN = list(range(1, 2 ** BITS))  # 1..15

def classical_solutions(n, domain):
    """All ordered pairs (x, y) in domain x domain with x**3 + y**3 == n."""
    sols = []
    for x, y in itertools.product(domain, repeat=2):
        if x ** 3 + y ** 3 == n:
            sols.append((x, y))
    return sols

solutions = classical_solutions(N, DOMAIN)

# Unordered representations (the quantity A025455 actually counts: x >= y).
unordered = {tuple(sorted(p, reverse=True)) for p in solutions}

print(f"Classical brute force over x,y in 1..{2**BITS - 1}, N = {N}:")
print(f"  ordered solution pairs (x,y): {solutions}")
print(f"  unordered representations x^3+y^3=N (x>=y): {sorted(unordered)}")
print(f"  count of unordered representations = {len(unordered)} "
      f"(expected classical value of A025455({N}) restricted to this "
      f"domain, computed here from first principles, not looked up)")

assert len(solutions) == 4, "expected exactly 4 ordered pairs for N=1729 in 1..15"
assert len(unordered) == 2, "expected exactly 2 unordered representations"

# Bit-encoding helper: state index = x_bits(4) << 4 | y_bits(4), x,y in 0..15.
def encode(x, y):
    return (x << BITS) | y

marked_states = sorted(encode(x, y) for (x, y) in solutions)
print(f"  marked computational basis states (x<<4|y): {marked_states}")

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle for these marked states.
# ---------------------------------------------------------------------------

NUM_QUBITS = 2 * BITS  # 8 qubits: qubits 0-3 = y, qubits 4-7 = x (Qiskit LSB-first)

def add_oracle(qc, marked, num_qubits):
    """Phase-flip each marked basis state (multi-controlled Z per state)."""
    for state in marked:
        bits = format(state, f"0{num_qubits}b")[::-1]  # bit i -> qubit i (LSB first)
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        # Multi-controlled Z on all qubits: controls = all but last, target = last.
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        if zero_qubits:
            qc.x(zero_qubits)

def add_diffuser(qc, num_qubits):
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))

# ---------------------------------------------------------------------------
# 3. Assemble the full Grover circuit with the optimal iteration count.
# ---------------------------------------------------------------------------

M = len(marked_states)          # number of marked states = 4
SPACE = 2 ** NUM_QUBITS          # 256
optimal_iters = max(1, round((math.pi / 4) * math.sqrt(SPACE / M)))
print(f"  search space size = {SPACE}, marked = {M}, "
      f"Grover iterations = {optimal_iters}")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(optimal_iters):
    add_oracle(qc, marked_states, NUM_QUBITS)
    add_diffuser(qc, NUM_QUBITS)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
SHOTS = 4096
result = sim.run(qc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit counts keys are big-endian bitstrings over the classical register,
# matching qubit order MSB(qubit num_qubits-1) ... LSB(qubit 0).
def bitstring_to_state(bstr):
    return int(bstr, 2)

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_states = [bitstring_to_state(b) for b, _ in sorted_counts[:M]]

marked_shots = sum(c for b, c in counts.items() if bitstring_to_state(b) in marked_states)
frac_marked = marked_shots / SHOTS

print(f"  top {M} measured states (by shot count): {sorted(top_states)}")
print(f"  fraction of shots landing on a marked (solution) state: {frac_marked:.3f}")

quantum_top_set = set(top_states)
classical_set = set(marked_states)

verified = (quantum_top_set == classical_set) and (frac_marked > 0.5)

print()
if verified:
    print("PASS")
else:
    print("FAIL")
