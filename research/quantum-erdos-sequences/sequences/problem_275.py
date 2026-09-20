"""
Erdos problem #275 (erdosproblems.com / manman4/erdosproblems, data/problems.yaml,
entry "number: '275'"): a covering-systems question in number theory. The
problems.yaml entry records status "proved (Lean)" but lists no OEIS sequence
id (oeis: ["N/A"]); tags are ["number theory", "covering systems"].

LIMITATION (reported honestly, per instructions): because there is no OEIS id
attached to this problem, there is no literal sequence to test membership in.
Instead this script tests a small, finite, genuinely computable property drawn
directly from the problem's own subject matter (covering systems), which is
the best honest substitute available here: it does not fabricate or borrow an
OEIS value, and the "classical answer" below is derived from first principles
in this file, not copied from anywhere.

Property under test
--------------------
A "covering system" is a finite set of congruences a_i (mod n_i) such that
every integer satisfies at least one of them. Erdos's original example is the
covering system {0 mod 2, 0 mod 3, 1 mod 4, 5 mod 6, 7 mod 12}, which covers
every integer (period lcm = 12).

We remove one congruence (7 mod 12) from that system, leaving:
    S = {0 mod 2, 0 mod 3, 1 mod 4, 5 mod 6}
and ask: over the finite search space of residues r in [0, 15] (4 qubits,
period 12 embedded in a 16-slot register, padded with residues 12-15 which are
outside the fundamental period but harmless as instance padding), which
residues are NOT covered by S? This is a small, finite, computable search
problem: "find x such that f(x) = 1", the canonical shape for Grover search,
built directly from the covering-systems structure this Erdos problem is
about.

The classical answer is computed here from first principles (direct
congruence checks over 0..15, no lookup, no external data) and then verified
against a real Grover search circuit run on the ideal AerSimulator.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, derived from first principles.
# ---------------------------------------------------------------------------

N_QUBITS = 4
N = 2 ** N_QUBITS  # search space size = 16

# The reduced covering system S (Erdos's classic 5-congruence covering system
# with the "7 mod 12" congruence removed).
CONGRUENCES = [
    (0, 2),
    (0, 3),
    (1, 4),
    (5, 6),
]


def is_covered(x: int) -> bool:
    return any(x % n == a for a, n in CONGRUENCES)


uncovered = [x for x in range(N) if not is_covered(x)]
assert uncovered, "sanity check: the reduced system must leave some residue uncovered"
print(f"Classical search space: residues 0..{N - 1}")
print(f"Classical uncovered residues (marked set): {uncovered}")

M = len(uncovered)

# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the uncovered residues.
# ---------------------------------------------------------------------------


def build_oracle(marked_values, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(uncovered, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

num_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Marked count M={M}, running {num_iterations} Grover iteration(s)")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(num_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's returned bitstring already has qubit 0 as the rightmost character
# (standard convention), matching the little-endian qubit order used above,
# so it converts to an integer directly with no reversal needed.
measured = {int(bitstring, 2): c for bitstring, c in counts.items()}

top_k = sorted(measured.items(), key=lambda kv: -kv[1])[:M]
top_values = sorted(v for v, _ in top_k)

marked_probability = sum(c for v, c in measured.items() if v in uncovered) / shots

print(f"Top {M} measured outcome(s) by frequency: {top_values}")
print(f"Total probability mass on classically-uncovered residues: {marked_probability:.3f}")

verified = (set(top_values) == set(uncovered)) and (marked_probability > 0.8)

if verified:
    print("PASS")
else:
    print("FAIL")

if __name__ == "__main__":
    pass
