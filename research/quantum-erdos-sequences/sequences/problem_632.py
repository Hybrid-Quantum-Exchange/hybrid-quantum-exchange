"""
Erdos problem #632 (erdosproblems.com) -- quantum-testable lane.

Source metadata (from erdosproblems data, problems.yaml, entry "number: '632'"):
    prize: no
    status: disproved (as of 2025-08-31)
    oeis: ["N/A"]   <-- NO OEIS sequence id is associated with this problem.
    tags: ["graph theory", "chromatic number"]

LIMITATION (reported honestly, per task instructions): Problem #632 has no
OEIS id attached in the source data, so there is no "sequence" to test
membership/terms of in the sense the other lanes in this library use. There
is therefore no literal OEIS value to derive or check here. Fabricating an
OEIS-backed property would misrepresent the problem, so instead this script
targets a small, finite, genuinely computable property drawn directly from
the problem's own tags ("graph theory", "chromatic number"): PROPER
2-COLORABILITY of a small graph, i.e. deciding whether a graph is bipartite
by searching for a proper vertex coloring with 2 colors. This is exactly the
kind of decision problem "chromatic number" work is built from, and it is a
textbook fit for Grover's algorithm (unstructured search over an oracle that
marks valid colorings).

Concrete finite instance:
    Graph G = path P3: vertices {0, 1, 2}, edges {(0,1), (1,2)}.
    Encode a candidate coloring as 3 bits (q0 q1 q2), one qubit per vertex,
    bit value = color in {0, 1}. Search space size N = 2^3 = 8.
    A coloring is VALID iff every edge has differently-colored endpoints:
        q0 != q1  AND  q1 != q2

Classical answer (computed in this script by brute force over all 8
assignments before any quantum code runs): the set of valid colorings, and
its size M.

Quantum approach: Grover's algorithm. A phase oracle marks the valid
colorings using an XOR-into-ancilla / Toffoli-AND construction (uncomputed
afterwards so only the 3 "color" qubits stay entangled with the marked
phase), followed by the standard Grover diffusion operator on the 3 color
qubits, iterated the optimal number of times for N=8, M=|valid|. The circuit
is run on the ideal AerSimulator (statevector-based sampling), and the
script checks that the measured distribution is concentrated (all
high-probability outcomes) on exactly the classically-computed valid-coloring
set.

PASS/FAIL: PASS iff the set of outcomes that Grover amplified (probability
above a generous threshold) equals exactly the classically brute-forced set
of valid 2-colorings of P3.
"""

import itertools
import math

from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles by brute force.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2)]  # path graph P3 on vertices 0,1,2
N_VERTICES = 3


def is_valid_coloring(bits):
    """bits: tuple of 0/1, bits[i] = color of vertex i. True iff proper 2-coloring."""
    for u, v in EDGES:
        if bits[u] == bits[v]:
            return False
    return True


classical_valid = set()
for assignment in itertools.product([0, 1], repeat=N_VERTICES):
    if is_valid_coloring(assignment):
        classical_valid.add(assignment)

# Bit string convention used below: Qiskit reports classical bits with qubit 0
# as the rightmost character. We build the expected bitstring set in that
# same convention so comparison is direct.
def bits_to_qiskit_string(bits):
    # bits[i] is qubit i; qiskit string has qubit (n-1) leftmost, qubit 0 rightmost.
    return "".join(str(bits[i]) for i in reversed(range(len(bits))))


expected_bitstrings = {bits_to_qiskit_string(b) for b in classical_valid}

N = 2 ** N_VERTICES
M = len(expected_bitstrings)

print("Classical brute-force result:")
print(f"  Valid 2-colorings of P3 (edges {EDGES}): {sorted(classical_valid)}")
print(f"  Expected measurement bitstrings: {sorted(expected_bitstrings)}")
print(f"  N = {N} candidates, M = {M} valid colorings")

if M == 0 or M == N:
    raise RuntimeError("Degenerate instance (M=0 or M=N); Grover not meaningful here.")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle for "q0 != q1 AND q1 != q2".
# ---------------------------------------------------------------------------
#
# Registers:
#   color   : 3 qubits, the candidate coloring (q0, q1, q2)
#   anc     : 2 ancilla qubits, one per edge, used to hold XOR(u,v) for that edge
#   out     : 1 ancilla qubit, prepared in |-> to realize a phase oracle via
#             a standard Toffoli-AND of the two edge-XOR ancillas.
#
# XOR(u,v) into an ancilla initialized to |0>: CNOT(u -> anc); CNOT(v -> anc).
# anc ends up = 1 exactly when u,v differ (edge satisfied).
# Then out ^= anc0 AND anc1 (a Toffoli), which with out in |-> gives a phase
# flip exactly on states where both edges are satisfied. Ancillas are then
# uncomputed (reverse XORs) so they return to |0> and stay unentangled.

color = QuantumRegister(N_VERTICES, name="c")
anc = AncillaRegister(len(EDGES), name="a")
out = AncillaRegister(1, name="o")
creg = ClassicalRegister(N_VERTICES, name="m")


def build_oracle():
    qc = QuantumCircuit(color, anc, out, creg)
    # compute edge-satisfaction bits into ancillas
    for i, (u, v) in enumerate(EDGES):
        qc.cx(color[u], anc[i])
        qc.cx(color[v], anc[i])
    # AND the (exactly 2) edge-satisfaction ancillas into the phase via Toffoli
    # on the |-> output qubit (caller prepares out in |->).
    qc.ccx(anc[0], anc[1], out[0])
    # uncompute
    for i, (u, v) in enumerate(EDGES):
        qc.cx(color[v], anc[i])
        qc.cx(color[u], anc[i])
    return qc


def build_diffuser():
    qc = QuantumCircuit(color, anc, out, creg)
    qc.h(color)
    qc.x(color)
    # multi-controlled Z on the color register (phase flip on |000>)
    qc.h(color[-1])
    qc.mcx(list(color[:-1]), color[-1])
    qc.h(color[-1])
    qc.x(color)
    qc.h(color)
    return qc


oracle = build_oracle()
diffuser = build_diffuser()

# optimal number of Grover iterations for N candidates, M marked items
iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations used: {iterations}")

grover = QuantumCircuit(color, anc, out, creg)
grover.h(color)
# prepare output ancilla in |->
grover.x(out[0])
grover.h(out[0])

for _ in range(iterations):
    grover.compose(oracle, inplace=True)
    grover.compose(diffuser, inplace=True)

# restore output ancilla (not strictly necessary before measurement of color reg)
grover.h(out[0])
grover.x(out[0])

grover.measure(color, creg)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
job = sim.run(grover, shots=shots)
result = job.result()
counts = result.get_counts()

print("Measurement counts:", counts)

# Outcomes considered "amplified" by Grover: probability well above the
# uniform baseline 1/N (generous threshold at 3x uniform).
uniform_prob = 1.0 / N
threshold = 3 * uniform_prob
amplified = {
    bitstring
    for bitstring, count in counts.items()
    if (count / shots) > threshold
}

print(f"Uniform baseline probability: {uniform_prob:.4f}, threshold: {threshold:.4f}")
print(f"Amplified bitstrings (quantum result): {sorted(amplified)}")
print(f"Expected bitstrings (classical result): {sorted(expected_bitstrings)}")

verified = amplified == expected_bitstrings

if verified:
    print("PASS")
else:
    print("FAIL")
