"""
Erdos problem #158 -- quantum-testable instance.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: 158"):
    tags: ["sidon sets"]
    oeis: ["N/A"]   -- no OEIS sequence id is listed for this problem.

Because no OEIS id exists for problem #158, this script does not test
membership in an OEIS sequence. Instead it builds a genuine, finite,
computable property drawn directly from the problem's own subject
(Sidon sets: a set S of non-negative integers is a Sidon set iff all
pairwise sums a+b with a <= b, a,b in S, are distinct) and verifies a
Grover search over a small universe against the classically-computed
answer. This is disclosed as a limitation: the instance is inspired by
the problem's tag, not a direct test of an Erdos-158 OEIS term (there
is none to test).

Classical property tested
--------------------------
Fix the base Sidon set {1, 2, 5} (its own pairwise sums 2, 3, 6, 4, 7,
10 are already all distinct). For each candidate x in
U = {0, 1, ..., 15} with x not in {1, 2, 5}, form S = {1, 2, 5, x} and
ask: is S still a Sidon set? S is a Sidon set iff all ten pairwise
sums a+b (a <= b, a,b in S) are distinct.

The classical answer (computed in this script, from first principles,
by brute force over all 16 candidates) is the exact set of "marked"
x values for which {1, 2, 5, x} is a Sidon set. This is exactly the
kind of small, structured search problem Grover's algorithm is built
for: a black-box predicate over a bounded search space (here 4 qubits,
|U| = 16, with 7 of the 16 candidates marked), where Grover amplifies
the marked basis states above the uniform baseline.

Quantum circuit
----------------
4 index qubits encode x in {0,...,15} (little-endian binary).
The oracle is a diagonal phase oracle built directly from the
classically-computed marked set (multi-controlled Z gates on the
basis states corresponding to each marked x) -- a standard way to
realize a black-box Grover oracle for a small, explicitly known
predicate. The diffuser is the standard Grover diffusion operator.
The optimal number of Grover iterations for this instance is computed
from the true number of marked states.

We run the circuit on the ideal AerSimulator, take the most-sampled
basis states, and PASS iff the set of measured x values with
probability mass above a fixed threshold matches exactly the
classically-computed marked set.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator

N_QUBITS = 4
UNIVERSE = list(range(2 ** N_QUBITS))  # {0, ..., 15}
BASE = (1, 2, 5)


def is_sidon(s):
    """Classical check: all pairwise sums a+b (a<=b) of set s are distinct."""
    elems = sorted(s)
    sums = []
    for i in range(len(elems)):
        for j in range(i, len(elems)):
            sums.append(elems[i] + elems[j])
    return len(sums) == len(set(sums))


def classical_marked_set():
    """Brute-force, from first principles, every x in UNIVERSE such that
    BASE + {x} (x distinct from all elements of BASE) is a Sidon set."""
    marked = []
    for x in UNIVERSE:
        if x in BASE:
            continue
        candidate = set(BASE) | {x}
        if is_sidon(candidate):
            marked.append(x)
    return sorted(marked)


def build_oracle(marked, n_qubits):
    """Diagonal phase oracle flipping the sign of each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_circuit(marked, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    marked = classical_marked_set()
    m = len(marked)
    n = N_QUBITS
    total = 2 ** n

    print("Erdos problem #158 (tags: sidon sets; oeis: N/A)")
    print(f"Universe U = {UNIVERSE}")
    print(f"Base Sidon set = {BASE}")
    print(f"Classically computed marked x (BASE + {{x}} is Sidon): {marked}")

    if m == 0 or m == total:
        print("Degenerate marked set (0 or all states) -- Grover not meaningful.")
        print("FAIL")
        return False, marked, marked

    # Optimal number of Grover iterations
    theta = math.asin(math.sqrt(m / total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = grover_circuit(marked, n, iterations)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bitstrings are ordered c[n-1]...c[0] (MSB first), and clbit i
    # was measured from qubit i, which is exactly our little-endian
    # encoding of x (qubit 0 = least-significant bit). So the bitstring,
    # read directly as a binary numeral, is already x -- no reversal.
    dist = {}
    for bitstring, cnt in counts.items():
        x = int(bitstring, 2)
        dist[x] = dist.get(x, 0) + cnt

    print("Measured distribution (x: probability):")
    for x in sorted(dist, key=lambda k: -dist[k]):
        print(f"  x={x}: {dist[x] / shots:.4f}")

    # Theoretical per-state probabilities after `iterations` Grover steps
    # (exact quantum-mechanical prediction, not a guess): the marked-state
    # amplitude is sin((2*iterations+1)*theta), so each marked basis state
    # carries sin^2(...)/m of the total probability and each unmarked
    # state carries the remaining (1 - sin^2(...))/(total - m). The
    # decision threshold is the geometric mean of those two per-state
    # probabilities, which correctly separates marked from unmarked
    # whenever Grover amplification worked.
    p_marked_total = math.sin((2 * iterations + 1) * theta) ** 2
    p_marked_each = p_marked_total / m
    p_unmarked_each = (1 - p_marked_total) / (total - m)
    threshold = math.sqrt(p_marked_each * p_unmarked_each)
    print(f"Theoretical per-state prob: marked={p_marked_each:.4f}, "
          f"unmarked={p_unmarked_each:.4f}, threshold={threshold:.4f}")

    found = sorted(x for x, c in dist.items() if c / shots > threshold)

    verified = found == marked
    print(f"Grover-found marked set: {found}")
    print(f"Classical marked set:    {marked}")

    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified, found, marked


if __name__ == "__main__":
    ok, found, marked = main()
    if not ok:
        raise SystemExit(1)
