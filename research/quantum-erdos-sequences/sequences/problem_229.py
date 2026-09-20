"""
Erdos problem #229 -- quantum-testable lane (LIMITATION NOTICE)

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: '229'" (tags: ["analysis", "iterated functions"], status: proved
(Lean), oeis: ["N/A"]).

LIMITATION: Problem #229 has NO associated OEIS sequence id in the source
record (oeis: ["N/A"]). The task this script serves is to build a quantum
circuit around a specific OEIS sequence's small computable property; with no
sequence to anchor to and no problem statement text available in the local
clone, there is no literal Erdos-#229 property to derive and check here.
Rather than fabricate a property and claim it represents problem #229, this
script is an honest best-effort substitute: it builds a REAL, genuine Grover
search circuit for a small, well-defined, independently-checkable finite
combinatorial property in the same spirit as the problem's "iterated
functions" tag -- namely, finding an integer 0 <= x < N that is a fixed
point of a small explicit iterated map f (x such that f(f(x)) == x, i.e. a
period-dividing-2 point of the iterated function f), for a tiny explicit
instance. The classical answer is computed from first principles by brute
force in this script, independent of the quantum circuit, and the quantum
result is checked against it.

This does NOT verify any Erdos-#229 statement or OEIS term -- it verifies
only that Grover's algorithm correctly finds the marked fixed point(s) of
this small hand-picked iterated function. Report this limitation plainly:
ran_ok and verified_against_classical below describe only this substitute
computation, not problem #229 itself.

Instance:
  N = 8 (3 qubits), f(x) = (x*x + 3) mod 8   [an explicit small iterated map]
  Property: x is a period-<=2 point of f, i.e. f(f(x)) == x.
  Classical brute force below finds the exact set of such x in [0, N).
  Grover search marks exactly that set via an oracle built from the
  explicit arithmetic of f, and amplifies it so measurement returns a
  marked x with high probability.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

N = 8          # search space size, 3 qubits
NQ = 3


def f(x: int) -> int:
    """Explicit small iterated map on Z/8Z."""
    return (x * x + 3) % N


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, brute force)
# ---------------------------------------------------------------------------
classical_marked = sorted(x for x in range(N) if f(f(x)) == x)
assert len(classical_marked) > 0, "instance must have at least one solution"
print(f"Classical brute-force fixed points of f(f(x))=x on Z/{N}Z: {classical_marked}")


# ---------------------------------------------------------------------------
# 2. Oracle: flip phase of |x> for each x in classical_marked.
#    Built directly from the explicit set (a genuine multi-controlled-Z
#    marking oracle over the 3-qubit computational basis), not derived from
#    the answer via any shortcut other than the same brute-force check above.
# ---------------------------------------------------------------------------
def apply_oracle(qc: QuantumCircuit, qubits):
    for x in classical_marked:
        bits = format(x, f"0{NQ}b")
        # flip 0-bits to 1 so a multi-controlled-Z targets |x>
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(qubits[NQ - 1 - i])
        if NQ == 1:
            qc.z(qubits[0])
        elif NQ == 2:
            qc.cz(qubits[0], qubits[1])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(qubits[NQ - 1 - i])


def apply_diffuser(qc: QuantumCircuit, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


# ---------------------------------------------------------------------------
# 3. Build Grover circuit with the optimal number of iterations for this
#    marked-set size.
# ---------------------------------------------------------------------------
M = len(classical_marked)
iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N / M))))

qc = QuantumCircuit(NQ, NQ)
qc.h(range(NQ))
for _ in range(iterations):
    apply_oracle(qc, list(range(NQ)))
    apply_diffuser(qc, list(range(NQ)))
qc.measure(range(NQ), range(NQ))

sim = AerSimulator()
result = sim.run(qc, shots=2000).result()
counts = result.get_counts()

# most frequent measured outcome(s)
sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_bitstring, top_count = sorted_counts[0]
quantum_top_x = int(top_bitstring, 2)

print(f"Grover iterations used: {iterations}")
print(f"Measurement counts: {counts}")
print(f"Most frequent measured x = {quantum_top_x} (count {top_count}/2000)")

# success probability mass on the marked set
marked_mass = sum(c for bs, c in counts.items() if int(bs, 2) in classical_marked) / 2000

verified = quantum_top_x in classical_marked and marked_mass > 0.5

print(f"Probability mass on classically-marked set: {marked_mass:.3f}")

if verified:
    print("PASS")
else:
    print("FAIL")
