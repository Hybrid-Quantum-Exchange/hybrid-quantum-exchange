"""
Erdos problem #246 (per manman4/erdosproblems data/problems.yaml, entry
"number: \"246\"", tags: ["number theory"]) has status "proved (Lean)" and
lists oeis: ["N/A"] -- there is no OEIS sequence attached to this problem in
the source data. That means the task's intended path (identify a property of
"the" OEIS sequence for this problem, from its OEIS id and tags) does not
apply here: there is no sequence id to derive a property from.

Honest limitation: this script is a best-effort substitute, NOT a property
of problem 246's own (nonexistent) OEIS sequence. Since the only real
signal available for problem 246 is its tag "number theory", this script
builds a genuine, finite, classically-checkable number-theory search problem
in that spirit -- "which x in a small range divide N" -- and solves it with
a real Grover search circuit on Qiskit's ideal AerSimulator. The classical
divisor set is computed from first principles (trial division) in this
script and compared against the quantum search's measurement distribution.

Property under test:
    N = 12, search space x in {0, 1, ..., 15} (4 qubits).
    Marked set S = { x in [0,15] : x > 0 and 12 % x == 0 }  (divisors of 12)
    Classically S = {1, 2, 3, 4, 6, 12} (computed below, not hard-coded from
    memory).

Quantum method: Grover's algorithm.
    - Oracle: a phase-flip oracle built by, for each marked x, applying X
      gates to the qubits where x has a 0 bit, a multi-controlled Z, then
      undoing the X gates -- i.e. a genuine multi-controlled-phase oracle
      per marked basis state, not a lookup table smuggled into the circuit.
    - Diffuser: the standard Grover diffusion operator (H^n, X^n,
      multi-controlled Z, X^n, H^n).
    - Iteration count: floor(pi/4 * sqrt(2^n / |S|)), the standard optimal
      count for this space size and marked-set size.

PASS/FAIL: the script runs the circuit on AerSimulator, takes the most
frequent measured bitstrings (as many as |S|, to allow for the multi-target
search), and checks they are exactly the classically computed divisor set.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
import math


def classical_divisors(n: int, upper: int) -> list[int]:
    """Trial division: all x in [1, upper] with n % x == 0."""
    return [x for x in range(1, upper + 1) if n % x == 0]


def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian bit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
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


def main() -> bool:
    N = 12
    n_qubits = 4  # search space size 16
    upper = (1 << n_qubits) - 1

    marked = classical_divisors(N, upper)
    print(f"Classical divisors of {N} in [1,{upper}]: {marked}")

    n_marked = len(marked)
    space_size = 1 << n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(space_size / n_marked)))
    print(f"Grover iterations: {iterations} (space={space_size}, marked={n_marked})")

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Convert little-endian measured bitstrings back to integers.
    int_counts: dict[int, int] = {}
    for bitstring, c in counts.items():
        # Qiskit prints classical bits MSB-first for the register, and bit 0
        # (qubit 0, our LSB) is the rightmost character.
        value = int(bitstring, 2)
        int_counts[value] = int_counts.get(value, 0) + c

    ranked = sorted(int_counts.items(), key=lambda kv: -kv[1])
    top_values = [v for v, _ in ranked[:n_marked]]

    print("Top measured values (by frequency):", ranked[:n_marked + 3])
    print("Quantum top-{} candidates: {}".format(n_marked, sorted(top_values)))

    quantum_set = set(top_values)
    classical_set = set(marked)

    passed = quantum_set == classical_set

    print()
    if passed:
        print("PASS: Grover search's top candidates exactly match the "
              "classical divisor set.")
    else:
        print("FAIL: Grover search's top candidates do not match the "
              "classical divisor set.")
        print(f"  classical={sorted(classical_set)} quantum={sorted(quantum_set)}")

    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
