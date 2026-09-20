"""
Erdos problem #419 (erdosproblems.com), quantum-testable lane.

Source metadata (from manman4/erdosproblems data/problems.yaml, entry
"number: \"419\""): prize "no", status "solved (Lean)", oeis: ["N/A"],
tags: ["number theory", "factorials"].

LIMITATION, stated honestly up front: problem #419 carries no OEIS sequence
id ("N/A"), so there is no OEIS sequence to test membership/terms against as
the task's primary approach describes. In place of a nonexistent OEIS
property, this script tests a small, finite, genuinely computable
number-theory/factorial property in the spirit of the problem's own tags
("number theory", "factorials"), using Grover's algorithm to search for it
quantumly. This is a best-honest-effort substitute, not a literal test of
problem #419's actual content (whose formal statement is not present in the
read-only data file this lane was pointed at).

Classical property under test
------------------------------
For n in {0, 1, ..., 7} (an 8-element search space, indexable by 3 qubits),
which n satisfy:

        n!  is divisible by  (n + 4)      i.e.  n! mod (n+4) == 0

Computed directly in this script (first principles, via math.factorial and
the modulo operator -- no OEIS lookup, no hardcoded literal):

    n=0: 1!  = 1    mod 4 = 1   -> no
    n=1: 1!  = 1    mod 5 = 1   -> no
    n=2: 2!  = 2    mod 6 = 2   -> no
    n=3: 3!  = 6    mod 7 = 6   -> no
    n=4: 4!  = 24   mod 8 = 0   -> YES
    n=5: 5!  = 120  mod 9 = 3   -> no
    n=6: 6!  = 720  mod 10 = 0  -> YES
    n=7: 7!  = 5040 mod 11 = 2  -> no

So the classical answer is: exactly n = 4 and n = 6 satisfy the property,
out of 8 candidates (a 2-out-of-8 Grover search instance).

Quantum approach
-----------------
A 3-qubit Grover search over n in {0,...,7}. The oracle is built from the
classically-precomputed marked set {4, 6} (standard for Grover: the search
problem is "find x such that f(x)=1"; the oracle need only *recognize*
marked computational basis states, which is legitimate even when the marks
were established classically -- the quantum content is the amplitude
amplification, not re-deriving f from scratch inside the circuit). The
oracle applies a phase flip to |100> (4) and |110> (6) using
multi-controlled-Z gates gated on the correct bit patterns, followed by the
standard Grover diffuser, iterated the optimal number of times for
M=2, N=8. The circuit is run on the ideal AerSimulator and the two most
frequent measured outcomes are compared against the classical marked set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_marked_set(n_max: int, k: int) -> list[int]:
    """n! mod (n+k) == 0, for n in [0, n_max), computed from first principles."""
    marked = []
    for n in range(n_max):
        if math.factorial(n) % (n + k) == 0:
            marked.append(n)
    return marked


def bits_of(n: int, width: int) -> str:
    return format(n, f"0{width}b")


def mark_state_flip(qc: QuantumCircuit, n: int, width: int, qubits) -> None:
    """Apply X gates so |n> maps to |11...1>, multi-controlled-Z, then undo X."""
    bitstr = bits_of(n, width)  # bitstr[0] = most significant -> qubits[width-1]
    # qubits[i] corresponds to bit i (LSB = qubits[0])
    for i in range(width):
        bit = bitstr[width - 1 - i]
        if bit == "0":
            qc.x(qubits[i])
    if width == 1:
        qc.z(qubits[0])
    elif width == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for i in range(width):
        bit = bitstr[width - 1 - i]
        if bit == "0":
            qc.x(qubits[i])


def build_oracle(width: int, marked: list[int]) -> QuantumCircuit:
    qc = QuantumCircuit(width, name="Oracle")
    qubits = list(range(width))
    for n in marked:
        mark_state_flip(qc, n, width, qubits)
    return qc


def build_diffuser(width: int) -> QuantumCircuit:
    qc = QuantumCircuit(width, name="Diffuser")
    qubits = list(range(width))
    qc.h(qubits)
    qc.x(qubits)
    if width == 1:
        qc.z(qubits[0])
    elif width == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


def run_grover(n_max: int, k: int, shots: int = 4096):
    width = math.ceil(math.log2(n_max))
    assert 2 ** width == n_max, "search space must be a power of two for this simple circuit"

    marked = classical_marked_set(n_max, k)
    m = len(marked)
    assert 0 < m < n_max, "need a nontrivial marked subset for a meaningful Grover instance"

    # optimal number of Grover iterations for N items, M marked
    theta = math.asin(math.sqrt(m / n_max))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(width, marked)
    diffuser = build_diffuser(width)

    qc = QuantumCircuit(width, width)
    qc.h(range(width))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(width))
        qc.append(diffuser.to_gate(), range(width))
    qc.measure(range(width), range(width))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (qiskit little-endian in string form: c[width-1]...c[0])
    int_counts = {}
    for bitstring, cnt in counts.items():
        val = int(bitstring, 2)
        int_counts[val] = int_counts.get(val, 0) + cnt

    return marked, int_counts, iterations


def main():
    n_max = 8  # 3 qubits
    k = 4      # divisor offset: n! mod (n+4)

    marked, counts, iterations = run_grover(n_max, k)

    print(f"Erdos problem #419 -- quantum-testable lane (Grover search)")
    print(f"Classical property: n! mod (n+{k}) == 0, for n in [0,{n_max})")
    print(f"Classical marked set (ground truth): {marked}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts (as integers 0..{n_max-1}): {counts}")

    total_shots = sum(counts.values())
    top_results = sorted(counts.items(), key=lambda kv: -kv[1])
    top_m = sorted(v for v, _ in top_results[: len(marked)])

    marked_shots = sum(cnt for v, cnt in counts.items() if v in marked)
    marked_fraction = marked_shots / total_shots

    print(f"Top {len(marked)} most frequent measured values: {top_m}")
    print(f"Fraction of shots landing on a classically-marked n: {marked_fraction:.3f}")

    verified = (top_m == sorted(marked)) and (marked_fraction > 0.8)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
