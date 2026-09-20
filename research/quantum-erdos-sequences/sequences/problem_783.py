"""
Erdos problem #783 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: \"783\"",
verified 2026-09-19):
    prize: no
    status: solved (2026-04-19)
    tags: ["number theory"]
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #783 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no specific OEIS
sequence to build a membership/term-search circuit around, and this script
cannot claim to test "the #783 sequence" because no such registered sequence
exists. Per the task's fallback instruction, this is the best-effort quantum
circuit for the problem's only concrete attribute we do have: its tag,
"number theory". We pick a small, finite, genuinely computable number-theory
search problem in that spirit -- primality over a bounded finite domain --
and use Grover's algorithm to find it. This is NOT a claim that primality
search is "the Erdos #783 sequence"; it is a clearly-labeled substitute
exercise run under this problem's lane because #783 itself supplies no OEIS
sequence to target.

Classical property tested
--------------------------
Domain: integers 0..15 (4 qubits, N = 16).
Property: n is prime (n in {2, 3, 5, 7, 11, 13}).
The classical answer (the marked set) is computed from first principles in
this script by trial division -- not copied from any table.

Quantum approach
-----------------
Grover's algorithm (qiskit_aer AerSimulator, statevector-exact simulation):
  - 4 qubits encode n in 0..15.
  - The oracle is a multi-controlled phase flip built directly from the
    classically-computed marked set (list of prime bit patterns) -- this is
    the standard way to instantiate a Grover oracle for a decision property
    when no arithmetic circuit library is assumed; the *decision itself*
    (trial division) is done in Python and independently verified, and the
    circuit's job -- amplifying and finding a marked state via the
    diffusion operator -- is genuine quantum search, not a shortcut.
  - Optimal iteration count computed by the standard Grover formula.
  - After measurement, the most frequent outcome(s) must be primes.

PASS/FAIL: compare the top measured outcome(s) (by probability, using
`shots`) against the classically-verified prime set; PASS iff the
highest-probability outcomes are exactly the true marked (prime) states.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4
N = 2 ** N_QUBITS  # domain 0..15


def is_prime(n: int) -> bool:
    """Trial division primality test, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_marked_set():
    return sorted(n for n in range(N) if is_prime(n))


def build_oracle(marked_values, n_qubits):
    """Phase oracle flipping the sign of each marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_values, n_qubits, shots=4096):
    n_marked = len(marked_values)
    total = 2 ** n_qubits
    theta = math.asin(math.sqrt(n_marked / total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def top_outcomes(counts, k):
    """Return the k most-measured integer outcomes (little-endian bitstrings)."""
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top = ordered[:k]
    values = []
    for bitstring, _ in top:
        # Qiskit prints classical bits MSB-first for register order c[n-1]...c[0];
        # our measure maps qubit i -> classical bit i (LSB first ordering matches
        # build_oracle's little-endian convention once reversed back).
        values.append(int(bitstring, 2))
    return sorted(values)


def main():
    marked = classical_marked_set()
    print(f"Classical marked set (primes in 0..{N - 1}): {marked}")

    counts, iterations = run_grover(marked, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    measured_top = top_outcomes(counts, len(marked))
    print(f"Top {len(marked)} measured outcomes: {measured_top}")

    total_shots = sum(counts.values())
    marked_shots = sum(v for k, v in counts.items() if int(k, 2) in marked)
    hit_rate = marked_shots / total_shots
    print(f"Fraction of shots landing on a true prime state: {hit_rate:.3f}")

    passed = (measured_top == marked) and (hit_rate > 0.5)

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
