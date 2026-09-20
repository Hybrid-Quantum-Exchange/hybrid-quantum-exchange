"""
Erdos problem #728 -- quantum-testable companion script.

Source metadata (data/problems.yaml, entry `number: "728"`):
    prize: no
    informal_status: proved (Lean, last_update 2026-01-05)
    oeis: ["N/A"]
    tags: ["number theory", "factorials"]

LIMITATION, stated up front: problem #728 has no OEIS id attached in the
source data (oeis: ["N/A"]), so there is no literal sequence to target with
a membership/term-search circuit as the task description's primary
examples assume. Rather than fabricate an OEIS id or copy a term that
doesn't exist, this script builds a genuine, independently-checkable
finite/computable property drawn directly from the problem's own tags
("number theory", "factorials"): Wilson's theorem.

    Wilson's theorem: for integer n >= 2,
        (n - 1)! === -1  (mod n)      iff n is prime.

This is a real, small, finite decision property (not a fabricated one):
for each n in a fixed range we can classically compute (n-1)! mod n and
compare it to n-1, and the set of n for which it holds is exactly the
primes in that range -- a fact we verify from first principles below,
independently of any lookup table.

Quantum approach: Grover search over n in [0, 15] (4 qubits) for the
condition "(n-1)! mod n == n-1". The oracle is built directly from the
classically-precomputed marked set (a small, honest search space -- this
is exactly the "small search space whose answer is a known [classically
verifiable] term" case), using multi-controlled Z gates, i.e. an exact
diagonal phase oracle over 16 basis states. Grover's algorithm is then run
on the ideal AerSimulator for the optimal number of iterations, and the
measured distribution is compared against the classically-derived answer
set.

No OEIS term is copied uninspected: the marked set is computed in this
script by direct factorial/modulo arithmetic, and separately cross-checked
against a plain primality sieve, before ever being handed to the circuit.
"""

import math
import sys

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

N_QUBITS = 4
N = 2 ** N_QUBITS  # search space: n = 0 .. 15


def wilson_holds(n: int) -> bool:
    """True iff (n-1)! mod n == n-1 (Wilson's theorem condition)."""
    if n < 2:
        return False
    return math.factorial(n - 1) % n == n - 1


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_marked_set():
    """Compute, from first principles, the n in [0, N) satisfying Wilson's
    theorem, and cross-check the result equals the set of primes in range."""
    wilson_set = sorted(n for n in range(N) if wilson_holds(n))
    prime_set = sorted(n for n in range(N) if is_prime(n))
    assert wilson_set == prime_set, (
        f"Wilson's theorem mismatch in verification range: "
        f"{wilson_set} != {prime_set}"
    )
    return wilson_set


def build_oracle(marked, n_qubits):
    """Exact diagonal phase oracle: flips the sign of each marked basis
    state |n> via a multi-controlled Z (built from H + MCX + H)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        # multi-controlled Z on all n_qubits, target = last qubit
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
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked, n_qubits, shots=4096):
    n_marked = len(marked)
    n_total = 2 ** n_qubits
    # optimal number of Grover iterations
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    marked = classical_marked_set()
    print(f"Search space: n in [0, {N})")
    print(f"Classical marked set (Wilson's theorem holds, == primes): {marked}")

    counts, iterations = run_grover(marked, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    # Aggregate measured probability mass landing on marked states
    total_shots = sum(counts.values())
    marked_bitstrings = {format(m, f"0{N_QUBITS}b")[::-1] for m in marked}
    # Qiskit's classical register bit order in counts keys is big-endian
    # over qubit index (c[n-1] ... c[0]); build the matching set directly.
    marked_keys = set()
    for m in marked:
        bits = format(m, f"0{N_QUBITS}b")  # MSB..LSB matches qubit N-1..0
        marked_keys.add(bits)

    marked_hits = sum(v for k, v in counts.items() if k in marked_keys)
    marked_prob = marked_hits / total_shots

    # Most frequent measured outcome
    top_bitstring = max(counts, key=counts.get)
    top_n = int(top_bitstring, 2)

    print(f"Counts: {counts}")
    print(f"Fraction of shots landing on a marked (Wilson-satisfying) n: "
          f"{marked_prob:.3f}")
    print(f"Most frequent measured n: {top_n} "
          f"(in classical marked set: {top_n in marked})")

    # Success criteria:
    #  1. The most-frequent measured outcome must itself satisfy Wilson's
    #     theorem (i.e. be a genuine answer, not noise).
    #  2. Grover amplification must concentrate a majority of shots on the
    #     marked subspace, well above the uniform-random baseline
    #     (|marked| / N).
    baseline = len(marked) / N
    success = (top_n in marked) and (marked_prob > max(0.5, 2 * baseline))

    if success:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
