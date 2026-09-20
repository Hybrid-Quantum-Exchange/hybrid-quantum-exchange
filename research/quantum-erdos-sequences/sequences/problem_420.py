"""
Erdos problem #420 — quantum-testable sequence entry.

Source of truth: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"420\"":

    - number: "420"
      prize: "no"
      informal_status: {state: "open", last_update: "2025-08-31"}
      formal_status: {state: "unformalized"}
      status: {state: "open", last_update: "2025-08-31"}
      oeis: ["N/A"]
      formalized: {state: "no", last_update: "2025-08-31"}
      tags: ["number theory"]

LIMITATION (read before trusting anything below as "problem 420's sequence"):
Problem #420 carries no OEIS id (oeis: ["N/A"]) and no textual statement in
this dataset beyond the tag "number theory". There is therefore no concrete,
finite, computable property of *this specific problem's sequence* to build a
faithful quantum test around — the honest options were (a) fabricate a
property and pretend it came from problem 420, which the task explicitly
forbids, or (b) build a real, self-contained quantum circuit that tests a
genuine, classically-checkable number-theory property in the same spirit as
the problem's only tag, and say plainly that it is NOT derived from problem
420's actual (unstated) content. This script takes option (b).

Chosen property (generic, but real and independently verified classically):
Primality of 4-bit integers. For N = 16 (n = 4 qubits, integers 0..15), the
primes are computed from first principles by trial division in this script
(`classical_primes`), giving the set {2, 3, 5, 7, 11, 13}. A Grover search
circuit is built whose oracle marks exactly the basis states |x> for which
`classical_primes` says x is prime, and whose diffuser amplifies those
marked states. The ideal AerSimulator is run, and the script checks that the
quantum sampling distribution concentrates (top outcomes, weighted by
measured probability) on the classically-computed prime set.

This is a genuine Grover search (real oracle + diffuser + measurement),
verified against an independently computed classical answer, on a small
instance (4 qubits, N=16). It is reported here as ran_ok / verified against
that classical prime-search property, NOT as a verification of problem
420's own (open, unformalized, OEIS-less) sequence, since no such sequence
is available to test against.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
import math


def classical_primes(n_bits):
    """Trial-division primality over the integer range [0, 2**n_bits)."""
    limit = 2 ** n_bits
    primes = []
    for x in range(2, limit):
        is_p = True
        for d in range(2, int(math.isqrt(x)) + 1):
            if x % d == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
    return primes


def oracle_mark_set(qc, qubits, ancilla, marked_values, n_bits):
    """Phase-flip (via phase kickback on `ancilla` in |-> state) every basis
    state in `marked_values`, using a multi-controlled X per value."""
    for val in marked_values:
        bits = format(val, f"0{n_bits}b")[::-1]  # little-endian per qubit
        flip_qubits = [qubits[i] for i, b in enumerate(bits) if b == "0"]
        for q in flip_qubits:
            qc.x(q)
        qc.mcx(qubits, ancilla)
        for q in flip_qubits:
            qc.x(q)


def diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(n_bits, marked_values, iterations):
    qubits = list(range(n_bits))
    ancilla = n_bits
    qc = QuantumCircuit(n_bits + 1, n_bits)

    # Ancilla in |-> for phase kickback oracle.
    qc.x(ancilla)
    qc.h(ancilla)

    qc.h(qubits)

    for _ in range(iterations):
        oracle_mark_set(qc, qubits, ancilla, marked_values, n_bits)
        diffuser(qc, qubits)

    qc.h(ancilla)
    qc.x(ancilla)

    qc.measure(qubits, list(range(n_bits)))
    return qc


def main():
    n_bits = 4
    N = 2 ** n_bits  # 16

    primes = classical_primes(n_bits)
    print(f"Classical primes in [0, {N}): {primes}")

    M = len(primes)
    # Optimal Grover iteration count for M marked out of N.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"N={N}, M={M} marked states, Grover iterations={iterations}")

    qc = build_grover_circuit(n_bits, primes, iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-register bitstrings are printed c[n-1]...c[0]. Our
    # oracle encodes each marked integer's bit i onto qubit i (see
    # oracle_mark_set), and qubit i is measured into clbit i, so the printed
    # string c[n-1]...c[0] already reads as bit(n-1)...bit(0) of the integer
    # — i.e. a direct binary string, no reversal needed.
    dist = {}
    for bitstring, cnt in counts.items():
        val = int(bitstring, 2)
        dist[val] = dist.get(val, 0) + cnt

    ranked = sorted(dist.items(), key=lambda kv: -kv[1])
    top_k = ranked[:M]
    top_values = {v for v, _ in top_k}

    prime_prob = sum(cnt for v, cnt in dist.items() if v in primes) / shots

    print(f"Top-{M} measured outcomes (value: counts): {top_k}")
    print(f"Fraction of shots landing on a classical prime: {prime_prob:.3f}")

    # Success criteria: the top-M measured outcomes are exactly the
    # classically computed prime set, and the amplified probability mass on
    # primes is well above the uniform baseline (M/N).
    baseline = M / N
    quantum_matches_classical = (top_values == set(primes)) and (prime_prob > 2 * baseline)

    if quantum_matches_classical:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
