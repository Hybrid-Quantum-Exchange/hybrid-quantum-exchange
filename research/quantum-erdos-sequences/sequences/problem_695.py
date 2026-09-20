"""
Erdos problem #695 -- quantum-testable instance from OEIS A061092.

Erdos problem #695 (see erdosproblems.com / manman4/erdosproblems data,
number: "695", tags: ["number theory"]) is linked to OEIS sequence
A061092: "a(0) = 1; for n>0, a(n) is the smallest prime of the form
k*a(n-1) + 1 (k a positive integer)."  The sequence begins
1, 2, 3, 7, 29, 59, 709, 2837, 22697, ...
Its infinitude rests on Dirichlet's theorem on primes in arithmetic
progressions (every prime p admits some prime of the form k*p + 1).

Classical property tested here (computed from first principles below,
not copied from OEIS):

    Fix base = a(3) = 7 (the 4th term of A061092).  Consider all
    k in {0, 1, ..., 7} (3 bits).  A value k is a "hit" iff 7*k + 1
    is prime.  We classically enumerate the hit set, and in
    particular confirm that the *smallest* positive hit is k = 4,
    which reproduces a(4) = 7*4 + 1 = 29, the next term of A061092.
    This is a genuine finite search problem: "which k in a bounded
    range make k*base + 1 prime", i.e. exactly the search Dirichlet's
    theorem guarantees a solution to, restricted to a small window.

Quantum circuit:

    A 3-qubit Grover search over k in {0,...,7}. The oracle marks
    exactly the classically-precomputed hit set {k : 7k+1 is prime}
    by applying a multi-controlled phase flip on each hit's basis
    pattern (bits are computed classically and hard-wired into which
    control qubits get X-gates -- no classical hit is asserted
    without having been primality-tested in this script first). One
    Grover iteration (optimal for |hits|=2 out of N=8) amplifies the
    hit amplitudes. We run the circuit on the ideal AerSimulator and
    check that the measurement distribution is concentrated (with
    high total probability) on the classically-known hit set, and
    that the single most-sampled outcome equals the smallest hit,
    k = 4, matching a(4) = 29 in A061092.

PASS/FAIL is decided by comparing the quantum sampling result against
the independently, classically computed hit set and smallest hit.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np


def is_prime(n: int) -> bool:
    """Trial-division primality test, first principles, no libraries."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def classical_hits(base: int, n_bits: int):
    """All k in [0, 2**n_bits) with base*k + 1 prime."""
    hits = []
    for k in range(2 ** n_bits):
        if is_prime(base * k + 1):
            hits.append(k)
    return hits


def bits_of(k: int, n_bits: int):
    """LSB-first bit list of k, matching Qiskit's little-endian qubit order."""
    return [(k >> i) & 1 for i in range(n_bits)]


def mark_state(qc: QuantumCircuit, k: int, n_bits: int):
    """Apply a phase flip to basis state |k> via multi-controlled Z."""
    bits = bits_of(k, n_bits)
    zero_qubits = [i for i, b in enumerate(bits) if b == 0]
    for q in zero_qubits:
        qc.x(q)
    if n_bits == 1:
        qc.z(0)
    elif n_bits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    for q in zero_qubits:
        qc.x(q)


def diffuser(qc: QuantumCircuit, n_bits: int):
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))


def build_grover(hits, n_bits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    for _ in range(iterations):
        for k in hits:
            mark_state(qc, k, n_bits)
        diffuser(qc, n_bits)
    qc.measure(range(n_bits), range(n_bits))
    return qc


def main():
    base = 7          # a(3) in A061092 (1,2,3,7,29,...)
    n_bits = 3         # search k in 0..7
    N = 2 ** n_bits

    # --- classical ground truth, computed here from first principles ---
    hits = classical_hits(base, n_bits)
    assert hits, "no hits found -- Dirichlet guarantees one should exist"
    smallest_hit = min(h for h in hits if h > 0)
    next_term = base * smallest_hit + 1
    print(f"Classical: base={base}, hits(k with {base}*k+1 prime) = {hits}")
    print(f"Classical: smallest positive hit k = {smallest_hit}, "
          f"{base}*{smallest_hit}+1 = {next_term}")
    assert smallest_hit == 4 and next_term == 29, (
        "classical computation does not match expected A061092 term 29"
    )

    M = len(hits)
    theta = np.arcsin(np.sqrt(M / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
    print(f"Grover: N={N} states, M={M} marked, iterations={iterations}")

    qc = build_grover(hits, n_bits, iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical register bit string is MSB-first over the
    # measured qubits; qubit i (LSB-first k) maps to bitstring[::-1][i].
    def bitstring_to_k(bs: str) -> int:
        bits = bs[::-1]
        return sum(int(bits[i]) << i for i in range(n_bits))

    dist = {}
    for bs, c in counts.items():
        k = bitstring_to_k(bs)
        dist[k] = dist.get(k, 0) + c

    print("Quantum measurement distribution (k -> counts):",
          dict(sorted(dist.items())))

    hit_prob = sum(dist.get(k, 0) for k in hits) / shots
    most_sampled_k = max(dist, key=dist.get)

    print(f"Quantum: probability mass on classical hit set = {hit_prob:.3f}")
    print(f"Quantum: most-sampled k = {most_sampled_k}")

    amplification_ok = hit_prob > 0.8  # far above uniform baseline M/N = 0.25
    correct_answer_ok = most_sampled_k == smallest_hit

    verified = amplification_ok and correct_answer_ok

    print(f"amplification_ok={amplification_ok} correct_answer_ok={correct_answer_ok}")
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
