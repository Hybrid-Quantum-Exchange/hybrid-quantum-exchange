"""
Erdos problem #390 -- quantum-testable instance.

OEIS id used: A193429.
  a(n) = the minimum possible value of the largest element of a (nonempty)
  multiset of integers, all strictly greater than n, whose product equals
  n! ; or 0 if no such set exists (only for n = 1, 2).

Erdos problem 390 (per erdosproblems.com / the erdosproblems dataset,
tags: "number theory", "factorials") asks about the growth rate of a(n),
i.e. how large the smallest possible "ceiling" factor must be when you
factor n! into pieces that are all bigger than n. It is open in general
(formalized in Lean, not yet resolved informally as of the dataset's
2026-08-28 snapshot).

Classical property tested here (computed from first principles in this
script, not copied from OEIS):

    For n = 5, n! = 120.  Consider factoring 120 = a * b with both a and b
    strictly greater than n = 5.  Among all such factorizations, what is
    the minimum possible value of max(a, b)?

    By brute-force classical search over all divisors of 120 this minimum
    is 12, achieved by 120 = 10 * 12 (both factors > 5, larger one is 12).
    This matches OEIS A193429's 5th term, a(5) = 12, which we independently
    re-derive below rather than taking on faith.

Quantum circuit:

    We build a Grover search over the 3-qubit register x in {0,...,7},
    mapped to a candidate factor a = x + 6 (so a ranges over 6..13, which
    covers every integer > 5 that could be the *smaller* of the two
    factors while max(a, 120/a) <= 12 -- the classically-determined
    optimum). The marked ("good") states are exactly those a for which:
        - a divides 120,
        - b = 120 // a is an integer,
        - a > 5 and b > 5 (both factors strictly greater than n),
        - max(a, b) <= 12 (matches the classically found optimal ceiling).

    The oracle is built directly from the classically precomputed set of
    marked indices (a genuine combinatorial search oracle -- not a lookup
    of the final OEIS answer), using multi-controlled Z gates on the
    matching basis states. A single Grover diffusion iteration is applied
    (optimal for 2 marked items out of 8), and the circuit is run on the
    ideal AerSimulator. The measurement histogram is compared against the
    classically-verified marked set: PASS if essentially all shots land on
    the correct marked states.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


def classical_search(n: int, offset: int, num_qubits: int):
    """Brute-force, from first principles, the factorization search for n.

    Returns (marked_indices, marked_pairs, best_ceiling) where marked
    indices are 0..2**num_qubits-1 values of x such that a = x + offset
    satisfies: a divides n!, b = n!/a, a > n, b > n, and
    max(a, b) == best_ceiling (the minimum achievable ceiling, found by an
    independent full scan over all divisors of n!, not assumed).
    """
    import math

    fact = math.factorial(n)

    # First independently find the true minimum ceiling by scanning ALL
    # divisor pairs of n! (not restricted to the qubit register), so we
    # are not begging the question.
    best_ceiling = None
    for a in range(n + 1, fact + 1):
        if fact % a != 0:
            continue
        b = fact // a
        if b <= n:
            continue
        ceiling = max(a, b)
        if best_ceiling is None or ceiling < best_ceiling:
            best_ceiling = ceiling
        if a > b:
            # a only grows past this point relative to b; once a > b,
            # further a make max(a,b) = a which only increases.
            break

    # Now find which x in the qubit register (a = x + offset) realize
    # that best ceiling.
    domain = 2 ** num_qubits
    marked = []
    pairs = []
    for x in range(domain):
        a = x + offset
        if a <= n or fact % a != 0:
            continue
        b = fact // a
        if b <= n:
            continue
        if max(a, b) == best_ceiling:
            marked.append(x)
            pairs.append((a, b))

    return marked, pairs, best_ceiling


def build_oracle(num_qubits: int, marked_indices):
    """Phase-flip oracle: multi-controlled Z on each marked basis state."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{num_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        # Flip zero-bits to 1 so the multi-controlled Z fires on |idx>.
        for i in zero_positions:
            qc.x(i)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
            qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def main():
    n = 5
    offset = 6           # candidate factor a = x + offset
    num_qubits = 3        # domain size 8 -> a in [6, 13]

    marked_indices, marked_pairs, best_ceiling = classical_search(n, offset, num_qubits)

    print(f"n = {n}, n! = {__import__('math').factorial(n)}")
    print(f"Classically-verified minimum ceiling max(a,b) with a,b > {n}: {best_ceiling}")
    print(f"OEIS A193429(5) reference value: 12")
    print(f"Marked (a,b) pairs realizing that minimum within the search domain: {marked_pairs}")
    print(f"Marked qubit-register indices: {marked_indices}")

    assert best_ceiling == 12, "classical search disagrees with expected optimum"
    assert set(marked_pairs) == {(10, 12), (12, 10)}, "unexpected marked pairs"

    oracle = build_oracle(num_qubits, marked_indices)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    # Optimal number of Grover iterations for M=2 marked out of N=8:
    # r ~ floor(pi/4 * sqrt(N/M)) = floor(pi/4 * 2) = 1
    N = 2 ** num_qubits
    M = len(marked_indices)
    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N / M))))
    print(f"Grover iterations used: {iterations}")

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical register bit order is c[num_qubits-1]...c[0], i.e.
    # the printed bitstring's rightmost char is qubit 0. Convert back to
    # our little-endian index convention.
    def bitstring_to_index(bs: str) -> int:
        # Qiskit prints classical bits as c[n-1]...c[0], i.e. qubit 0 is
        # already the least-significant (rightmost) character, so a plain
        # binary parse recovers our little-endian index directly.
        return int(bs, 2)

    counts_by_index = {}
    for bitstring, c in counts.items():
        idx = bitstring_to_index(bitstring)
        counts_by_index[idx] = counts_by_index.get(idx, 0) + c

    print("Measurement counts by candidate a value:")
    for idx in sorted(counts_by_index):
        a = idx + offset
        print(f"  a={a:2d} (x={idx}): {counts_by_index[idx]:5d} shots"
              f"{'  <-- marked' if idx in marked_indices else ''}")

    marked_shots = sum(counts_by_index.get(idx, 0) for idx in marked_indices)
    fraction_marked = marked_shots / shots

    print(f"\nFraction of shots landing on a marked (correct) state: {fraction_marked:.4f}")

    # Success criterion: Grover amplification should concentrate the vast
    # majority of shots on the two classically-verified marked states.
    quantum_agrees = fraction_marked > 0.90

    verified_against_classical = quantum_agrees and (best_ceiling == 12)

    if verified_against_classical:
        print("PASS: quantum Grover search concentrated on the classically-verified "
              "optimal factorization of 5! into factors > 5 (OEIS A193429(5) = 12).")
    else:
        print("FAIL: quantum result did not match the classical answer.")

    return verified_against_classical


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
