"""
Erdos problem #475 -- quantum-testable lane.

Source: /home/user/manman4/erdosproblems/data/problems.yaml, entry
    - number: "475"
      prize: "no"
      informal_status: {state: "decidable", last_update: "2026-02-23"}
      formal_status: {state: "unformalized"}
      status: {state: "decidable", last_update: "2026-02-23"}
      oeis: ["N/A"]
      tags: ["number theory", "additive combinatorics"]

LIMITATION (read before trusting the "PASS" below):
Problem #475 carries no OEIS sequence id -- its `oeis` field is the literal
string "N/A". There is therefore no concrete integer sequence tied to this
problem number to build a genuine, problem-specific quantum test around, and
no further textual statement of the problem is available in this read-only
clone to derive one from either. Per the task's fallback instruction, this
script is my best honest attempt rather than a fabricated pass: it builds a
REAL Grover-search circuit on the ideal AerSimulator that tests a genuine,
independently-checkable, finite, computable number-theoretic property from
the problem's own tags ("number theory" -- primality is about as canonical
as number theory gets, and it is decidable and finite for any bounded
search space, matching the "decidable" status recorded above). It is a
generic stand-in, NOT derived from problem #475's actual (unavailable)
statement, and that is reported honestly in the result below rather than
claimed as a verification of Erdos problem #475 itself.

Chosen small instance:
  Search space: x in {0, 1, ..., 15}  (4 qubits, N = 16)
  Property tested: "x is prime"
  Classical answer (computed here from first principles, trial division,
  no library call and no OEIS lookup): the marked set is
  {2, 3, 5, 7, 11, 13} (6 of 16 values). N=8 with exactly half the values
  prime ({2,3,5,7} of 8) was tried first but is the well-known Grover
  degenerate case M=N/2, where a single iteration exactly zeroes out the
  marked amplitudes instead of amplifying them -- so N=16 (M=6/16, not a
  degenerate ratio) is used instead, and this is the instance actually run.

Circuit: exact-count Grover search (2 grover iterations optimal for
M=4 marked out of N=8, since the analytic optimal iteration count
floor(pi/4 * sqrt(N/M)) = floor(pi/4 * sqrt(2)) = 1, but with M=N/2 the
uniform superposition after 1 iteration is provably no better than 0 --
so we instead compute the optimal iteration count from the exact Grover
formula and use it below). The oracle phase-flips exactly the basis
states corresponding to primes in range, built directly from the
classically-computed marked set (a standard, non-fabricated way to build
a Grover oracle for a property with no simpler arithmetic circuit); the
diffuser is the standard Grover diffusion operator. Success is verified
by checking, on the ideal simulator, that measurement outcomes concentrate
on the classically-computed prime set with much higher probability than
the uniform-superposition baseline (1 - 4/8 = 50% baseline is not a
strict enough test for M=N/2, so instead we verify each of the 4 marked
outcomes individually exceeds each of the 4 unmarked outcomes in
measured probability -- a concentration test that a working Grover
oracle+diffuser must satisfy and a broken one will not).
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_marked_set(n_qubits: int):
    """Compute, from first principles, the set of x in [0, 2**n_qubits)
    that are prime."""
    N = 2 ** n_qubits
    return sorted(x for x in range(N) if is_prime(x))


def build_oracle(n_qubits: int, marked):
    """Phase-flip oracle: for each marked value, X-gate the 0-bits so the
    marked bitstring becomes all-ones, apply a multi-controlled Z (via
    H-MCX-H on the last qubit), then undo the X-gates."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked:
        bits = format(value, f"0{n_qubits}b")  # MSB first
        # qiskit qubit 0 is least-significant; align bits accordingly
        bits_lsb_first = bits[::-1]
        flip_qubits = [i for i, b in enumerate(bits_lsb_first) if b == "0"]
        if flip_qubits:
            qc.x(flip_qubits)
        # multi-controlled Z across all n_qubits, implemented as H + MCX + H
        # on the target qubit (last qubit acts as target/phase kickback)
        target = n_qubits - 1
        controls = [q for q in range(n_qubits) if q != target]
        qc.h(target)
        qc.mcx(controls, target)
        qc.h(target)
        if flip_qubits:
            qc.x(flip_qubits)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    target = n_qubits - 1
    controls = [q for q in range(n_qubits) if q != target]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_iterations(n_qubits: int, num_marked: int) -> int:
    N = 2 ** n_qubits
    if num_marked <= 0 or num_marked >= N:
        return 0
    theta = math.asin(math.sqrt(num_marked / N))
    # optimal integer iteration count for exact-count Grover
    r = round((math.pi / (4 * theta)) - 0.5)
    return max(1, int(r))


def main():
    n_qubits = 4  # search space {0,...,15}; avoids the M=N/2 degenerate case
    N = 2 ** n_qubits

    marked = classical_marked_set(n_qubits)
    unmarked = sorted(set(range(N)) - set(marked))
    print(f"Classical answer -- primes in [0,{N-1}]: {marked}")
    print(f"Non-primes (unmarked) in [0,{N-1}]: {unmarked}")

    iterations = grover_iterations(n_qubits, len(marked))
    print(f"Grover iterations used: {iterations}")

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports bitstrings MSB-first over the classical register, which
    # here mirrors qubit order 0=LSB, so parse as a binary integer directly.
    value_counts = Counter()
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        value_counts[value] += c

    probs = {v: value_counts.get(v, 0) / shots for v in range(N)}
    print("Measured probabilities:", {v: round(p, 4) for v, p in sorted(probs.items())})

    min_marked_prob = min(probs[v] for v in marked)
    max_unmarked_prob = max(probs[v] for v in unmarked)
    print(f"min P(marked)={min_marked_prob:.4f}  max P(unmarked)={max_unmarked_prob:.4f}")

    # Concentration test: every marked (prime) outcome must be measured more
    # often than every unmarked (composite/0/1) outcome -- the signature of a
    # correctly amplifying Grover oracle+diffuser pair, verified against the
    # independently, classically computed prime set.
    verified = min_marked_prob > max_unmarked_prob

    # Sanity cross-check: also confirm the single most frequent outcome is
    # itself prime.
    most_common_value = max(probs, key=probs.get)
    most_common_is_prime = most_common_value in marked
    verified = verified and most_common_is_prime

    print(f"Most frequent measured value: {most_common_value} "
          f"(prime: {most_common_is_prime})")

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
