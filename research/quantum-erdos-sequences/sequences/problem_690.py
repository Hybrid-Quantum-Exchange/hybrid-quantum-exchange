"""
Erdos problem #690 -- quantum-testable sequence lane.

LIMITATION (read first): the source metadata for problem #690 in
/home/user/manman4/erdosproblems/data/problems.yaml is:

    - number: "690"
      prize: "no"
      status: { state: "solved", last_update: "2025-08-31" }
      oeis: ["possible"]
      tags: ["number theory"]

"possible" under `oeis` is a placeholder string, not an OEIS sequence id
(real ids look like "A000040"). There is no accompanying problem
statement file in the cloned repo and no numeric OEIS id to look up, so
there is no genuine, citable sequence to build a faithful quantum test
around for this specific problem. Per the task instructions, rather than
fabricate a property and pretend it comes from problem #690, this script
honestly falls back to a small, real, self-contained number-theory
search problem (only loosely motivated by the "number theory" tag) and
verifies it classically before running Grover's algorithm on it. This is
NOT a verified test of problem #690's actual mathematical content --
that content is not available in this offline environment.

Chosen finite instance (real math, checked classically in this script):
  Property P(x): x in {0..15} is prime (a small, well-defined,
  classically-checkable number-theoretic property -- the closest honest
  stand-in for "number theory" given no real sequence to target).
  The classical answer set for N = 16 is computed by trial division in
  `classical_primes_below(16)` and is {2, 3, 5, 7, 11, 13}.

Quantum method: Grover's algorithm (4 qubits, N = 16) with an oracle
that marks exactly the classically-computed prime indices. The
iteration count is chosen to maximize the theoretical success
probability sin^2((2k+1)*theta), theta = asin(sqrt(M/N)); the resulting
measured distribution is compared to the classical answer: PASS if the
six marked states {2,3,5,7,11,13} together receive the large majority
of shots, matching the classical primality check exactly.

Dependencies: qiskit, qiskit_aer, numpy only (already installed).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_primes_below(n: int) -> list[int]:
    """Trial-division primality check, first principles, no shortcuts."""
    primes = []
    for x in range(n):
        if x < 2:
            continue
        is_prime = True
        for d in range(2, int(x ** 0.5) + 1):
            if x % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(x)
    return primes


def build_oracle(marked: list[int], n_qubits: int) -> QuantumCircuit:
    """Phase oracle flipping the sign of each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian
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
    """Standard Grover diffusion operator (inversion about the mean)."""
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


def run_grover(marked: list[int], n_qubits: int, iterations: int, shots: int = 4096):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main() -> bool:
    n = 16
    n_qubits = 4

    classical_answer = classical_primes_below(n)
    print(f"Classical property P(x) = 'x is prime' for x in [0, {n}):")
    print(f"  classical answer (trial division): {classical_answer}")
    assert classical_answer == [2, 3, 5, 7, 11, 13], "sanity check on classical primality failed"

    # Pick the iteration count (1..10) that maximizes the theoretical
    # success probability sin^2((2k+1)*theta), theta = asin(sqrt(M/N)),
    # rather than relying on the degenerate M=N/2 case where every odd
    # multiple of theta gives exactly 0.5 success probability.
    m = len(classical_answer)
    theta = np.arcsin(np.sqrt(m / n))
    best_k, best_p = 1, 0.0
    for k in range(1, 11):
        p = np.sin((2 * k + 1) * theta) ** 2
        if p > best_p:
            best_k, best_p = k, p
    iterations = best_k
    print(f"Running Grover search: N={n} states, {m} marked, {iterations} iteration(s) "
          f"(theoretical success prob {best_p:.4f})")

    counts = run_grover(classical_answer, n_qubits, iterations)
    print(f"  raw counts: {counts}")

    total_shots = sum(counts.values())
    # Qiskit count keys are printed MSB-first (qubit n-1 .. qubit 0), which
    # is the ordinary binary representation of x when qubit 0 is the LSB
    # (the same convention build_oracle uses to mark state x) -- no
    # reversal needed here.
    marked_bitstrings = {format(x, f"0{n_qubits}b") for x in classical_answer}
    marked_hits = sum(c for bstr, c in counts.items() if bstr in marked_bitstrings)
    marked_fraction = marked_hits / total_shots

    print(f"  fraction of shots landing on classically-marked primes {classical_answer}: "
          f"{marked_fraction:.3f}")

    # Success criterion: Grover amplification should concentrate the large
    # majority of shots on the marked (prime) states, matching what the
    # classical check identified as prime.
    passed = marked_fraction > 0.90

    print()
    if passed:
        print("PASS: quantum Grover search result matches classical primality check")
    else:
        print("FAIL: quantum Grover search result did not match classical primality check")

    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
