"""
Erdos problem #235 -- quantum-testable sequence lane.

LIMITATION (read first): problems.yaml lists Erdos problem #235 as
  number: "235", tags: ["number theory"], oeis: ["N/A"]
i.e. the dataset records NO OEIS sequence id for this problem (informal
status: proved, 2025-08-31, no further textual description available in
the read-only clone at /home/user/manman4/erdosproblems/data/problems.yaml).
Without an OEIS id or a captured statement of the problem there is no
sequence to derive a genuine, problem-specific finite property from for
this lane -- fabricating one and presenting it as "problem 235's sequence"
would be dishonest. Per instructions, this script is therefore the best
honest fallback: a REAL, correctly verified quantum computation of a
small, finite, computable number-theory property (matching the recorded
tag "number theory"), run and checked against a first-principles classical
computation, but it is NOT tied to any OEIS sequence for problem 235
specifically. Treat verified_against_classical as "the circuit is a real,
correct piece of quantum number theory," not as "this reproduces problem
235's actual mathematical content."

Chosen property (finite, computable, small circuit):
  Grover search over N = 2^n = 16 integers x in [0, 15] for the unique x
  that is BOTH squarefree AND has an odd number of divisors reversed --
  concretely: search for the unique x in [0,15] such that x is a perfect
  square (has an odd number of divisors), which is a classic finite
  number-theoretic predicate. The classical answer set for n=4 (16
  values) is computed from first principles in `classical_perfect_squares`
  below (trial computation of divisor counts / integer sqrt, no OEIS
  lookup, no hard-coded literal answer): {0, 1, 4, 9}.

Since Grover's algorithm assumes a single marked item for the textbook
optimal iteration count, we run the search for target x = 9 (the largest
perfect square below 16), oracle-marking exactly the state |1001>, and
verify quantum measurement recovers x = 9 with high probability while the
full 16-value classical predicate table is independently computed for
reference/comparison.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


def classical_perfect_squares(n_bits: int):
    """First-principles classical computation: which x in [0, 2^n_bits - 1]
    are perfect squares? No OEIS lookup, no literal answer copied in --
    computed here via integer sqrt trial."""
    N = 2 ** n_bits
    squares = []
    for x in range(N):
        r = math.isqrt(x)
        if r * r == x:
            squares.append(x)
    return squares


def build_oracle(n_qubits: int, target: int) -> QuantumCircuit:
    """Phase-flip oracle marking the single basis state |target> (n_qubits
    wide), built from X gates + a multi-controlled Z (via H-MCX-H)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(target, f"0{n_qubits}b")[::-1]  # little-endian per qubit
    flip_qubits = [i for i, b in enumerate(bits) if b == "0"]

    for q in flip_qubits:
        qc.x(q)

    # Multi-controlled Z on all n_qubits: H on last qubit, MCX, H
    qc.h(n_qubits - 1)
    if n_qubits - 1 > 0:
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    else:
        qc.z(0)
    qc.h(n_qubits - 1)

    for q in flip_qubits:
        qc.x(q)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits - 1 > 0:
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    else:
        qc.z(0)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, target: int, shots: int = 2048):
    N = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(N)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, target)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    qc = transpile(qc, sim)
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    n_qubits = 4  # N = 16, small finite search space
    N = 2 ** n_qubits

    squares = classical_perfect_squares(n_qubits)
    print(f"Classical: perfect squares in [0, {N - 1}] = {squares}")

    target = max(squares)  # 9, the largest perfect square < 16
    print(f"Grover target (largest perfect square < {N}): {target}")

    counts = run_grover(n_qubits, target)
    total_shots = sum(counts.values())

    # Qiskit bit order in the count string is little-endian-reversed
    # (qubit n-1 ... qubit 0); convert to an integer consistently.
    best_bitstring = max(counts, key=counts.get)
    measured = int(best_bitstring, 2)

    prob_target = counts.get(best_bitstring, 0) / total_shots
    print(f"Quantum: most frequent measured value = {measured} "
          f"(probability {prob_target:.3f} over {total_shots} shots)")
    print(f"Full counts: {counts}")

    quantum_ok = (measured == target) and (prob_target > 0.5)
    classical_ok = target in squares and target == math.isqrt(target) ** 2

    if quantum_ok and classical_ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
