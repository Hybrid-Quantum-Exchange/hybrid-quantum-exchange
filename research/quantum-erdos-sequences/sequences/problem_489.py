"""
Erdos problem #489 -- quantum-testable sequence lane.

LIMITATION (read this first): in the source data
(/home/user/manman4/erdosproblems/data/problems.yaml, entry "number: \"489\""),
problem #489 has oeis: ["N/A"] -- it carries no OEIS sequence id at all. Its only
other metadata is tags: ["number theory"], prize: "no", and status "open"
(unformalized). There is therefore no real OEIS sequence for this lane to test
membership/terms of, and this script does NOT fabricate one or borrow an
unrelated OEIS id and pretend it belongs to #489.

Honest best-effort substitute: since the one piece of real content problem #489
gives us is the tag "number theory", this script instead builds a genuine,
finite, classically-checkable number-theory search problem -- finding the
PERFECT NUMBERS (n such that the sum of n's proper divisors equals n) in the
range [1, 32] -- and solves it with a real Grover search circuit on
AerSimulator. This is presented as exactly what it is: a stand-in exercise in
the spirit of the tag, NOT a derivation of problem #489's actual (nonexistent)
sequence. verified_against_classical below refers only to this stand-in
search, not to problem #489 itself, which remains untestable in this form.

Classical property under test
------------------------------
For n in [0, 31] (5 qubits, N = 32 basis states), define
    sigma_proper(n) = sum of divisors d of n with 1 <= d < n   (n >= 1; sigma_proper(0) := 0)
n is a PERFECT NUMBER iff sigma_proper(n) == n and n > 0.
Computed classically below (first principles, no OEIS lookup): among 0..31 the
perfect numbers are {6, 28} (n=1..5 have no proper divisors summing to n;
n=6: 1+2+3=6; n=28: 1+2+4+7+14=28). This matches OEIS A000396 (6, 28, 496, ...)
but that identification is NOT used anywhere in the code -- the marked set is
derived purely by the classical sigma_proper() function below.

Quantum approach
-----------------
Grover's algorithm over 5 qubits (search space size N=32, 2 marked items).
The oracle is built directly from the classically-computed marked set (a
standard, honest way to instantiate a black-box Grover oracle for a property
whose "hard part" is the classical search/verification, not arithmetic
circuit synthesis) via a multi-controlled Z gate on each marked basis state.
Optimal number of Grover iterations for M=2 marked out of N=32 is
round(pi/4 * sqrt(N/M)) = round(pi/4 * sqrt(16)) = round(pi) = 3.

The circuit is run on the ideal AerSimulator with many shots; PASS requires
that essentially all measured outcomes land on the classically-computed
marked set {6, 28}, with the two marked outcomes as the top two most frequent
results and negligible probability elsewhere.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def sigma_proper(n: int) -> int:
    """Sum of proper divisors of n (divisors d with 1 <= d < n). sigma_proper(0) = 0."""
    if n <= 0:
        return 0
    total = 0
    for d in range(1, n):
        if n % d == 0:
            total += d
    return total


def classical_perfect_numbers(limit: int) -> list:
    """All n in [0, limit) with sigma_proper(n) == n and n > 0, computed from first principles."""
    return [n for n in range(limit) if n > 0 and sigma_proper(n) == n]


def build_oracle(num_qubits: int, marked_states: list) -> QuantumCircuit:
    """Phase-flip oracle: applies -1 phase to each marked computational basis state."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_grover_circuit(num_qubits: int, marked_states: list, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    oracle = build_oracle(num_qubits, marked_states)
    diffuser = build_diffuser(num_qubits)
    for _ in range(iterations):
        qc.compose(oracle, range(num_qubits), inplace=True)
        qc.compose(diffuser, range(num_qubits), inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def main() -> bool:
    num_qubits = 5
    N = 2 ** num_qubits  # 32

    marked_states = classical_perfect_numbers(N)
    print(f"Classical search space: n in [0, {N - 1}]")
    print(f"Classically computed perfect numbers in range: {marked_states}")
    expected = [6, 28]
    if marked_states != expected:
        print(f"FAIL: classical computation disagrees with expected {expected}")
        return False

    M = len(marked_states)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
    print(f"Grover iterations used: {iterations} (N={N}, M={M})")

    qc = build_grover_circuit(num_qubits, marked_states, iterations)

    sim = AerSimulator()
    shots = 4096
    job = sim.run(qc, shots=shots)
    counts = job.result().get_counts()

    # Convert bitstrings (Qiskit: little-endian, qubit0 rightmost) back to integers.
    int_counts = Counter()
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        int_counts[n] += c

    print(f"Top measured outcomes: {int_counts.most_common(5)}")

    top_two = {n for n, _ in int_counts.most_common(M)}
    marked_set = set(marked_states)
    marked_prob = sum(int_counts[n] for n in marked_set) / shots

    print(f"Set of marked states: {marked_set}")
    print(f"Top-{M} measured states: {top_two}")
    print(f"Probability mass on marked states: {marked_prob:.4f}")

    success = (top_two == marked_set) and (marked_prob > 0.90)

    if success:
        print("PASS")
    else:
        print("FAIL")
    return success


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
