"""
Erdos problem #452 -- quantum-testable fallback.

Source metadata (erdosproblems.com data, data/problems.yaml, entry "number: \"452\""):
    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION (read before trusting the "PASS"):
    The `oeis` field for problem 452 is the literal string "possible", which is
    not an OEIS sequence identifier (no A-number). There is no sequence
    attached to this problem in the source data to build a quantum-testable
    property from, and no further description of the problem's statement is
    available in this read-only clone. Per the task's own instructions, this
    is therefore the documented "no genuine tie to the target problem's
    sequence" case: rather than fabricate an OEIS id or invent unfounded
    content for problem 452, this script implements a genuine, honestly
    unrelated-to-452 number-theory instance -- Grover search for the primes
    below 16 -- so that a real quantum circuit with verifiable classical
    ground truth still lives at this path. The classical property tested
    (primality of the integers 0..15) is real, is computed from first
    principles in this script (trial division), and is checked against the
    quantum result; it is simply not derived from problem 452's own content,
    because problem 452 has none usable here.

Classical property under test:
    For N = 16 (4 qubits, search space {0, ..., 15}), the set of prime
    integers is computed by trial division: P = {2, 3, 5, 7, 11, 13}.

Quantum method:
    Grover's algorithm on 4 qubits. The oracle is built directly from the
    classically-computed prime set P (a phase oracle that flips the sign of
    each basis state |x> with x in P). ceil(pi/4 * sqrt(2^4/|P|)) Grover
    iterations are applied on the ideal AerSimulator, then the register is
    measured. PASS means the measurement distribution is concentrated on
    exactly the classically-computed prime set P (i.e. quantum amplitude
    amplification of a classically-verified, non-fabricated set), which is a
    genuine (if problem-452-independent) computation, not a copied literal.

ran_ok / verified_against_classical are reported honestly in the printed
output and are not faked.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_primes_below(n: int) -> list[int]:
    """Trial-division primality test, computed from first principles."""
    primes = []
    for x in range(n):
        if x < 2:
            continue
        is_prime = True
        for d in range(2, int(math.isqrt(x)) + 1):
            if x % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(x)
    return primes


def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase oracle flipping the sign of each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for x in marked:
        bits = format(x, f"0{n_qubits}b")[::-1]  # little-endian
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
    """Standard Grover diffuser (inversion about the mean)."""
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


def run_grover(n_qubits: int, marked: list[int], shots: int = 2000) -> Counter:
    n_states = 2 ** n_qubits
    ideal = (math.pi / 4) * math.sqrt(n_states / len(marked))
    candidates = [max(1, math.floor(ideal)), max(1, round(ideal)), max(1, math.ceil(ideal))]
    # Pick the iteration count that maximizes the theoretical success
    # probability sin^2((2k+1) * theta), theta = arcsin(sqrt(|marked|/n_states)).
    theta = math.asin(math.sqrt(len(marked) / n_states))
    iterations = max(set(candidates), key=lambda k: math.sin((2 * k + 1) * theta) ** 2)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports bit strings MSB-first over the classical register order,
    # with qubit 0 as the rightmost character (little-endian), matching the
    # oracle's own little-endian convention above.
    decoded = Counter()
    for bitstring, count in counts.items():
        value = int(bitstring, 2)
        decoded[value] += count
    return decoded


def main():
    n_qubits = 4
    n_states = 2 ** n_qubits

    classical = sorted(classical_primes_below(n_states))
    print(f"Classical primes below {n_states} (trial division): {classical}")

    counts = run_grover(n_qubits, classical, shots=2000)
    total_shots = sum(counts.values())

    print("Measurement distribution (value: count):")
    for value in sorted(counts, key=lambda v: -counts[v]):
        print(f"  {value:2d} ({'prime' if value in classical else 'non-prime':>9s}): {counts[value]}")

    hits_on_primes = sum(c for v, c in counts.items() if v in classical)
    fraction_on_primes = hits_on_primes / total_shots

    # Success criterion: amplitude amplification concentrated the vast
    # majority of measurement outcomes on the classically-verified prime set.
    # With |marked|=6 out of 16 states the achievable Grover success
    # probability peaks at sin^2(3*theta) ~= 0.844 for the optimal (single)
    # iteration count -- there is no larger number of iterations that does
    # better for this ratio, so the bar is set just below that ceiling.
    threshold = 0.80
    verified = fraction_on_primes >= threshold

    print(f"\nFraction of shots landing on a classically-verified prime: "
          f"{fraction_on_primes:.4f} (threshold {threshold})")

    ran_ok = True
    if verified:
        print("PASS")
    else:
        print("FAIL")

    return ran_ok, verified


if __name__ == "__main__":
    main()
