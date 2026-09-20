"""
Erdos problem #644 — quantum-testable sequence entry.

LIMITATION (read first): problem #644's entry in erdosproblems/data/problems.yaml
has oeis: ["possible"], tags: ["combinatorics"], and no further description in
the repository. "possible" is not a real OEIS sequence id (no A-number), and no
other metadata (statement text, defining recurrence, etc.) is available in the
read-only clone. There is therefore no genuine OEIS sequence for problem #644
to build a quantum circuit around, and this script does NOT fabricate one.

Best-honest-attempt fallback: since no real property of problem #644's
(nonexistent, in this data) sequence can be derived, this script instead
builds and runs a REAL, correctness-verified Grover search circuit on a small
well-defined classical/number-theoretic property (primality among 3-bit
integers, i.e. N in [0, 7]) purely as a working demonstration of the intended
methodology (Grover search verified against a classical brute-force answer).
This substitute property is NOT derived from problem #644 or from any OEIS
sequence tied to it — that connection could not be established.

Classical property actually tested: which N in {0, ..., 7} (3 qubits) are
prime. Computed from first principles (trial division) in `classical_primes()`
below. Grover's algorithm is built to amplify exactly the marked (prime)
basis states of a 3-qubit register, using an oracle built from the classical
primality set (not hard-coded from any external source), and the result is
compared against the classical answer.

ran_ok / verified_against_classical for this entry should be reported as:
ran_ok=True (the script runs and prints PASS/FAIL), but the quantum result
verifies a *substitute* classical property, not an authentic OEIS sequence
tied to Erdos problem #644, because no such id/property exists in the source
data.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_primes(n_bits: int) -> list[int]:
    """Return, by trial division from first principles, all primes < 2**n_bits."""
    limit = 2 ** n_bits
    primes = []
    for k in range(2, limit):
        is_prime = True
        for d in range(2, int(math.isqrt(k)) + 1):
            if k % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(k)
    return primes


def build_oracle(n_bits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each integer in `marked` (multi-controlled Z)."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_bits}b")[::-1]  # little-endian per qubit index
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover(n_bits: int, marked: list[int], shots: int = 4096) -> Counter:
    n_items = 2 ** n_bits
    n_marked = len(marked)
    # Optimal number of Grover iterations.
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_items / n_marked)))

    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_bits), range(n_bits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return Counter(counts)


def main() -> None:
    n_bits = 4  # search space N in [0, 15]; primes are ~37.5% of the space,
    # which gives Grover's rotation angle real room to amplify (an exact 50%
    # marked fraction, as with 3 qubits, is already at its rotation-invariant
    # fixed point and Grover cannot improve on it).
    classical_answer = sorted(classical_primes(n_bits))
    print(f"Classical primes in [0, {2**n_bits - 1}]: {classical_answer}")

    counts = run_grover(n_bits, classical_answer, shots=4096)
    total_shots = sum(counts.values())

    # Qiskit's classical bitstrings are printed MSB-first as c[n-1]...c[0],
    # which is already standard binary with qubit i worth 2**i - matching the
    # encoding used in build_oracle/build_diffuser - so a direct int() parse
    # recovers the measured integer.
    measured_ints = Counter()
    for bitstring, cnt in counts.items():
        value = int(bitstring, 2)
        measured_ints[value] += cnt

    print("Measured distribution (value: count):")
    for value, cnt in sorted(measured_ints.items(), key=lambda kv: -kv[1]):
        print(f"  {value}: {cnt}")

    marked_set = set(classical_answer)
    marked_hits = sum(cnt for v, cnt in measured_ints.items() if v in marked_set)
    marked_fraction = marked_hits / total_shots

    # Grover amplification should concentrate the large majority of shots on
    # marked (prime) outcomes, far above the uniform-random baseline.
    baseline = len(marked_set) / (2 ** n_bits)
    success = marked_fraction > 0.75 and marked_fraction > baseline * 2

    print(f"Fraction of shots landing on a prime (marked) outcome: {marked_fraction:.3f}")
    print(f"Uniform-random baseline would be: {baseline:.3f}")

    if success:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
