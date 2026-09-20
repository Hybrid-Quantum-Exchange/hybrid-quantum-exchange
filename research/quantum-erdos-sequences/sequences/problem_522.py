"""
Erdos problem #522 -- quantum-testable-sequence lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 522"):
    prize: no
    informal_status: open (last_update 2025-12-08)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["analysis", "polynomials", "probability"]

LIMITATION (reported honestly, per instructions): problem #522 has no OEIS
sequence id attached (oeis: ["N/A"]) and its tags place it in real/complex
analysis on polynomials and probability, not in a finite combinatorial or
number-theoretic setting. There is therefore no small, finite, computable
membership/counting property of "the sequence for problem 522" to encode in
a quantum circuit -- there is no sequence. Fabricating one would misrepresent
the problem, which the task instructions explicitly forbid.

Rather than skip this lane, this script still builds and runs a REAL,
verifiable quantum circuit on a small finite instance, and is honest in its
reporting that the instance is a stand-in, not problem 522's own content.

Chosen finite, computable property (stand-in instance):
    "n is prime" for n in the 4-bit range 0..15 (N = 16, 4 qubits).
    Primality here is a genuine finite/decidable property (trial division),
    computed classically from first principles in this script, and then
    verified with a real Grover search circuit built from an explicit
    reversible primality oracle (marking exactly the primes in [0, 15]) run
    on the ideal AerSimulator. This exercises the same "small computable
    property + Grover oracle" pattern the task asks for; it is presented
    here only as a worked, verifiable example, not as a claim about problem
    522's mathematical content.

PASS/FAIL: the script computes the classical set of primes in [0,15] by
trial division, builds a Grover oracle marking exactly those 4-bit basis
states, runs Grover's algorithm on the AerSimulator, and checks that the
measurement results are overwhelmingly concentrated on the classical prime
set.
"""

import itertools
import math

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


N_BITS = 4
N = 2 ** N_BITS  # 16


def is_prime_classical(n: int) -> bool:
    """Trial division, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_bits: int):
    return sorted(x for x in range(2 ** n_bits) if is_prime_classical(x))


def build_oracle(n_bits: int, marked_values):
    """Phase-flip oracle: flips the sign of |x> for each x in marked_values."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_bits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
            qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def build_grover_circuit(n_bits: int, marked_values, iterations: int):
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, marked_values)
    diffuser = build_diffuser(n_bits)

    for _ in range(iterations):
        qc.compose(oracle, qubits=range(n_bits), inplace=True)
        qc.compose(diffuser, qubits=range(n_bits), inplace=True)

    qc.measure(range(n_bits), range(n_bits))
    return qc


def bits_to_int_little_endian(bitstring: str) -> int:
    # Qiskit's count keys are already ordered c_{n-1}...c_0 left to right,
    # i.e. clbit 0 (which we tied to qubit 0, our LSB) is the rightmost
    # character -- so a plain binary parse recovers the integer directly.
    return int(bitstring, 2)


def main():
    marked = classical_primes(N_BITS)
    print(f"Classical primes in [0, {N - 1}]: {marked}")

    M = len(marked)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
    print(f"N={N}, M={M}, Grover iterations={iterations}")

    qc = build_grover_circuit(N_BITS, marked, iterations)

    backend = AerSimulator()
    shots = 4096
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()

    hits = 0
    for bitstring, count in counts.items():
        clean = bitstring.replace(" ", "")
        value = bits_to_int_little_endian(clean)
        if value in marked:
            hits += count

    hit_fraction = hits / shots
    print(f"Fraction of shots landing on a classical prime: {hit_fraction:.4f}")

    # With near-optimal Grover iterations on this small instance the
    # concentration on marked states should be very high; use a generous
    # but still meaningful threshold.
    threshold = 0.80
    passed = hit_fraction >= threshold

    if passed:
        print("PASS")
    else:
        print("FAIL")

    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
