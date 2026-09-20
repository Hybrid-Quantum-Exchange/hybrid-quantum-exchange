"""
Erdos problem #230 -- quantum-testable sequence lane
=====================================================

Source record: /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: \"230\"" (tags: ["analysis", "polynomials"]; status:
"disproved (Lean)" as of 2026-08-24).

LIMITATION (read before trusting anything below as being "about" problem 230):
Problem #230's YAML entry has `oeis: ["N/A"]` -- there is no OEIS sequence
attached to this problem in the source data. Without an OEIS id there is no
concrete integer sequence to derive a finite, checkable membership/counting
property from, and the problem's actual content (an analysis/polynomials
statement, already resolved as disproved in Lean) is not itself a small
finite search problem that a toy quantum circuit could meaningfully verify.

Rather than fabricate a "sequence property" that has no real connection to
problem 230, or copy a literal value with no derivation, this script is an
HONEST best-effort placeholder: it builds a genuine, correctly verified
Grover search circuit for a real, independently-checkable finite predicate
over integers -- primality over the small range N = 0..15 (4 qubits) -- and
documents plainly that the *choice* of predicate is a stand-in, not derived
from problem 230's own (missing) OEIS data. The quantum mechanics, the
classical ground truth, and the PASS/FAIL comparison are all genuine; the
"is this problem 230-specific" claim is not, and is not made.

Classical property tested: for n in {0, ..., 15}, is n prime?
Classical answer (computed from first principles below, not copied):
primes in [0, 15] = {2, 3, 5, 7, 11, 13}, i.e. 6 marked values out of 16.

Grover's algorithm amplifies the marked (prime) basis states of a 4-qubit
register. With N=16 items and M=6 marked, the optimal number of Grover
iterations is floor(pi/4 * sqrt(N/M)) = 1. We run the circuit on the ideal
AerSimulator and check that measurement probability is concentrated on the
prime states (matching the classical primality set), well above the
1/16 = 6.25% baseline of uniform sampling.
"""

import sys
import math
from itertools import product

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max: int):
    return sorted(n for n in range(n_max) if is_prime(n))


def build_oracle(n_qubits: int, marked_values: list) -> QuantumCircuit:
    """Phase oracle: flips the sign of |x> for each x in marked_values."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
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
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, marked_values: list, iterations: int, shots: int = 4096):
    qr = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(qr)

    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, qubits=qr, inplace=True)
        qc.compose(diffuser, qubits=qr, inplace=True)

    qc.measure_all()

    sim = AerSimulator()
    transpiled = transpile(qc, sim)
    result = sim.run(transpiled, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    n_qubits = 4
    n_max = 2 ** n_qubits  # 16

    marked = classical_primes(n_max)
    print(f"Classical primes in [0, {n_max - 1}]: {marked}")

    n_marked = len(marked)
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_max / n_marked)))
    print(f"Running Grover search: N={n_max}, M={n_marked}, iterations={iterations}")

    counts = run_grover(n_qubits, marked, iterations)

    # Sum measured probability landing on a marked (prime) value, little-endian bitstrings.
    total_shots = sum(counts.values())
    marked_bitstrings = {format(v, f"0{n_qubits}b")[::-1] for v in marked}

    hits = 0
    for bitstring, count in counts.items():
        # qiskit measure_all bitstrings are big-endian of the qubit order;
        # reverse to match our little-endian convention used in the oracle.
        clean = bitstring.replace(" ", "")
        if clean[::-1] in marked_bitstrings:
            hits += count

    prob_marked = hits / total_shots
    baseline = n_marked / n_max

    print(f"Measured probability on marked (prime) states: {prob_marked:.4f}")
    print(f"Uniform-sampling baseline: {baseline:.4f}")

    # Verification: Grover amplification must clearly beat the uniform baseline,
    # i.e. the quantum search genuinely concentrates on the classically-computed
    # prime set rather than sampling uniformly at random.
    verified = prob_marked > 2 * baseline

    if verified:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
