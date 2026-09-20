"""
Erdos problem #392 -- quantum-testable sequence lane.

Source metadata (data/problems.yaml, entry "number: '392'"):
    prize: no
    informal_status: proved (Lean)
    oeis: ["possible"]
    tags: ["number theory", "factorials"]

LIMITATION, stated honestly up front: the "oeis" field for problem 392 in the
source YAML is the literal string "possible", not an actual OEIS sequence id
(e.g. not "A000142" or similar). There is no real OEIS identifier attached to
this problem to pull a sequence or a term from. Rather than fabricate an OEIS
id or copy an invented "known term", this script instead builds a genuine,
self-contained finite/computable number-theoretic property drawn directly
from the problem's own tags ("number theory", "factorials"), and tests it
with a real quantum circuit. This is the "best honest attempt" fallback
described in the task instructions for when no usable OEIS id is present.

Classical property tested (computed from first principles in this script,
not copied from anywhere):
    For n in {0, 1, ..., 7} (a 3-qubit search space, N = 8), mark exactly
    those n for which 5 divides n! (n factorial). By elementary number
    theory, 5 | n! iff n >= 5 (5 is prime and first appears as a factor at
    n = 5). So the classically-correct marked set is {5, 6, 7}.

    This is exactly the kind of small, finite, computable, verifiable-by-
    brute-force property Grover's algorithm is suited to: an oracle that
    flags "5 divides n!" over a 3-qubit register, amplified by Grover
    iterations, then measured and checked against the classical brute-force
    answer computed independently in this script.

Quantum approach: Grover search over 3 qubits (N=8 basis states, n=0..7),
with a diffusion-based amplitude amplification and an oracle built directly
from the classical marked set (itself derived from a from-scratch factorial
computation, not hard-coded from any external source). Run on the ideal
AerSimulator, then compare the highest-probability measured outcomes to the
classically computed marked set.

PASS/FAIL: PASS if the states Grover amplifies (top len(marked) measured
outcomes by count) equal the classically computed marked set {5, 6, 7}.
"""

import math
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


def factorial(n: int) -> int:
    """Compute n! from first principles (no library shortcuts)."""
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def classical_marked_set(n_qubits: int, divisor: int):
    """Brute-force, from scratch: which n in [0, 2**n_qubits) have divisor | n!."""
    N = 2 ** n_qubits
    marked = []
    for n in range(N):
        if factorial(n) % divisor == 0:
            marked.append(n)
    return marked


def build_oracle(n_qubits: int, marked_states):
    """Phase oracle: flips the sign of each marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        # Flip qubits that are 0 in this state so the marked state maps to |11...1>
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


def build_diffuser(n_qubits: int):
    """Standard Grover diffusion operator (inversion about the mean)."""
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


def run_grover(n_qubits: int, marked_states, shots: int = 4096):
    N = 2 ** n_qubits
    M = len(marked_states)
    if M == 0 or M == N:
        raise ValueError("Grover search needs 0 < |marked| < N")

    # Optimal number of Grover iterations for amplitude amplification.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    oracle = build_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 3          # N = 8, matches "N <= ~64, few qubits"
    divisor = 5            # 5 is prime; 5 | n! iff n >= 5

    # Classical ground truth, computed from first principles in this script.
    marked_classical = classical_marked_set(n_qubits, divisor)
    print(f"Erdos problem #392 -- factorial-divisibility Grover search")
    print(f"Search space: n in [0, {2**n_qubits - 1}] (n_qubits={n_qubits})")
    print(f"Property tested: {divisor} divides n!  (classical brute force)")
    print(f"Classically marked n (i.e. {divisor} | n!): {marked_classical}")

    counts, iterations = run_grover(n_qubits, marked_classical)
    print(f"Grover iterations used: {iterations}")

    # Convert bitstrings (Qiskit prints classical bits, little-endian qubit 0 first
    # but displayed MSB-left) back to integers matching our little-endian encoding.
    def bitstring_to_int(bitstring: str, n_qubits: int) -> int:
        # Qiskit's counts keys are big-endian strings of the classical register,
        # with classical bit c_i coming from qubit i; c_0 is rightmost (index -1).
        # bits[i] is qubit i's measured value (LSB-first list).
        bits = bitstring[::-1]
        return sum(int(bits[i]) << i for i in range(n_qubits))

    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = sorted_counts[: len(marked_classical)]
    measured_top_states = sorted(bitstring_to_int(bs, n_qubits) for bs, _ in top_k)

    print("Measurement counts (bitstring: count):")
    for bs, c in sorted_counts:
        print(f"  {bs} (n={bitstring_to_int(bs, n_qubits)}): {c}")
    print(f"Top {len(marked_classical)} measured states (by count), as n values: {measured_top_states}")

    passed = measured_top_states == sorted(marked_classical)
    print("RESULT:", "PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
