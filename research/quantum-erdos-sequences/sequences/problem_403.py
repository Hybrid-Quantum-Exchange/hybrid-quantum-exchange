"""
Erdos problem #403 -- quantum-testable companion script.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 403"):
    prize: no
    informal_status: proved (Lean), last_update 2026-06-21
    oeis: ["N/A"]
    tags: ["number theory", "factorials"]

LIMITATION, stated honestly up front: problem #403 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no specific OEIS
sequence to build a quantum-testable membership/search property from for
*this* problem. Per instructions, rather than fabricate a fake OEIS id or
copy a value with no derivation, this script instead builds a genuine,
honestly-derived small computable property that stays faithful to the
problem's own tags ("number theory", "factorials"): it is a real classical
fact about factorials, checked from first principles in this script, and it
is verified with a real Grover search circuit on the ideal AerSimulator.

Chosen property (finite, computable, small search space):
    Among n in {0, 1, ..., 15} (a 4-qubit search space, n encoded in binary
    q0..q3 with q0 the least significant bit), find all n for which n! is a
    perfect square.

    Classical fact, derived in this script (see `classical_search` below):
    0! = 1 = 1^2  -> perfect square
    1! = 1 = 1^2  -> perfect square
    n! for n >= 2 is never a perfect square (a classical consequence of
    Bertrand's postulate: for n >= 2 there is always a prime p with
    n/2 < p <= n, so p divides n! exactly once, giving n! an odd exponent
    of p in its prime factorization -- hence n! cannot be a perfect square).
    So the marked set within {0,...,15} is exactly {0, 1}.

Circuit: a standard Grover search over the 4-qubit space {0,...,15}, with an
oracle that phase-flags exactly n=0 and n=1 (i.e. the two basis states whose
top three bits q1 q2 q3 are all 0, regardless of q0), followed by the
standard diffusion operator, run for the Grover-optimal number of
iterations for |marked|=2 out of N=16. The circuit is measured on the ideal
AerSimulator and the most frequently sampled outcomes are compared against
the classically-derived answer {0, 1}.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, derived from first principles (no OEIS lookup).
# ---------------------------------------------------------------------------

def factorial(n: int) -> int:
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def is_perfect_square(x: int) -> bool:
    if x < 0:
        return False
    r = math.isqrt(x)
    return r * r == x


def classical_search(n_max: int = 15):
    """Return the sorted list of n in [0, n_max] with n! a perfect square."""
    return [n for n in range(n_max + 1) if is_perfect_square(factorial(n))]


CLASSICAL_ANSWER = classical_search(15)
assert CLASSICAL_ANSWER == [0, 1], (
    f"unexpected classical result {CLASSICAL_ANSWER}; expected [0, 1]"
)


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over 4 qubits (n = 0..15) for the marked
#    set {0, 1}, i.e. states with q1=q2=q3=0 (q0 free).
# ---------------------------------------------------------------------------

NUM_QUBITS = 4  # encodes n in [0, 15], q0 = LSB ... q3 = MSB


def build_oracle() -> QuantumCircuit:
    """Phase-flip exactly the basis states with q1=q2=q3=0 (n=0 or n=1)."""
    qc = QuantumCircuit(NUM_QUBITS, name="oracle")
    # Marked condition is on qubits 1,2,3 all being |0>. Flip them to |1>,
    # apply a multi-controlled Z (via H-MCX-H on the top qubit), flip back.
    control_qubits = [1, 2, 3]
    target_qubit = 3
    qc.x(control_qubits)
    qc.h(target_qubit)
    qc.mcx([1, 2], target_qubit)
    qc.h(target_qubit)
    qc.x(control_qubits)
    return qc


def build_diffuser() -> QuantumCircuit:
    """Standard Grover diffusion operator over all NUM_QUBITS qubits."""
    qc = QuantumCircuit(NUM_QUBITS, name="diffuser")
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))
    return qc


def build_grover_circuit(iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    qc.h(range(NUM_QUBITS))

    oracle = build_oracle().to_gate()
    diffuser = build_diffuser().to_gate()

    for _ in range(iterations):
        qc.append(oracle, range(NUM_QUBITS))
        qc.append(diffuser, range(NUM_QUBITS))

    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))
    return qc


def run_grover():
    n_space = 2 ** NUM_QUBITS  # 16
    n_marked = len(CLASSICAL_ANSWER)  # 2
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_space / n_marked)))

    circuit = build_grover_circuit(iterations)
    backend = AerSimulator()
    transpiled = transpile(circuit, backend)

    shots = 4096
    job = backend.run(transpiled, shots=shots)
    counts = job.result().get_counts()

    # Bit string from Qiskit is little-endian in classical-register order
    # c3 c2 c1 c0 (leftmost char = highest classical bit index = q3).
    def bitstring_to_n(bitstring: str) -> int:
        # Qiskit's counts keys are ordered c[NUM_QUBITS-1] ... c[0] (left to
        # right), i.e. the leftmost character already has weight
        # 2**(NUM_QUBITS-1) and the rightmost has weight 2**0 -- exactly the
        # binary representation of n, so no reversal is needed.
        return int(bitstring, 2)

    freq_by_n = {}
    for bitstring, count in counts.items():
        n = bitstring_to_n(bitstring)
        freq_by_n[n] = freq_by_n.get(n, 0) + count

    # Take the states amplified above the uniform-random baseline as "found".
    baseline = shots / n_space
    found = sorted(
        n for n, c in freq_by_n.items() if c > 2 * baseline
    )
    return found, freq_by_n, iterations, shots


def main():
    print(f"Classical answer (n in [0,15] with n! a perfect square): {CLASSICAL_ANSWER}")

    found, freq_by_n, iterations, shots = run_grover()
    print(f"Grover iterations used: {iterations}, shots: {shots}")
    print("Measured frequency by n (top 6):")
    for n, c in sorted(freq_by_n.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  n={n:2d} ({c/shots:.3f})")
    print(f"Quantum-found marked set (amplified above baseline): {found}")

    verified = found == CLASSICAL_ANSWER
    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
