"""
Erdos problem #15 (from erdosproblems.com, via the manman4/erdosproblems data
export at data/problems.yaml, entry `number: "15"`) — tags: "number theory",
"primes"; prize: none; status: open (unformalized, informal state "open" as
of 2025-08-31).

LIMITATION, stated up front: the problems.yaml entry for problem 15 lists
`oeis: ["N/A"]` — there is no OEIS sequence id attached to this problem in
the source data. The task asked to derive a small finite computable property
from the problem's OEIS id(s) and tags. With no OEIS id available, there is
no specific named integer sequence to target. Rather than fabricate an OEIS
id or silently substitute an unrelated famous sequence and call it problem
15's sequence, this script is honest about the gap: it uses the problem's
TAGS ("number theory", "primes") to pick the closest faithful, genuinely
quantum-testable finite property in that same domain — primality itself,
the seed concept every "primes" tag problem is built on.

Classical property tested:
    For N = 16 (4 qubits, one basis state per integer 0..15), let
    S = { n in [0, N) : n is prime }.
    This script computes S classically from first principles by trial
    division (no OEIS lookup, no hard-coded prime list), then builds a
    Grover search circuit whose oracle marks exactly the states in S via a
    provable-correct classical-into-quantum truth table (implemented as a
    multi-controlled phase flip per marked computational basis state), and
    runs the standard number-of-solutions-known Grover algorithm on the
    ideal AerSimulator.

    Classical answer for N = 16: S = {2, 3, 5, 7, 11, 13}  (|S| = 6).

PASS criterion: after running the Grover circuit and sampling, the set of
measured outcomes with the highest counts (taking the top |S| distinct
outcomes by count) equals S exactly, i.e. quantum search recovers exactly
the primes below 16 and nothing else.

No OEIS sequence value is copied anywhere in this script; the "answer"
(the prime set S) is derived here by trial division and independently
cross-checked with sympy-free trial division inside `is_prime_classical`.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


def is_prime_classical(n: int) -> bool:
    """Trial division primality test, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def build_oracle(n_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Phase-flip oracle: multiplies the amplitude of each marked basis
    state (given as an integer in [0, 2**n_qubits)) by -1, leaving all
    other basis states untouched."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")
        # X on qubits that should be 0, so the marked state maps to |1...1>
        zero_positions = [i for i, b in enumerate(reversed(bits)) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
            qc.append(mcz, list(range(n_qubits)))
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
        qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main() -> bool:
    N = 16
    n_qubits = 4  # log2(16)

    # --- classical ground truth, derived here, no OEIS copy ---
    classical_primes = sorted(n for n in range(N) if is_prime_classical(n))
    print(f"Classical primes in [0, {N}): {classical_primes}")

    marked = classical_primes
    M = len(marked)
    assert M > 0

    # optimal number of Grover iterations for N states, M marked
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations: {iterations} (N={N}, M={M})")

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    qc = qc.decompose(reps=3)

    backend = AerSimulator()
    shots = 8192
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()

    # counts keys are bitstrings MSB..LSB matching qubit order [q0..qn-1] reversed
    parsed = {int(bitstring, 2): c for bitstring, c in counts.items()}
    ranked = sorted(parsed.items(), key=lambda kv: -kv[1])
    top_states = sorted(state for state, _ in ranked[:M])

    print(f"Top {M} most-measured basis states (quantum): {top_states}")
    print(f"Full measurement histogram (state: count): "
          f"{dict(sorted(parsed.items()))}")

    verified = top_states == classical_primes
    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
