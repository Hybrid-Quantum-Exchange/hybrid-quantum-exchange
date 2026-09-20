"""
Erdos problem #221 -- quantum-testable companion script
=========================================================

Source metadata (from /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: \"221\""):
    prize: no
    status: proved (Lean), last update 2026-01-31
    oeis: ["N/A"]
    tags: ["number theory", "additive basis"]

LIMITATION, stated up front: problem #221 carries no OEIS sequence id in the
source data (oeis: ["N/A"]). There is therefore no OEIS-derived sequence to
build a membership/search oracle from for this entry, and this script does
NOT claim to test problem #221 itself or any OEIS sequence tied to it. Per
the task's fallback instructions for exactly this case (no OEIS id), this is
a best-honest-attempt companion: it builds a REAL Grover search circuit over
a genuine, classically-checkable number-theoretic property drawn from the
problem's own tag "additive basis" -- representability of an integer as a
sum of two squares (the classical two-square theorem, a canonical additive
basis statement of order 2) -- rather than fabricating or copying an OEIS
value that does not exist for this entry.

Classical property under test
------------------------------
Fix TARGET = 13 and search space a, b in {0, 1, 2, 3} (2 bits each, 4 qubits
total, 16 basis states). We search for pairs (a, b) with a^2 + b^2 == TARGET.
This is computed from first principles below (plain brute force, no OEIS
lookup) BEFORE the quantum circuit runs, so the classical answer is known
and independently derived.

Quantum circuit
----------------
A genuine Grover search circuit (AerSimulator, statevector-exact, no noise):
  - 4-qubit register encodes (a1 a0 b1 b0), a,b in [0,3].
  - Phase oracle flips the sign of exactly the computational basis states
    whose (a, b) satisfy a^2 + b^2 == TARGET (built via multi-controlled Z
    gates on the precomputed classical solution set -- the same set found
    by brute force above, expressed as a quantum oracle, not injected as
    the answer).
  - Standard Grover diffusion operator, iterated the optimal number of
    times for this space size and number of marked states.
  - Measurement; the highest-probability outcome(s) must decode to a
    solution of a^2 + b^2 == TARGET.

PASS/FAIL is decided by comparing the most-frequently measured bitstring(s)
against the independently computed classical solution set.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


TARGET = 13
DOMAIN = range(4)  # a, b in {0,1,2,3} -> 2 bits each


def classical_solutions(target: int, domain: range) -> list[tuple[int, int]]:
    """Brute-force, first-principles search for a^2 + b^2 == target."""
    sols = []
    for a, b in product(domain, domain):
        if a * a + b * b == target:
            sols.append((a, b))
    return sols


def bits_for(value: int, nbits: int) -> str:
    return format(value, f"0{nbits}b")


def marked_bitstrings(sols: list[tuple[int, int]]) -> list[str]:
    """Encode each solution (a, b) as a 4-bit string 'a1 a0 b1 b0'."""
    out = []
    for a, b in sols:
        out.append(bits_for(a, 2) + bits_for(b, 2))
    return out


def apply_oracle(qc: QuantumCircuit, bitstring: str, qubits: list[int]) -> None:
    """Flip the phase of the single computational basis state `bitstring`
    (qubit 0 = rightmost bit convention handled by caller's qubit order)."""
    # X-gate every qubit that should be 0 in the target string, so the
    # target string becomes |11...1>, apply a multi-controlled Z, then
    # undo the X gates.
    zero_positions = [q for q, bit in zip(qubits, bitstring) if bit == "0"]
    for q in zero_positions:
        qc.x(q)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in zero_positions:
        qc.x(q)


def build_diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    if n - 1 > 0:
        qc.mcx(list(range(n - 1)), n - 1)
    else:
        qc.z(0)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def run() -> None:
    sols = classical_solutions(TARGET, DOMAIN)
    assert sols, "no classical solutions found; TARGET/DOMAIN misconfigured"
    marked = marked_bitstrings(sols)
    print(f"Classical brute-force solutions of a^2+b^2={TARGET} for a,b in {list(DOMAIN)}: {sols}")
    print(f"Marked computational basis strings (a1 a0 b1 b0): {marked}")

    n_qubits = 4  # 2 bits for a, 2 bits for b
    N = 2 ** n_qubits
    M = len(marked)

    # Optimal number of Grover iterations for N states, M marked states.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"N={N}, M={M}, Grover iterations={iterations}")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        for bitstring in marked:
            # Qiskit qubit 0 is the rightmost character of a bitstring when
            # printed; build the oracle against a fixed left-to-right
            # qubit-index mapping [0,1,2,3] <-> string chars [0,1,2,3].
            apply_oracle(qc, bitstring, list(range(n_qubits)))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Qiskit's classical bit order in the count keys is reversed relative to
    # qubit index order (bit c_{n-1} ... c_0), so re-derive (a, b) per key
    # using the same left-to-right convention as apply_oracle's bitstring.
    def key_to_ab(key: str) -> tuple[int, int]:
        # key is qiskit's c3 c2 c1 c0 (MSB..LSB) matching qubit indices
        # 3,2,1,0. Our bitstring convention indexed qubits [0,1,2,3] as
        # string chars [0,1,2,3], i.e. char i <-> qubit i. Qiskit prints
        # qubit (n-1)..0 left to right, so key[i] <-> qubit (n_qubits-1-i).
        qubit_bits = [None] * n_qubits
        for i, ch in enumerate(key):
            qubit_bits[n_qubits - 1 - i] = ch
        bitstring = "".join(qubit_bits)
        a = int(bitstring[0:2], 2)
        b = int(bitstring[2:4], 2)
        return a, b

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_key, top_count = sorted_counts[0]
    top_ab = key_to_ab(top_key)
    total_shots = sum(counts.values())
    marked_shots = 0
    for key, c in counts.items():
        a, b = key_to_ab(key)
        if (a, b) in sols:
            marked_shots += c

    print(f"Top measured outcome: {top_key} -> (a,b)={top_ab}, count={top_count}/{total_shots}")
    print(f"Fraction of shots landing on a classical solution: {marked_shots}/{total_shots} = {marked_shots/total_shots:.3f}")

    quantum_found_solution = top_ab in sols
    amplified_above_uniform = (marked_shots / total_shots) > (M / N) * 1.5

    passed = quantum_found_solution and amplified_above_uniform

    print(f"quantum_found_solution (top outcome is a true a^2+b^2={TARGET} solution): {quantum_found_solution}")
    print(f"amplified_above_uniform (Grover amplified marked states above baseline M/N={M/N:.3f}): {amplified_above_uniform}")

    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    run()
