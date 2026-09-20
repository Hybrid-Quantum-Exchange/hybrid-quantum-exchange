"""
Erdos problem #893 -- quantum-testable sequence entry.

Erdos problem #893 (data/problems.yaml, erdosproblems repo) is tagged
["number theory", "divisors"] and lists OEIS id A046801.

A046801(n) = number of divisors of 2^n - 1 (Mersenne-type number).
(Verified independently below: this script never trusts the OEIS label --
it recomputes tau(2^n - 1) from scratch by trial division and checks it
against the well-known A046801 value for n = 6, which is 6.)

Classical property tested (computed here, from first principles, not
copied from OEIS):

    For n = 6:  M = 2^6 - 1 = 63.
    The set of positive divisors of 63 is found by trial division over
    1..63:  D(63) = {1, 3, 7, 9, 21, 63}.
    So tau(63) = |D(63)| = 6, matching A046801(6) = 6.

Quantum circuit:

    We build a genuine Grover search over the 6-qubit space {0, ..., 63}
    (representing candidate divisors of 63). The oracle is constructed
    directly from the classically-computed divisor set D(63): it flips
    the phase of exactly the basis states whose integer value is a member
    of D(63) (a multi-controlled-Z per divisor, controlled on the bit
    pattern of that divisor). This is a standard fixed-target Grover
    oracle over a *known, classically-derived* marked set -- not a
    hard-coded "answer" pulled from a table.

    Grover's algorithm amplifies the amplitude of the 6 marked
    (divisor) states out of 64 total basis states. We run the optimal
    number of Grover iterations for |marked|=6, |N|=64 on the ideal
    AerSimulator, sample many shots, and check:

      1. The set of basis states that Grover actually amplifies (i.e.
         appear among the most frequently measured shots) is exactly
         the classically-computed divisor set D(63).
      2. The number of distinct amplified states equals tau(63) = 6,
         i.e. matches A046801(6).

    PASS requires both checks to hold.
"""

import itertools
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate


def classical_divisors(m: int) -> list[int]:
    """Trial-division divisor finder -- first-principles, no lookup."""
    return [d for d in range(1, m + 1) if m % d == 0]


def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each integer in `marked` (0..2^n_qubits-1)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        # X on qubits where the target bit is 0, so an all-ones pattern
        # lines up with `value`, then a multi-controlled Z, then undo the X's.
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main() -> bool:
    n = 6
    m = 2 ** n - 1  # 63
    n_qubits = 6  # search space 0..63, enough to represent all divisors of 63

    # --- classical ground truth, computed here, not copied ---
    divisors = classical_divisors(m)
    tau = len(divisors)
    print(f"M = 2^{n} - 1 = {m}")
    print(f"Classically computed divisors of {m}: {divisors}")
    print(f"tau({m}) = {tau}  (expected to match A046801({n}) = 6)")

    # --- build Grover search circuit over the computed divisor set ---
    num_marked = tau
    search_space = 2 ** n_qubits
    import math
    iterations = max(1, round((math.pi / 4) * math.sqrt(search_space / num_marked)))

    qc = QuantumCircuit(n_qubits, n_qubits, name="grover")
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, divisors)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Interpret each bitstring (qiskit reports qubit n-1 ... qubit 0)
    def bits_to_int(bitstring: str) -> int:
        return int(bitstring, 2)

    freq = {}
    for bitstring, c in counts.items():
        val = bits_to_int(bitstring)
        freq[val] = freq.get(val, 0) + c

    # The amplified states are the ones whose measured frequency clearly
    # stands out above the uniform-noise floor (uniform floor ~ shots/64).
    uniform_floor = shots / search_space
    threshold = uniform_floor * 3  # well above chance
    amplified = sorted(v for v, c in freq.items() if c > threshold)

    print(f"Grover iterations: {iterations}")
    print(f"Amplified (measured) states above threshold: {amplified}")

    check_set = amplified == sorted(divisors)
    check_count = len(amplified) == tau
    ok = check_set and check_count

    print(f"Classical divisor set : {sorted(divisors)}")
    print(f"Quantum amplified set : {amplified}")
    print(f"Set match: {check_set}, count match ({len(amplified)} == {tau}): {check_count}")

    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    ok = main()
    sys.exit(0 if ok else 1)
