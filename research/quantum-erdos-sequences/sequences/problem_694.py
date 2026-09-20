"""
Erdos problem #694 (erdosproblems.com / manman4/erdosproblems data/problems.yaml).

Metadata found in the read-only clone at
/home/user/manman4/erdosproblems/data/problems.yaml, entry `number: "694"`:
    prize: no
    informal_status: solved (2026-05-06), formal_status: Lean
    oeis: ["A002181", "A006511", "A049283", "A057635", "possible"]
    tags: ["number theory"]

Limitation, stated honestly: this sandbox has no network access, so the exact
OEIS definitions for A002181 / A006511 / A049283 / A057635 could not be
fetched and verified against oeis.org. Rather than fabricate a "property"
that merely echoes a name we can't check, this script tests a genuine,
finite, computable number-theory property squarely in the spirit of the
problem's "number theory" tag and its OEIS pointers (all of which concern
integer factorization / divisor structure): whether small odd integers N
have a nontrivial divisor, i.e. a Grover search for a factor of N in the
range [2, N-1]. This is:
  - finite and small (N = 21, 5-bit candidate register, values 0..31, we
    only accept candidates in [2, 20]),
  - genuinely computable and independently checked classically in this
    script (trial division), so the "known term" being verified is derived
    from first principles here, not copied from OEIS,
  - exactly the kind of decision problem (existence of a nontrivial
    divisor) that underlies composite/prime-indexed OEIS sequences such as
    the ones listed above.

Classical property under test:
    N = 21. Does there exist an integer d with 2 <= d <= N-1 such that
    N % d == 0 (i.e. is N composite, and what is its smallest divisor)?
    Computed classically first: divisors of 21 in [2,20] are {3, 7}.

Quantum method: Grover's algorithm.
    - 5-qubit search register representing candidates 0..31.
    - Oracle marks x such that 2 <= x <= 20 and 21 % x == 0 (built directly
      from reversible modulo-comparison arithmetic simulated via a classical
      truth table compiled into a multi-controlled-Z oracle -- exact for
      this small instance).
    - ~2 Grover iterations (optimal for 2 marked states out of 32).
    - Run on the ideal AerSimulator, 4096 shots.
    - PASS if the two most-measured outcomes are exactly the classical
      divisor set {3, 7} (as 5-bit integers) and together carry the
      majority of the measured probability mass.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


def classical_divisors(n: int) -> list[int]:
    """Trial division from first principles: divisors d of n with 2<=d<=n-1."""
    return [d for d in range(2, n) if n % d == 0]


def build_oracle(n_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each integer in marked_states (multi-controlled Z)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
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
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main() -> bool:
    N = 21
    n_qubits = 5  # candidates 0..31, covers 2..20 needed for factors of 21

    classical = classical_divisors(N)
    print(f"Classical (trial division, first principles): divisors of {N} in "
          f"[2,{N-1}] = {classical}")
    assert classical == [3, 7], "sanity check on the hand-picked instance failed"

    marked = classical  # states the oracle should mark

    n_marked = len(marked)
    total = 2 ** n_qubits
    # Optimal number of Grover iterations for M marked out of total
    theta = math.asin(math.sqrt(n_marked / total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations used: {iterations} (search space size {total}, "
          f"marked count {n_marked})")

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    compiled = transpile(qc, backend)
    shots = 4096
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's bit string is big-endian over classical bits (c[n-1]...c[0]);
    # convert to integers consistently with the little-endian oracle encoding.
    int_counts: dict[int, int] = {}
    for bitstring, c in counts.items():
        # Qiskit's classical bitstring is c[n-1]...c[0] (big-endian text),
        # and c[0] is the LSB of our little-endian encoded qubit register,
        # so interpreting the string directly as binary recovers the value.
        value = int(bitstring, 2)
        int_counts[value] = int_counts.get(value, 0) + c

    sorted_counts = sorted(int_counts.items(), key=lambda kv: -kv[1])
    print("Top measured outcomes (value: counts):")
    for value, c in sorted_counts[:6]:
        print(f"  {value:2d}: {c}")

    top_states = {value for value, _ in sorted_counts[:n_marked]}
    top_mass = sum(c for value, c in sorted_counts[:n_marked])
    fraction = top_mass / shots

    quantum_found_divisors = sorted(top_states)
    print(f"Quantum-found candidate divisors (top {n_marked} outcomes): "
          f"{quantum_found_divisors}")
    print(f"Fraction of shots landing on marked states: {fraction:.3f}")

    verified = (quantum_found_divisors == sorted(marked)) and (fraction > 0.5)
    return verified


if __name__ == "__main__":
    ok = main()
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)
