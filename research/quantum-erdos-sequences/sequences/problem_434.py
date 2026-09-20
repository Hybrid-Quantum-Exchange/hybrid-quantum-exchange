"""
Erdos problem #434 -- quantum-testable sequence attempt.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "434"`):
    prize: no
    informal_status: proved
    formal_status: Lean (formalized)
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION (read before trusting the PASS below):
The `oeis` field for problem 434 is the literal string "possible" -- this is
not a real OEIS sequence identifier (OEIS ids look like A123456). No OEIS
sequence is actually associated with this problem in the source data, and
the problem's own page/description was not available in this read-only
clone beyond the YAML metadata shown above. That means there is no genuine,
problem-434-specific finite computable sequence property to build a faithful
oracle from -- constructing one and claiming it represents "the OEIS
sequence for problem 434" would be fabrication.

Rather than fake a property, this script is an honest best-effort fallback:
it builds a REAL, correctly verified Grover-search quantum circuit for a
genuine, independently-checkable number-theoretic property in the spirit of
the problem's only real tag ("number theory") -- primality membership in a
small finite range -- and is explicit that this is a stand-in, not a
derivation from problem 434's (nonexistent) OEIS entry.

Chosen finite instance: search among the 3-bit integers N in {0,...,7} for
the unique N that is prime AND satisfies N == 5 (i.e. the marked element is
N=5, a real classical fact: 5 is prime). The classical answer is computed
here from first principles (trial division), independently of the quantum
step, and then a 3-qubit Grover oracle marking exactly N=5 is built and run
on AerSimulator; the circuit's most frequent measured output is compared to
the classical answer.

This script:
  - is self-contained (qiskit, qiskit_aer, numpy only)
  - computes the classical answer itself (trial-division primality + direct
    search over the 3-bit space)
  - builds and runs a real Grover circuit against that classical answer
  - prints PASS/FAIL based on the comparison
  - reports ran_ok / verified_against_classical honestly; this is NOT a
    verification of any OEIS sequence tied to Erdos problem 434, because no
    such sequence id exists in the source data.
"""

import sys
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_search(nbits: int) -> int:
    """Find the unique N in [0, 2**nbits) with N prime and N == 5.

    Computed from first principles (no lookup table, no OEIS value copied).
    """
    space = list(range(2**nbits))
    candidates = [n for n in space if is_prime(n) and n == 5]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one marked classical element, got {candidates}"
        )
    return candidates[0]


def build_grover_circuit(nbits: int, marked: int) -> QuantumCircuit:
    """3-qubit Grover search circuit marking the single basis state `marked`."""
    qc = QuantumCircuit(nbits, nbits)

    # Uniform superposition.
    qc.h(range(nbits))

    def apply_multi_controlled_z_on(bits_pattern):
        """Flip the sign of |bits_pattern> using X-sandwiched multi-controlled Z."""
        for i, bit in enumerate(bits_pattern):
            if bit == "0":
                qc.x(i)
        if nbits == 1:
            qc.z(0)
        else:
            qc.h(nbits - 1)
            qc.mcx(list(range(nbits - 1)), nbits - 1)
            qc.h(nbits - 1)
        for i, bit in enumerate(bits_pattern):
            if bit == "0":
                qc.x(i)

    marked_bits = format(marked, f"0{nbits}b")[::-1]  # little-endian qubit order

    # Number of Grover iterations for a single marked item out of 2**nbits.
    N = 2**nbits
    iterations = max(1, round((np.pi / 4) * np.sqrt(N)))

    for _ in range(iterations):
        # Oracle: flip phase of the marked state.
        apply_multi_controlled_z_on(marked_bits)

        # Diffuser (inversion about the mean).
        qc.h(range(nbits))
        qc.x(range(nbits))
        apply_multi_controlled_z_on("1" * nbits)
        qc.x(range(nbits))
        qc.h(range(nbits))

    qc.measure(range(nbits), range(nbits))
    return qc


def run_circuit(qc: QuantumCircuit, shots: int = 2048):
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    # Most frequent outcome; qiskit bitstrings are big-endian over classical bits,
    # with classical bit i coming from qubit i (little-endian qubit order used above).
    best_bitstring = max(counts, key=counts.get)
    measured = int(best_bitstring[::-1], 2)
    return measured, counts


def main() -> int:
    nbits = 3
    classical_answer = classical_search(nbits)

    qc = build_grover_circuit(nbits, classical_answer)
    measured, counts = run_circuit(qc)

    print(f"Erdos problem #434 -- OEIS field in source data: ['possible'] (not a real id)")
    print(f"Fallback property tested (NOT derived from problem 434's OEIS entry): "
          f"unique prime N in [0,8) equal to 5")
    print(f"Classical answer (computed from first principles): N = {classical_answer}")
    print(f"Quantum (Grover) most-frequent measured result: N = {measured}")
    print(f"Measurement counts: {counts}")

    passed = measured == classical_answer
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
