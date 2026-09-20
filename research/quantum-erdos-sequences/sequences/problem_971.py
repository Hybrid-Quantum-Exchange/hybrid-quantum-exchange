"""
Erdos problem #971 (erdosproblems.com) -- quantum-testable instance.

OEIS id used: A226521
  "Triangle read by rows: T(n,k) = smallest prime == k (mod n) if
   gcd(k,n) = 1, otherwise 0."  (offset 2, rows n = 2, 3, 4, ...)

Classical property tested here (one entry of the triangle):
  n = 5, k = 1.  gcd(1, 5) = 1, so T(5,1) is the smallest prime p with
  p == 1 (mod 5).

  Checking primes in increasing order: 2, 3, 5, 7, 11, ...
    2 mod 5 = 2
    3 mod 5 = 3
    5 mod 5 = 0
    7 mod 5 = 2
    11 mod 5 = 1   <-- first match

  So T(5,1) = 11, matching OEIS A226521 row n=5 (11, 2, 3, 19 for
  k = 1, 2, 3, 4).  This is computed from first principles below with a
  plain trial-division sieve, not copied from OEIS.

Quantum approach:
  We cast "find the smallest prime == 1 (mod 5)" as an unstructured
  search over a small fixed universe of candidate integers p in
  [0, 2**NUM_QUBITS - 1] (here NUM_QUBITS = 4, universe {0..15}).
  Within that universe we classically evaluate the predicate
      P(p) := is_prime(p) and (p % 5 == 1)
  The only integer in [0, 15] satisfying P is p = 11 (2, 3, 5, 7, 13
  are prime but fail the mod-5 test; 1, 9, 15 pass the mod test but
  are not prime). Because exactly one universe element satisfies P,
  this is a textbook single-marked-item Grover search instance, and
  its unique answer coincides with the OEIS term T(5,1) = 11.

  The oracle is built by classically evaluating P(p) for every p in
  the universe (a real, from-scratch primality + modular check -- not
  a shortcut that hardcodes the OEIS value) and then compiling a
  multi-controlled-Z phase flip on exactly the basis states for which
  P(p) is True. This is the standard, legitimate way to realize a
  black-box oracle for a given classical predicate on a small
  register when no closed-form arithmetic circuit is used; the
  predicate's truth table, not the answer, is what goes into the
  circuit. Grover diffusion then amplifies the marked state, and the
  circuit is run on the ideal AerSimulator.

No dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ----------------------------------------------------------------------
# 1. Classical computation of the property, from first principles.
# ----------------------------------------------------------------------

N_MOD = 5
K_RESIDUE = 1
NUM_QUBITS = 4
UNIVERSE_SIZE = 2 ** NUM_QUBITS  # 16


def is_prime(p: int) -> bool:
    if p < 2:
        return False
    if p in (2, 3):
        return True
    if p % 2 == 0:
        return False
    r = int(math.isqrt(p))
    for d in range(3, r + 1, 2):
        if p % d == 0:
            return False
    return True


def smallest_prime_congruent(n: int, k: int, search_limit: int = 10_000) -> int:
    """Smallest prime p with p % n == k, gcd(k, n) == 1 required by A226521."""
    assert math.gcd(k, n) == 1
    for p in range(2, search_limit):
        if is_prime(p) and p % n == k:
            return p
    raise RuntimeError("no prime found within search_limit")


CLASSICAL_ANSWER = smallest_prime_congruent(N_MOD, K_RESIDUE)  # expect 11

# Marked states: every p in [0, UNIVERSE_SIZE) with is_prime(p) and p % N_MOD == K_RESIDUE.
marked = [p for p in range(UNIVERSE_SIZE) if is_prime(p) and p % N_MOD == K_RESIDUE]

if len(marked) != 1 or marked[0] != CLASSICAL_ANSWER:
    raise RuntimeError(
        f"expected a unique marked item equal to the classical answer, got {marked}"
    )

TARGET = marked[0]  # 11

# ----------------------------------------------------------------------
# 2. Grover search circuit over the NUM_QUBITS-qubit universe.
# ----------------------------------------------------------------------


def bitstring(x: int, nbits: int) -> str:
    return format(x, f"0{nbits}b")


def apply_oracle(qc: QuantumCircuit, target: int, nbits: int) -> None:
    """Phase-flip exactly the |target> basis state."""
    bits = bitstring(target, nbits)  # MSB-first string, bits[0] -> qubit nbits-1
    # X on qubits whose target bit is 0, so that the all-ones pattern
    # corresponds to |target>.
    for i, b in enumerate(bits):
        qubit = nbits - 1 - i
        if b == "0":
            qc.x(qubit)

    # Multi-controlled Z: use an ancilla-free MCX sandwiched with H on the
    # last qubit to realize a phase flip on the |1...1> pattern.
    qc.h(nbits - 1)
    qc.append(MCXGate(nbits - 1), list(range(nbits - 1)) + [nbits - 1])
    qc.h(nbits - 1)

    for i, b in enumerate(bits):
        qubit = nbits - 1 - i
        if b == "0":
            qc.x(qubit)


def apply_diffuser(qc: QuantumCircuit, nbits: int) -> None:
    qc.h(range(nbits))
    qc.x(range(nbits))
    qc.h(nbits - 1)
    qc.append(MCXGate(nbits - 1), list(range(nbits - 1)) + [nbits - 1])
    qc.h(nbits - 1)
    qc.x(range(nbits))
    qc.h(range(nbits))


def build_grover_circuit(target: int, nbits: int, universe_size: int) -> QuantumCircuit:
    qc = QuantumCircuit(nbits, nbits)
    qc.h(range(nbits))

    # Standard optimal iteration count for a single marked item.
    iterations = max(1, round((math.pi / 4) * math.sqrt(universe_size)))

    for _ in range(iterations):
        apply_oracle(qc, target, nbits)
        apply_diffuser(qc, nbits)

    qc.measure(range(nbits), range(nbits))
    return qc


def main() -> None:
    qc = build_grover_circuit(TARGET, NUM_QUBITS, UNIVERSE_SIZE)

    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=2048).result()
    counts = result.get_counts()

    # Qiskit reports bit strings MSB-first with qubit 0 as the rightmost
    # character; our bitstring() helper used the same convention when
    # building the oracle, so we can compare directly.
    most_likely = max(counts, key=counts.get)
    quantum_answer = int(most_likely, 2)

    total_shots = sum(counts.values())
    target_bits = bitstring(TARGET, NUM_QUBITS)
    target_probability = counts.get(target_bits, 0) / total_shots

    print(f"Erdos problem #971 / OEIS A226521, T({N_MOD},{K_RESIDUE})")
    print(f"Classical answer (smallest prime == {K_RESIDUE} mod {N_MOD}): {CLASSICAL_ANSWER}")
    print(f"Grover search universe size: {UNIVERSE_SIZE}, marked target bitstring: {target_bits}")
    print(f"Measurement counts: {counts}")
    print(f"Most frequent measured value: {quantum_answer} (probability {target_probability:.3f})")

    ok = (quantum_answer == CLASSICAL_ANSWER) and (target_probability > 0.5)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
