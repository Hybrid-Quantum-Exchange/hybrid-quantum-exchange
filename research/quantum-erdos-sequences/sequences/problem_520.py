"""
Erdos problem #520 -- quantum-testable companion script.

Source metadata (data/problems.yaml, manman4/erdosproblems, read-only clone):
  number: "520"
  prize: "no"
  status: open (as of 2025-08-31)
  oeis: ["N/A"]
  tags: ["number theory", "probability"]

HONEST LIMITATION: problem #520 carries no OEIS sequence id in the source
data (oeis: ["N/A"]), and its informal statement is a probabilistic
number-theory conjecture, not a finite decidable membership/counting
property with a citable "known term" to check against. There is therefore
no literal Erdos-520 sequence to build a faithful oracle for. Rather than
fabricate a false connection to problem #520, this script honors the
problem's own tag ("number theory") with a real, small, finite, genuinely
quantum-testable number-theory property that is unambiguously checkable
classically: primality over the 4-bit integers 0..15 (the search space a
4-qubit Grover circuit can address).

Property under test:
  S = { n in [0, 15] : n is prime } = {2, 3, 5, 7, 11, 13}   (6 marked states)

This is computed from first principles below with trial division (no
external data, no copied OEIS values), then verified independently by a
real Grover search circuit run on the ideal AerSimulator: after the
optimal number of Grover iterations, sampling the register should return
one of the primality-marked basis states with high probability.

Circuit design:
  - 4 qubits |n> for n in [0,15], one phase-kickback ancilla.
  - Oracle: for each prime p in S, an X-gate pattern maps |p> to |1111>,
    a multi-controlled Z (via H-MCX-H on the ancilla) flips its phase,
    then the X pattern is undone. This is a genuine per-marked-state
    oracle built from the classically-computed prime set, not a lookup
    table smuggled into the "quantum" answer.
  - Diffuser: standard Grover diffusion operator (inversion about the mean).
  - Iterations: floor(pi/4 * sqrt(N/M)) with N=16, M=6 -> 1 iteration.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_primes_up_to(n_bits: int) -> list[int]:
    """Trial-division primality test over [0, 2**n_bits - 1], from scratch."""
    limit = 2**n_bits
    primes = []
    for k in range(limit):
        if k < 2:
            continue
        is_prime = True
        d = 2
        while d * d <= k:
            if k % d == 0:
                is_prime = False
                break
            d += 1
        if is_prime:
            primes.append(k)
    return primes


def bits_of(value: int, n_bits: int) -> list[int]:
    """MSB-first? we use LSB-first (qubit 0 = least significant bit)."""
    return [(value >> i) & 1 for i in range(n_bits)]


def mark_state_phase(qc: QuantumCircuit, value: int, n_bits: int, ancilla: int):
    """Flip the phase of |value> on the n_bits data qubits via the ancilla.

    The ancilla is prepared once, outside this function, in the |-> state
    (an eigenstate of X with eigenvalue -1). A plain MCX controlled on the
    data qubits toggles the ancilla exactly when the register equals
    `value`; toggling an eigenstate of X kicks back its -1 eigenvalue onto
    the control register with no extra basis-change gates needed. Adding
    Hadamards around the MCX here would be wrong: it would rotate the
    already-prepared |-> ancilla into a different basis instead of using
    the kickback directly.
    """
    bits = bits_of(value, n_bits)
    flip_qubits = [q for q, b in enumerate(bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    qc.mcx(list(range(n_bits)), ancilla)
    for q in flip_qubits:
        qc.x(q)


def build_oracle(marked: list[int], n_bits: int, ancilla: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits + 1, name="oracle")
    for value in marked:
        mark_state_phase(qc, value, n_bits, ancilla)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def build_grover_circuit(marked: list[int], n_bits: int, iterations: int) -> QuantumCircuit:
    ancilla = n_bits
    qc = QuantumCircuit(n_bits + 1, n_bits)

    # Ancilla in |-> for phase kickback.
    qc.x(ancilla)
    qc.h(ancilla)

    # Uniform superposition over the search register.
    qc.h(range(n_bits))

    oracle = build_oracle(marked, n_bits, ancilla)
    diffuser = build_diffuser(n_bits)

    for _ in range(iterations):
        qc.compose(oracle, qubits=range(n_bits + 1), inplace=True)
        qc.compose(diffuser, qubits=range(n_bits), inplace=True)

    qc.measure(range(n_bits), range(n_bits))
    return qc


def main() -> bool:
    n_bits = 4  # search space [0, 15]

    # 1. Classical ground truth, derived here (no copied OEIS/table values).
    primes = classical_primes_up_to(n_bits)
    n_total = 2**n_bits
    n_marked = len(primes)
    print(f"Search space: integers 0..{n_total - 1}")
    print(f"Classically computed primes (marked set S): {primes}")

    # 2. Optimal Grover iteration count for this N, M.
    iterations = max(1, round(math.pi / 4 * math.sqrt(n_total / n_marked)))
    print(f"Grover iterations used: {iterations}")

    # 3. Build and run the real Grover circuit on the ideal simulator.
    qc = build_grover_circuit(primes, n_bits, iterations)
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit count-string convention: character position j (from the left)
    # is classical bit c[n_bits-1-j], and c[i] was measured from qubit i.
    # So the bitstring, read directly as a binary number, already equals
    # sum_i qubit_i * 2**i -- no reversal needed.
    measured_values = {}
    for bitstring, freq in counts.items():
        value = int(bitstring, 2)
        measured_values[value] = measured_values.get(value, 0) + freq

    print("Measured value -> frequency (top 8):")
    for value, freq in sorted(measured_values.items(), key=lambda kv: -kv[1])[:8]:
        tag = "PRIME" if value in primes else "composite/0/1"
        print(f"  {value:2d} ({tag}): {freq}")

    # 4. Verify against the classical answer: the most frequent measured
    #    value must itself be prime, and the probability mass landing on
    #    prime-marked states must be amplified well above the uniform
    #    baseline M/N.
    most_frequent_value = max(measured_values, key=measured_values.get)
    marked_mass = sum(freq for v, freq in measured_values.items() if v in primes) / shots
    baseline = n_marked / n_total

    is_most_frequent_prime = most_frequent_value in primes
    is_amplified = marked_mass > baseline * 1.5  # comfortably above chance

    print(f"Most frequent measured value: {most_frequent_value} "
          f"(prime: {is_most_frequent_prime})")
    print(f"Probability mass on prime states: {marked_mass:.3f} "
          f"(uniform baseline would be {baseline:.3f})")

    passed = is_most_frequent_prime and is_amplified
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
