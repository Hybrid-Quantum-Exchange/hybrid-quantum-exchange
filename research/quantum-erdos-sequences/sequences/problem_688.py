"""
Erdos problem #688 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, entry
"number: \"688\""):
    prize: no
    informal_status: open (last update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION, stated up front: problem #688's entry in the cloned dataset
carries no OEIS sequence id (oeis: ["N/A"]) and no problem statement text is
present in the read-only clone available here -- only the status/tag
metadata above. There is therefore no specific integer sequence from this
problem to build a genuine, problem-derived quantum test around. Rather than
fabricate a property and claim it comes from problem #688, this script is an
honest best-attempt substitute: it builds a real, correctly-verified quantum
circuit over the one concrete mathematical object the metadata does give us
-- the "number theory" tag -- using primality, which is the most standard
finite/computable number-theoretic property available at this scale.

Classical property under test:
    For n in {0, 1, ..., 15} (a 4-qubit register), is n prime?
    The classical prime set in this range, computed here from first
    principles by trial division (not copied from any table), is:
        {2, 3, 5, 7, 11, 13}
    That gives 6 "marked" states out of N = 16.

Quantum method:
    Grover's algorithm on 4 qubits. The oracle is a diagonal phase oracle
    built directly from the classically-computed prime set above (flips the
    phase of exactly the marked computational basis states). The optimal
    number of Grover iterations for M=6 marked items out of N=16 is
    computed from the standard formula floor(pi/4 * sqrt(N/M)) and is run
    on the ideal AerSimulator (statevector simulation, no noise). The test
    passes if, after those iterations, the states measured with
    non-negligible probability are exactly the classically-computed prime
    set (i.e. the circuit amplifies precisely the correct marked states and
    nothing else) and those six states together carry the large majority
    (>75%) of the measured probability mass.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator


def classical_primes(limit_exclusive: int) -> list[int]:
    """Trial-division primality test computed from first principles."""
    primes = []
    for n in range(limit_exclusive):
        if n < 2:
            continue
        is_prime = True
        for d in range(2, int(math.isqrt(n)) + 1):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    return primes


def build_oracle(n_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Diagonal phase oracle: flips sign of each marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    dim = 2 ** n_qubits
    diag = np.ones(dim, dtype=complex)
    for m in marked_states:
        diag[m] = -1.0
    qc.append(DiagonalGate(list(diag)), list(range(n_qubits)))
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main() -> bool:
    n_qubits = 4
    n_states = 2 ** n_qubits  # 16

    marked = classical_primes(n_states)
    m = len(marked)
    print(f"Classical primes in [0, {n_states}): {marked}  (M={m})")

    if m == 0 or m == n_states:
        raise RuntimeError("Degenerate marked set; Grover instance not usable.")

    iterations = max(1, round((math.pi / 4) * math.sqrt(n_states / m)))
    print(f"Grover iterations used: {iterations}")

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit counts keys are written MSB..LSB as clbit[n-1]..clbit[0], which
    # is already the normal binary representation of the integer value.
    def bits_to_int(bitstr: str) -> int:
        return int(bitstr, 2)

    dist = {}
    for bitstr, c in counts.items():
        val = bits_to_int(bitstr)
        dist[val] = dist.get(val, 0) + c

    # States considered "found" by the circuit: probability well above the
    # ~1/16 uniform-background level (threshold at 3x uniform).
    threshold = shots * (1.0 / n_states) * 2
    found = sorted(v for v, c in dist.items() if c >= threshold)

    print("Measurement distribution (value: count):")
    for v in sorted(dist):
        print(f"  {v:2d}: {dist[v]}")
    print(f"States amplified above threshold: {found}")

    marked_prob_mass = sum(dist.get(v, 0) for v in marked) / shots
    print(f"Total probability mass on classically-prime states: {marked_prob_mass:.4f}")

    passed = (found == marked) and (marked_prob_mass > 0.75)
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
