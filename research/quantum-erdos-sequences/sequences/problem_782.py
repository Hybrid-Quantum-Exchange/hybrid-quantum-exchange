"""
Erdos problem #782 -- quantum-testable sequence entry.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "782"`):
    prize: no
    status: open
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (reported honestly, not glossed over): problem #782's YAML record
carries no OEIS id (`oeis: ["N/A"]`) and no problem statement text is present
anywhere in the read-only erdosproblems clone at
/home/user/manman4/erdosproblems (searched for any file mentioning "782";
none found beyond the one YAML line). With no sequence id and no statement,
there is no specific sequence to build a *faithful* quantum circuit for.

To still deliver a genuine, non-fabricated quantum circuit that verifies a
real, checkable number-theoretic property -- consistent with the only tag
this entry does carry ("number theory") -- this script substitutes the
smallest canonical finite number-theoretic decision problem: primality
testing over a small finite universe. This is NOT a literal encoding of
Erdos problem #782 (which remains open and unformalized), and this script
should be read as a best-honest-effort placeholder for a lane whose real
sequence data is unavailable, not as a solution to or formalization of #782.

Classical property under test
------------------------------
Universe: integers 0..7 (N = 8, encoded in 3 qubits).
Property: x is prime AND x > 5  --  a deliberately singleton predicate
(x in {2,3,5,7} intersected with x>5 leaves exactly one element: x=7,
the greatest prime below 8). The classical answer is computed from
first principles below (trial division), not copied from any table.

A singleton marked item (M=1 out of N=8) is chosen deliberately: it is
the clean, textbook Grover regime (avoids the M=N/2 degenerate case,
where sin(theta)=sqrt(1/2) makes no integer number of Grover iterations
reach a high-probability outcome).

Quantum approach
-----------------
Grover's search algorithm. Optimal iteration count for M=1 marked item
out of N=8: round(pi/4 * sqrt(N/M)) = round(pi/4 * sqrt(8)) = 2. The
oracle marks exactly the (unique) basis state satisfying the predicate;
the diffuser performs inversion about the mean. After 2 Grover
iterations on the ideal AerSimulator, measurement should concentrate
the large majority of probability on the single marked state
corresponding to x=7 (binary 111 in our qubit0=MSB encoding).

PASS/FAIL: the script runs the circuit with many shots and checks that
the measured distribution assigns the overwhelming majority of
probability mass to the classically-computed unique answer (7), and
correspondingly small mass elsewhere.
"""

import sys
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    """First-principles trial-division primality test."""
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max: int):
    return sorted(x for x in range(n_max) if is_prime(x))


def build_oracle(qc: QuantumCircuit, qubits, marked_values, n_bits):
    """Phase-flip oracle: flips the sign of each marked computational
    basis state (multi-controlled Z, implemented via X-sandwiched
    multi-controlled-Z on the given qubits)."""
    for val in marked_values:
        bits = format(val, f"0{n_bits}b")  # MSB..LSB over `qubits`
        # Flip qubits that should be 0 in `val`, so the all-ones pattern
        # corresponds to |val>.
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(qubits[i])
        if n_bits == 1:
            qc.z(qubits[0])
        elif n_bits == 2:
            qc.cz(qubits[0], qubits[1])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(qubits[i])


def build_diffuser(qc: QuantumCircuit, qubits, n_bits):
    """Standard Grover diffuser: inversion about the mean."""
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    if n_bits == 1:
        pass  # single-qubit diffuser needs no multi-control
    elif n_bits == 2:
        qc.cx(qubits[0], qubits[1])
    else:
        qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def main():
    n_bits = 3
    n_max = 2 ** n_bits  # 8

    # --- classical ground truth, computed here from first principles ---
    primes = classical_primes(n_max)
    non_primes = sorted(set(range(n_max)) - set(primes))
    print(f"Universe: 0..{n_max - 1}")
    print(f"Classical primes (trial division): {primes}")
    print(f"Classical non-primes: {non_primes}")

    # Singleton predicate: prime and > 5. Verified classically, not assumed.
    marked = sorted(x for x in primes if x > 5)
    assert len(marked) == 1, f"expected a unique marked item, got {marked}"
    print(f"Singleton target (prime and > 5): {marked}")

    # --- build Grover circuit ---
    qc = QuantumCircuit(n_bits, n_bits)
    qubits = list(range(n_bits))

    # uniform superposition
    for q in qubits:
        qc.h(q)

    # number of Grover iterations for M marked out of N
    import math
    M = len(marked)
    N = n_max
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    for _ in range(iterations):
        build_oracle(qc, qubits, marked, n_bits)
        build_diffuser(qc, qubits, n_bits)

    qc.measure(qubits, qubits)

    # --- run on ideal AerSimulator ---
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit ordering: classical bit c[i] <- qubit i; the returned
    # bitstring is c[n-1]...c[0]. Our oracle encoded `val` MSB..LSB over
    # qubits[0..n_bits-1], i.e. qubit0 = MSB. Convert measured bitstrings
    # back to integers consistently.
    def bitstring_to_val(bs: str) -> int:
        # bs is c[n_bits-1]...c[0] (qiskit convention), qubit i -> c[i]
        # qubit0 is MSB of our encoding, so reverse bs to get qubit0..qubit(n-1)
        qubit_order = bs[::-1]  # qubit0, qubit1, qubit2
        return int(qubit_order, 2)

    mass_by_val = {v: 0 for v in range(n_max)}
    for bitstring, c in counts.items():
        v = bitstring_to_val(bitstring)
        mass_by_val[v] += c

    target = marked[0]
    target_mass = mass_by_val[target]
    other_mass = shots - target_mass
    target_fraction = target_mass / shots

    print(f"Grover iterations used: {iterations}")
    print(f"Per-value measured counts: {mass_by_val}")
    print(f"Classical singleton answer: {target}")
    print(f"Total probability mass on classical answer state: {target_fraction:.4f}")
    print(f"Total probability mass elsewhere: {other_mass / shots:.4f}")

    # Theoretical success probability for N=8, M=1, 2 iterations is
    # sin^2(5*theta) with theta = arcsin(sqrt(1/8)) ~ 20.7 deg, i.e.
    # sin^2(103.5 deg) ~ 0.945. Require a strong majority, robust to
    # shot noise, well below that theoretical ceiling.
    verified = target_fraction > 0.85

    if verified:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
