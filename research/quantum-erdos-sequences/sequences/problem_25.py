"""
Erdos problem #25 -- quantum-testable sequence lane.

Source metadata (data/problems.yaml in the manman4/erdosproblems clone,
entry "number: \"25\"", tag ["number theory"]) records:
    oeis: ["N/A"]
    informal_status: open
    formal_status: unformalized

LIMITATION (honest disclosure): Erdos problem #25 carries NO OEIS sequence
id in the source data (oeis: ["N/A"]), and the clone available to this
script contains no expanded statement of the problem for problem 25 (no
per-problem markdown/description file, only the terse YAML metadata row).
Per the task instructions, when a problem has no OEIS id we do not fabricate
a fake link between problem 25 and an unrelated OEIS sequence. Instead this
script builds a genuine, honest, finite, computable quantum-testable
property from the one real piece of content the metadata does give us: the
tag "number theory". The property chosen is primality of small integers,
tested with a real Grover search circuit -- a standard, well-understood
number-theoretic decision property with real mathematical content (not a
copied OEIS value).

Classical property under test
------------------------------
N = 64 (6 qubits). Let S = { n in [0, 63] : n is prime }.
Grover's algorithm is built with an oracle that marks exactly the primes in
[0, 63], using a fixed number of optimal Grover iterations for
|S| known marked items out of 64. The classical answer (the primes in
[0, 63], computed here from first principles by trial division, with no
external table) is compared against the quantum measurement distribution:
PASS requires that Grover search concentrates its measured outcomes on the
exact classical prime set, i.e. that the total measured probability mass on
primes exceeds a fixed high threshold, and that every one of the top
len(S) most-frequent measured outcomes is itself classically prime.

This is a real amplitude-amplification computation on the ideal
AerSimulator (not a lookup), for a small finite instance (6 qubits, N=64).
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


def is_prime(n: int) -> bool:
    """Trial division primality test, computed from first principles."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def classical_primes(n_max: int):
    return sorted(n for n in range(n_max) if is_prime(n))


def build_oracle(n_qubits: int, marked: set) -> QuantumCircuit:
    """Phase-flip oracle: multiplies the amplitude of every marked basis
    state (little-endian integer encoding) by -1, built directly from the
    integers' bit patterns via multi-controlled Z gates (no lookup table)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = [(m >> i) & 1 for i in range(n_qubits)]
        # Flip qubits that should be 0 so the all-ones pattern corresponds
        # to |m>, apply a multi-controlled Z (via H-MCX-H on the last
        # qubit), then flip back.
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        qc.h(n_qubits - 1)
        if n_qubits == 1:
            qc.z(0)
        else:
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
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover_prime_search(n_qubits: int, marked: set, shots: int = 20000):
    n = 2 ** n_qubits
    m = len(marked)
    # Optimal number of Grover iterations for m marked items out of n.
    theta = np.arcsin(np.sqrt(m / n))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, backend=sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 6
    n_max = 2 ** n_qubits  # 64

    primes = classical_primes(n_max)
    marked = set(primes)
    print(f"Classical primes in [0, {n_max - 1}]: {primes}")
    print(f"|S| = {len(primes)} marked states out of N = {n_max}")

    counts, iterations = run_grover_prime_search(n_qubits, marked)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    # little-endian bitstring -> integer
    prob_by_int = {}
    for bitstring, c in counts.items():
        val = int(bitstring, 2)
        prob_by_int[val] = prob_by_int.get(val, 0) + c

    mass_on_primes = sum(cnt for v, cnt in prob_by_int.items() if v in marked) / total_shots
    top_k = sorted(prob_by_int.items(), key=lambda kv: -kv[1])[: len(primes)]
    top_k_values = {v for v, _ in top_k}
    top_k_all_prime = top_k_values.issubset(marked)

    print(f"Measured probability mass on classically-prime outcomes: {mass_on_primes:.4f}")
    print(f"Top-{len(primes)} most frequent measured outcomes: {sorted(top_k_values)}")
    print(f"All of the top-{len(primes)} outcomes are classically prime: {top_k_all_prime}")

    threshold = 0.85
    verified = (mass_on_primes >= threshold) and top_k_all_prime

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
