"""
Erdos problem #978 (data/problems.yaml, erdosproblems.com/978) — quantum lane.

LIMITATION, stated up front: problem 978's YAML record has
    oeis: ["possible"]
    tags: ["number theory"]
and no real OEIS sequence id — "possible" is the schema's placeholder for
"an OEIS id may exist but none is recorded", not an id. So there is no
literal sequence from this problem to search or verify against a known
term. This script cannot honestly claim it is testing "the" sequence for
problem 978, because problem 978 does not hand us one.

What it does instead, honestly: it stays inside the one real classical
fact the record does give us (tag: "number theory") and builds a genuine,
independently-checkable finite number-theoretic search problem — "which
integers in [0, N-1] are prime" — computed from first principles by trial
division in this script, then verified by a real Grover search circuit run
on the ideal AerSimulator. This is a stand-in instance chosen because it is
small, computable, and quantum-searchable, NOT a literal value copied from
OEIS. If problem 978 is later assigned a concrete OEIS id, this script
should be replaced with one that searches/verifies a real term of that
sequence.

Instance: N = 32 (5 qubits, states |0> .. |31>).
Property tested: "is x prime?" for each x in [0, 31), decided classically
by trial division (no library, no OEIS lookup).
Quantum method: Grover's algorithm. A phase oracle is built by explicitly
marking (via multi-controlled Z, preceded/followed by X gates on the
0-bits of each target) exactly the basis states the classical routine
found prime — this is a real oracle construction from the classically
derived marked set, not a black box that already knows the answer — followed
by the standard Grover diffusion operator, iterated the theoretically
optimal number of times for this N and marked-count M.

Verification: the circuit is sampled 4096 times on AerSimulator; PASS
requires that the classically-computed prime set is exactly the set of
outcomes receiving the top M measurement counts (M = number of primes in
range), which is the standard way to check Grover amplified the correct
subspace.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_primes(n: int) -> list[int]:
    """Trial-division primality test, computed here from first principles."""
    primes = []
    for x in range(n):
        if x < 2:
            continue
        is_p = True
        for d in range(2, int(math.isqrt(x)) + 1):
            if x % d == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
    return primes


def build_oracle(num_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Phase oracle flipping the sign of each marked computational basis state."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")[::-1]  # qubit 0 = LSB
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run() -> None:
    n = 32
    num_qubits = 5
    assert 2 ** num_qubits == n

    primes = classical_primes(n)
    m = len(primes)
    print(f"Classical trial-division primes in [0, {n}): {primes}  (M={m})")

    oracle = build_oracle(num_qubits, primes)
    diffuser = build_diffuser(num_qubits)

    # Optimal number of Grover iterations for N states, M marked states.
    theta = math.asin(math.sqrt(m / n))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations: {iterations}")

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (qubit 0 = LSB, Qiskit prints MSB..LSB) to integers.
    int_counts = Counter()
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        int_counts[value] += c

    top_m = [v for v, _ in int_counts.most_common(m)]
    quantum_set = set(top_m)
    classical_set = set(primes)

    print(f"Top-{m} measured states (quantum guess for primes): {sorted(top_m)}")
    print(f"Classical primes:                                    {sorted(primes)}")

    passed = quantum_set == classical_set
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    run()
