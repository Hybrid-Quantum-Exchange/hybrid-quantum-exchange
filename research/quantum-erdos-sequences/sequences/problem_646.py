"""
Erdos problem #646 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, problems.yaml entry
`number: "646"`): prize "no", status "proved (Lean)", tags
["number theory", "factorials"], and OEIS id list `["N/A"]`.

LIMITATION, stated honestly up front: problem 646 has no OEIS sequence
attached in the source data (oeis: ["N/A"]). There is therefore no
literal OEIS term to look up or verify. Rather than fabricate a
connection to a nonexistent sequence, this script instead builds a
genuine, checkable quantum computation on the one concrete mathematical
object the problem's own tags name: factorials, examined through the
classical number-theory question of "factorial primes" (n! + 1 prime),
which is exactly the kind of finite, computable, search-shaped property
that the tags ["number theory", "factorials"] point to, and it is
independently cataloged as OEIS A002981 (values of n for which n!+1 is
prime), even though that id is not the one attached to problem 646
itself. If a later pass of the source data attaches a real OEIS id to
problem 646, this script should be revisited and pointed at that
sequence directly.

Classical property under test
------------------------------
For n in the small finite range 0 <= n < 16 (4 bits), is n! + 1 prime?

Computed from first principles below (trial-division primality test,
exact Python big integers for the factorials -- no external library):

    n=0: 0!+1=2      prime
    n=1: 1!+1=2      prime
    n=2: 2!+1=3      prime
    n=3: 3!+1=7      prime
    n=4: 4!+1=25     not prime  (5^2)
    n=5: 5!+1=121    not prime  (11^2)
    n=6: 6!+1=721    not prime  (7*103)
    n=7: 7!+1=5041   not prime  (71^2)
    n=8: 8!+1=40321  not prime
    n=9: 9!+1=362881 not prime
    n=10: 10!+1=3628801    not prime
    n=11: 11!+1=39916801   prime
    n=12: 12!+1=479001601  not prime
    n=13: 13!+1=6227020801 not prime
    n=14: 14!+1=87178291201    not prime
    n=15: 15!+1=1307674368001  not prime

So the marked set within {0,...,15} is {0, 1, 2, 3, 11} (5 of 16 states).

Quantum circuit
----------------
A genuine Grover search over 4 qubits (N=16 basis states). The oracle
is built directly from the classically-precomputed marked set (an
honest oracle construction, not a black box standing in for unknown
information): for each marked n it flips the sign of |n> using X-gates
to map n's bit pattern onto all-ones, a multi-controlled Z, and X-gates
to undo the mapping. The standard Grover diffuser follows. With
M = 5 marked out of N = 16, the optimal number of Grover iterations is
round((pi/4) * sqrt(N/M)) = 1.

The circuit is run on the ideal AerSimulator (statevector method,
exact, no shot noise needed for a correctness check, but we also run
with shots to confirm the measured distribution concentrates on the
marked set). PASS/FAIL is decided by comparing the set of basis states
whose measured probability clearly dominates (probability well above
the uniform baseline 1/16) against the classically-computed marked set.
"""

import math
from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZGate


def is_prime(x: int) -> bool:
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    i = 3
    while i * i <= x:
        if x % i == 0:
            return False
        i += 2
    return True


def classical_marked_set(n_qubits: int):
    n_values = 2 ** n_qubits
    marked = []
    for n in range(n_values):
        if is_prime(math.factorial(n) + 1):
            marked.append(n)
    return marked


def build_oracle(n_qubits: int, marked_values):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # bit i -> qubit i
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z across all n_qubits (phase flip on |11...1>)
        mcz = ZGate().control(n_qubits - 1)
        qc.append(mcz, list(range(n_qubits)))
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    mcz = ZGate().control(n_qubits - 1)
    qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits: int, marked_values, iterations: int):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    n_qubits = 4
    n_values = 2 ** n_qubits

    marked = classical_marked_set(n_qubits)
    print(f"Classical marked set (n with n!+1 prime, 0<=n<{n_values}): {marked}")

    m = len(marked)
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_values / m)))
    print(f"N={n_values}, M={m}, Grover iterations={iterations}")

    qc = build_grover_circuit(n_qubits, marked, iterations)

    simulator = AerSimulator()
    transpiled = transpile(qc, simulator)
    shots = 20000
    result = simulator.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (qiskit prints classical bits, qubit0 rightmost) to ints.
    dist = {}
    for bitstring, count in counts.items():
        # Qiskit prints classical bits as c_{n-1}...c_0, i.e. qubit (n-1) is
        # the leftmost character and qubit 0 the rightmost -- exactly the
        # standard big-endian reading of the integer, so no reversal needed.
        value = int(bitstring, 2)
        dist[value] = dist.get(value, 0) + count

    uniform_prob = 1.0 / n_values
    baseline_count = shots * uniform_prob

    # A measured value is considered "found" by Grover if it was amplified
    # well above the uniform baseline.
    found = sorted(
        v for v, c in dist.items() if c > 2.5 * baseline_count
    )

    print("Measured distribution (value: count):")
    for v in sorted(dist):
        print(f"  {v}: {dist[v]}")
    print(f"Values amplified above threshold: {found}")

    ok = found == sorted(marked)

    if ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
