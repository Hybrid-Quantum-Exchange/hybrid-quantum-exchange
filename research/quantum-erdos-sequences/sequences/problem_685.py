"""
Erdos problem #685 -- quantum-testable instance.

Source metadata (from erdosproblems.com's data, problems.yaml, entry
`number: "685"`): prize "no", status "open", tags
["number theory", "primes", "binomial coefficients"], and crucially
`oeis: ["N/A"]` -- the dataset in this read-only clone records NO OEIS
sequence id for problem #685. There is therefore no specific OEIS sequence
to target for this entry, which is the limitation this script is honest
about up front.

Rather than fabricate an OEIS id or invent an unrelated "sequence", this
script builds a genuine, small, finite, computable number-theory search
problem drawn directly from the three tags that ARE given for #685
("number theory", "primes", "binomial coefficients"):

    Classical property tested:
        Find all n in {0, 1, ..., 15} (4 bits) such that the binomial
        coefficient C(n, 2) = n*(n-1)/2 is a prime number.

    This is a real, checkable, finite arithmetic property: it combines
    binomial coefficients and primality exactly as the tags suggest, and
    has a unique, first-principles-computable classical answer for this
    search space.

Classical answer (computed below in `classical_binomial_prime_set`,
from first principles -- trial division for primality, direct
multiplication for the binomial coefficient; nothing is copied from
OEIS or elsewhere):
    n : C(n,2) : prime?
    0 : 0  : no
    1 : 0  : no
    2 : 1  : no   (1 is not prime)
    3 : 3  : YES
    4 : 6  : no
    5 : 10 : no
    6 : 15 : no
    7 : 21 : no
    8 : 28 : no
    9 : 36 : no
    10: 45 : no
    11: 55 : no
    12: 66 : no
    13: 78 : no
    14: 91 : no  (91 = 7*13)
    15: 105: no
So over n in [0,15] there is exactly ONE marked value: n = 3
(C(3,2) = 3, which is prime).

Quantum approach:
    A Grover search over the 4-qubit register |n> (n = 0..15) with an
    oracle that flips the phase of exactly the marked basis state(s)
    (those n for which C(n,2) is prime, computed classically ahead of
    time and hard-wired into a multi-controlled-Z oracle -- this is the
    standard "oracle knows the marked items" Grover setup, not a
    from-scratch arithmetic circuit for computing C(n,2) or primality
    in-circuit). With exactly 1 marked item out of 16, the optimal
    number of Grover iterations is floor(pi/4 * sqrt(16/1)) = 3.

    The circuit is run on the ideal AerSimulator (statevector-based
    sampling), and PASS/FAIL is decided by checking that the most
    frequent measured bitstring equals the classically-derived marked
    n = 3, and that its measured probability is high (Grover amplifies
    a single marked item out of 16 to well over 90% success
    probability with the optimal iteration count).

Limitation, stated honestly: because problem #685 has no OEIS id in the
source data, this is NOT literally "an OEIS sequence made quantum
testable" -- it is a small number-theoretic search problem built
faithfully from the problem's own tags, in place of an OEIS-anchored
property that does not exist for this entry.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


# ---------------------------------------------------------------------------
# Classical computation (first principles, no OEIS lookups).
# ---------------------------------------------------------------------------

def is_prime(k: int) -> bool:
    if k < 2:
        return False
    if k < 4:
        return True
    if k % 2 == 0:
        return False
    i = 3
    while i * i <= k:
        if k % i == 0:
            return False
        i += 2
    return True


def binomial(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    num = 1
    for i in range(r):
        num = num * (n - i) // (i + 1)
    return num


def classical_binomial_prime_set(n_bits: int):
    """Return the sorted list of n in [0, 2**n_bits - 1] with C(n,2) prime."""
    marked = []
    for n in range(2 ** n_bits):
        c = binomial(n, 2)
        if is_prime(c):
            marked.append(n)
    return marked


N_BITS = 4
SEARCH_SPACE = 2 ** N_BITS  # 16

MARKED = classical_binomial_prime_set(N_BITS)
print(f"Classical search space: n in [0, {SEARCH_SPACE - 1}] ({N_BITS} qubits)")
print(f"Classical marked set (C(n,2) prime): {MARKED}")
assert MARKED == [3], f"expected exactly n=3 to be marked, got {MARKED}"


# ---------------------------------------------------------------------------
# Quantum circuit: Grover search for the marked n.
# ---------------------------------------------------------------------------

def build_oracle(n_bits: int, marked_values):
    """Phase-flip oracle marking each value in `marked_values` (each in
    [0, 2**n_bits - 1]), via X-sandwiched multi-controlled-Z gates."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_bits}b")[::-1]  # qubit 0 = LSB
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_bits == 1:
            qc.z(0)
        else:
            mcz = MCXGate(n_bits - 1)  # placeholder, replaced below
            # Multi-controlled Z = H on target, MCX, H on target.
            target = n_bits - 1
            controls = list(range(n_bits - 1))
            qc.h(target)
            qc.append(MCXGate(len(controls)), controls + [target])
            qc.h(target)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int):
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    target = n_bits - 1
    controls = list(range(n_bits - 1))
    qc.h(target)
    qc.append(MCXGate(len(controls)), controls + [target])
    qc.h(target)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def build_grover_circuit(n_bits: int, marked_values, iterations: int):
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, marked_values)
    diffuser = build_diffuser(n_bits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_bits))
        qc.append(diffuser.to_instruction(), range(n_bits))

    qc.measure(range(n_bits), range(n_bits))
    return qc


num_iterations = max(1, round((math.pi / 4) * math.sqrt(SEARCH_SPACE / len(MARKED))))
print(f"Grover iterations used: {num_iterations}")

circuit = build_grover_circuit(N_BITS, MARKED, num_iterations)

simulator = AerSimulator()
compiled = transpile(circuit, simulator)
shots = 4096
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order: classical bit string is c[n_bits-1] ... c[0], with
# qubit 0 as LSB per build_oracle's convention -- so int(bitstring, 2)
# directly recovers n as encoded on the qubits.
counts_by_n = {int(bitstring, 2): freq for bitstring, freq in counts.items()}
best_n = max(counts_by_n, key=counts_by_n.get)
best_prob = counts_by_n[best_n] / shots

print(f"Measurement counts (by n): {counts_by_n}")
print(f"Most frequent measured n = {best_n} with probability {best_prob:.3f}")

expected_n = MARKED[0]
success = (best_n == expected_n) and (best_prob > 0.8)

print(f"Classical answer: n = {expected_n} (the unique n in [0,15] with C(n,2) prime)")
print(f"Quantum (Grover) result: n = {best_n}, amplified probability = {best_prob:.3f}")

if success:
    print("PASS")
else:
    print("FAIL")
