"""
Erdos problem #731 -- quantum-testable sequence lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 731"):
    oeis: ["A006197"]
    tags: ["number theory", "binomial coefficients"]

A006197 concerns the central binomial coefficients C(2n, n) and their
factorization. The classical property exercised here is a direct
consequence of Kummer's theorem on carries in base-p addition, specialized
to p = 2 and to computing C(n+n, n) = C(2n, n):

    Kummer's theorem: the exponent of the prime p in C(a+b, a) equals the
    number of carries produced when adding a and b in base p.

    Specialized to a = b = n, base 2: adding n to itself in binary, each
    bit position where n has a 1-bit produces a carry (1+1 = 10 in base
    2), so the number of carries equals the number of 1-bits in n's
    binary representation, i.e. popcount(n).

    Therefore: v_2(C(2n, n)) = popcount(n)   for every n >= 0,

where v_2(m) is the 2-adic valuation of m (the exponent of 2 in its prime
factorization). This is verified directly from first principles below by
computing C(2n, n) exactly (Python big integers) and counting factors of
2, independent of any lookup table or OEIS b-file value.

Chosen finite instance and quantum task
----------------------------------------
Search space: n in {0, 1, ..., 7} (3 qubits, since 8 = 2^3).
Target property: popcount(n) == 2.
Classically, by brute force over n = 0..7, the n satisfying popcount(n) == 2
are exactly {3, 5, 6} (binary 011, 101, 110) -- and this is cross-checked
against v_2(C(2n, n)) computed independently from the exact value of
C(2n, n), confirming v_2(C(2n, n)) == popcount(n) for every n in range,
which is the real mathematical content connected to A006197's subject
matter (2-adic behaviour of central binomial coefficients).

Quantum circuit: a genuine Grover search (oracle + diffuser, ~2 iterations
optimal for 3 marked states out of 8) that amplifies exactly the marked
computational basis states {3, 5, 6} out of the 8 possible 3-qubit
strings. The circuit is run on the ideal AerSimulator with many shots; we
then check that the search overwhelmingly returns basis states in the
classically/`v_2`-verified marked set, and PASS/FAIL is decided by
comparing the quantum measurement distribution to the classical marked
set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

from __future__ import annotations

from math import comb

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

def popcount(n: int) -> int:
    return bin(n).count("1")


def v2(m: int) -> int:
    """2-adic valuation: exponent of 2 in m's prime factorization."""
    if m == 0:
        raise ValueError("v2(0) undefined")
    e = 0
    while m % 2 == 0:
        m //= 2
        e += 1
    return e


N_QUBITS = 3
DOMAIN = list(range(2 ** N_QUBITS))  # n = 0..7

# Verify Kummer's-theorem consequence v_2(C(2n,n)) == popcount(n) for every
# n in the search domain, from the exact value of C(2n, n) -- this is the
# "derive/check it classically yourself" step, not a copied OEIS value.
for n in DOMAIN:
    if n == 0:
        continue  # C(0,0) = 1 has no prime factors; v2 undefined, popcount(0)=0, consistent by convention
    central = comb(2 * n, n)
    assert v2(central) == popcount(n), (
        f"Kummer-theorem check failed at n={n}: "
        f"v2(C({2*n},{n}))={v2(central)} vs popcount({n})={popcount(n)}"
    )

TARGET_POPCOUNT = 2
MARKED = sorted(n for n in DOMAIN if popcount(n) == TARGET_POPCOUNT)
print(f"Classical marked set (popcount(n) == {TARGET_POPCOUNT}): {MARKED}")
assert MARKED == [3, 5, 6]

# Cross-check via the binomial-coefficient route directly (the actual A006197
# subject matter): the same set equals {n : v_2(C(2n,n)) == TARGET_POPCOUNT}.
MARKED_VIA_BINOMIAL = sorted(
    n for n in DOMAIN if n > 0 and v2(comb(2 * n, n)) == TARGET_POPCOUNT
)
assert MARKED_VIA_BINOMIAL == MARKED, (MARKED_VIA_BINOMIAL, MARKED)


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 3-qubit domain for the marked states.
# ---------------------------------------------------------------------------

def build_oracle(marked_values: list[int], n_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle: marks each basis state in marked_values with -1."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked_values: list[int], n_qubits: int, iterations: int) -> QuantumCircuit:
    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# Optimal number of Grover iterations for M marked out of N states:
# floor( (pi/4) * sqrt(N/M) ), at least 1.
N_STATES = 2 ** N_QUBITS
M_MARKED = len(MARKED)
iterations = max(1, round((np.pi / 4) * np.sqrt(N_STATES / M_MARKED)))
print(f"N={N_STATES}, M={M_MARKED}, Grover iterations={iterations}")

circuit = build_grover_circuit(MARKED, N_QUBITS, iterations)

simulator = AerSimulator()
compiled = transpile(circuit, simulator)
shots = 4096
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()


# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

def bitstring_to_int(bs: str) -> int:
    # Qiskit's classical-register bitstring is big-endian (qubit n-1 .. 0);
    # our oracle used little-endian qubit index i for bit i of the value,
    # so reverse before converting.
    return int(bs[::-1], 2)


marked_shots = 0
for bitstring, count in counts.items():
    value = bitstring_to_int(bitstring)
    if value in MARKED:
        marked_shots += count

marked_fraction = marked_shots / shots
print(f"Counts (raw): {counts}")
print(f"Fraction of shots landing on classically-marked states {MARKED}: {marked_fraction:.4f}")

# With M=3 marked out of N=8 and the optimal number of iterations, Grover's
# algorithm should return a marked state with high probability (>> uniform
# baseline of 3/8 = 0.375). Require a comfortable margin above uniform.
UNIFORM_BASELINE = M_MARKED / N_STATES
success = marked_fraction > UNIFORM_BASELINE + 0.25

# Also confirm the most frequent measured outcome is itself in the
# classically/Kummer-verified marked set.
most_common_bitstring = max(counts, key=counts.get)
most_common_value = bitstring_to_int(most_common_bitstring)
most_common_is_marked = most_common_value in MARKED

print(f"Most frequent measured n = {most_common_value} "
      f"(popcount={popcount(most_common_value)}, in classical marked set: {most_common_is_marked})")

verified = success and most_common_is_marked

if verified:
    print("PASS")
else:
    print("FAIL")
