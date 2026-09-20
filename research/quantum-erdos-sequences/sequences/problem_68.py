"""
Erdos problem #68 (https://www.erdosproblems.com/68) -- open, no prize,
tags: number theory, irrationality.

OEIS id used: A331373, "Decimal expansion of Sum_{k>=2} 1/(k! - 1)".
Erdos asked whether this constant,

    S = sum_{k=2}^{infinity} 1/(k! - 1) = 1/1 + 1/5 + 1/23 + 1/119 + ...
      = 1.25349875569995347164336093790579894036923220833201...

is irrational (still open). The problem itself (irrationality of an infinite
sum) is not something a small quantum circuit can decide -- it is not a
finite, computable question. What *is* finite and computable is: "what is
the value of the n-th decimal digit (term) of A331373?" This is exactly the
quantity the OEIS b-file enumerates, term by term, e.g. terms 1..16 are
1,2,5,3,4,9,8,7,5,5,6,9,9,9,5,3 (the digits of 1.253498755699953...).

Classical property tested here
-------------------------------
Let a(n) denote the n-th term of A331373 (1-indexed; a(1) is the units digit
"1", a(2) is the first digit after the decimal point, etc.). We fix n = 8
and compute a(8) classically from first principles, with no OEIS lookup: we
sum 1/(k!-1) for k = 2..K with K large enough that truncation error is far
below the precision needed, using Python's arbitrary-precision Decimal type,
then read off the 8th digit of the resulting decimal expansion. This is
cross-checked in-script against the known initial digit list quoted above
(itself just a consequence of the same computation, not trusted blindly --
we recompute S ourselves).

Quantum circuit
----------------
We then pose this digit lookup as a Grover search: over the 4-qubit space
{0,...,15} (10 valid decimal digit values 0-9, padded to a 16-dimensional
computational basis), find the unique value v such that v == a(8). The
oracle is built directly from the classically-computed target digit's binary
representation (a multi-controlled Z, flanked by X gates on the 0-bits), so
the circuit "knows" only the marked bitstring, exactly as in a standard
Grover's-algorithm instance. We run the optimal number of Grover iterations
for N=16, sample the ideal AerSimulator, and check that the most frequently
measured 4-bit string decodes to the same digit a(8) computed classically.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

from decimal import Decimal, getcontext

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of a(n) for A331373, from first principles.
# ---------------------------------------------------------------------------

def decimal_expansion_of_S(num_digits: int, K: int = 60) -> str:
    """Return the first `num_digits` digits (as a string, no decimal point)
    of S = sum_{k=2}^{K} 1/(k! - 1), computed to ample precision.

    K = 60 is far more than enough: 60! is astronomically larger than the
    precision we need, so summing k=2..60 already matches the true infinite
    sum to well beyond `num_digits` significant digits (the terms shrink
    faster than 1/k!, which itself is already ~10^-82 at k=60).
    """
    getcontext().prec = num_digits + 30  # generous guard digits
    total = Decimal(0)
    fact = Decimal(1)
    for k in range(2, K + 1):
        fact *= k  # fact = k!
        total += Decimal(1) / (fact - 1)

    # total looks like "1.253498755699953..."; strip the decimal point to
    # get the raw digit string "1253498755699953...".
    s = format(total, "f")
    int_part, frac_part = s.split(".")
    digits = int_part + frac_part
    return digits[:num_digits]


N_TERMS = 12
digits = decimal_expansion_of_S(N_TERMS)

# Sanity check against the independently known initial terms of A331373
# (OEIS b-file values for offset 1..16): 1,2,5,3,4,9,8,7,5,5,6,9,9,9,5,3
_known_prefix = "125349875569"
assert digits == _known_prefix, (
    f"classical computation of A331373 digits disagreed: got {digits!r}, "
    f"expected prefix {_known_prefix!r}"
)

N = 8  # 1-indexed term to search for
target_digit = int(digits[N - 1])  # a(8), computed purely classically
print(f"Classical computation: a(1..{N_TERMS}) of A331373 = {digits}")
print(f"Target term a({N}) = {target_digit}")


# ---------------------------------------------------------------------------
# 2. Grover search over 4 qubits (16 basis states) for the marked digit.
# ---------------------------------------------------------------------------

NUM_QUBITS = 4
SPACE_SIZE = 2 ** NUM_QUBITS  # 16


def build_oracle(target: int, num_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle marking the single basis state |target>."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    bits = format(target, f"0{num_qubits}b")
    # Flip qubits that should be 0 in the target, so the marked state
    # becomes |11...1>, apply a multi-controlled Z, then flip back.
    for i, b in enumerate(reversed(bits)):
        if b == "0":
            qc.x(i)
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    for i, b in enumerate(reversed(bits)):
        if b == "0":
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


# Optimal number of Grover iterations for 1 marked item out of SPACE_SIZE.
num_iterations = max(1, round((np.pi / 4) * np.sqrt(SPACE_SIZE)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

oracle = build_oracle(target_digit, NUM_QUBITS)
diffuser = build_diffuser(NUM_QUBITS)

for _ in range(num_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first matching classical register order
# (qubit 0 = rightmost char); convert back to an integer consistently.
most_likely_bitstring = max(counts, key=counts.get)
measured_value = int(most_likely_bitstring, 2)
measured_probability = counts[most_likely_bitstring] / shots

print(f"Grover search ran {num_iterations} iteration(s) over {SPACE_SIZE} basis states.")
print(f"Most frequent measurement: {most_likely_bitstring} -> {measured_value} "
      f"(probability {measured_probability:.3f} over {shots} shots)")
print(f"Classical target value: {target_digit}")

verified = measured_value == target_digit and measured_probability > 0.5

if verified:
    print("PASS")
else:
    print("FAIL")
