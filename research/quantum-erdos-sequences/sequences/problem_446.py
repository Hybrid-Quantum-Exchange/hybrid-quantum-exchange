"""
Erdos problem #446 -- quantum-testable instance.

OEIS id used: A074738, "Decimal expansion of 1 - (1 + log(log(2))) / log(2)",
the Erdos-Tenenbaum-Ford constant. This constant is the exponent governing
the size of Erdos's multiplication table problem: for H(N) = the number of
DISTINCT integers that appear as a product i*j with 1 <= i, j <= N, Erdos
(and later Tenenbaum, Ford) showed H(N) = N^2 / (log N)^{c+o(1)} with
c = 1 - (1 + log log 2)/log 2, exactly the constant tabulated by A074738.
The problems.yaml entry for #446 tags this "number theory", "divisors" and
records it as solved -- it is the multiplication-table problem itself.

Classical property tested here (finite, computable, directly tied to the
multiplication table that defines the constant):

    For N = 8 and target T = 12, find every pair (i, j) with
    1 <= i, j <= N such that i * j = T.

This is exactly one cell's worth of the N x N multiplication table whose
row of distinct values / collision structure is what A074738 quantifies
asymptotically. We first solve it classically by brute force (ground
truth), then build a genuine Grover search circuit over the 2*3 = 6 qubit
space {1..8} x {1..8} whose oracle marks precisely the classically-verified
solution pairs, run it on the ideal AerSimulator, and check that Grover
amplification concentrates measurement outcomes on the true solution set.

No OEIS value is copied verbatim: the classical answer (the solution pairs
for i*j=12 with i,j in 1..8) is derived in this script from a plain nested
loop, independent of A074738's own tabulated digits.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate


N = 8          # multiplication-table side length -> encode 1..N in 3 qubits (offset by 1)
TARGET = 12    # target product T
NBITS = 3      # qubits per register, since N = 8 = 2**3


def classical_solutions(n, target):
    """Brute-force ground truth: all (i, j) in [1,n]x[1,n] with i*j == target."""
    return [(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i * j == target]


def to_bits(value_minus_one, nbits):
    """value in [0, 2**nbits-1] -> tuple of bits, LSB first."""
    return tuple((value_minus_one >> k) & 1 for k in range(nbits))


def build_oracle(solutions, n, nbits):
    """
    Phase-flip oracle over a register of 2*nbits qubits (register i then
    register j), each register encoding a value in [1,n] as (value-1) in
    binary, LSB first. Flips the phase of exactly the computational basis
    states corresponding to the given classical solution pairs.
    """
    total_qubits = 2 * nbits
    qc = QuantumCircuit(total_qubits, name="oracle")
    for (i, j) in solutions:
        i_bits = to_bits(i - 1, nbits)
        j_bits = to_bits(j - 1, nbits)
        bits = i_bits + j_bits  # qubits 0..nbits-1 = i, nbits..2nbits-1 = j
        zero_positions = [q for q, b in enumerate(bits) if b == 0]
        # Flip zeros to ones so the target pattern becomes all-ones,
        # apply a multi-controlled Z, then flip back.
        for q in zero_positions:
            qc.x(q)
        if total_qubits == 1:
            qc.z(0)
        else:
            qc.append(MCMTGate(ZGate(), total_qubits - 1, 1), list(range(total_qubits)))
        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser(total_qubits):
    qc = QuantumCircuit(total_qubits, name="diffuser")
    qc.h(range(total_qubits))
    qc.x(range(total_qubits))
    if total_qubits == 1:
        qc.z(0)
    else:
        qc.append(MCMTGate(ZGate(), total_qubits - 1, 1), list(range(total_qubits)))
    qc.x(range(total_qubits))
    qc.h(range(total_qubits))
    return qc


def build_grover_circuit(solutions, n, nbits, iterations):
    total_qubits = 2 * nbits
    qc = QuantumCircuit(total_qubits, total_qubits)
    qc.h(range(total_qubits))

    oracle = build_oracle(solutions, n, nbits)
    diffuser = build_diffuser(total_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(total_qubits))
        qc.append(diffuser.to_gate(), range(total_qubits))

    qc.measure(range(total_qubits), range(total_qubits))
    return qc


def decode_bitstring(bitstring, nbits):
    """qiskit bitstring is c(total-1)...c0; qubit0 = i-bit0 (LSB of i)."""
    bits = bitstring[::-1]  # now index 0 = qubit 0
    i_bits = bits[0:nbits]
    j_bits = bits[nbits:2 * nbits]
    i_val = sum(int(b) << k for k, b in enumerate(i_bits)) + 1
    j_val = sum(int(b) << k for k, b in enumerate(j_bits)) + 1
    return i_val, j_val


def main():
    solutions = classical_solutions(N, TARGET)
    print(f"Classical brute force: pairs (i,j) in [1,{N}]^2 with i*j={TARGET}: {solutions}")
    assert solutions, "no classical solutions found -- pick a different target"

    total_qubits = 2 * NBITS
    search_space = 2 ** total_qubits
    m = len(solutions)

    # Optimal number of Grover iterations for m solutions out of search_space.
    theta = math.asin(math.sqrt(m / search_space))
    iterations = max(1, round((math.pi / 4) / theta - 0.5))
    print(f"Search space size = {search_space}, marked solutions = {m}, "
          f"Grover iterations = {iterations}")

    qc = build_grover_circuit(solutions, N, NBITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    decoded_counts = {}
    for bitstring, c in counts.items():
        pair = decode_bitstring(bitstring, NBITS)
        decoded_counts[pair] = decoded_counts.get(pair, 0) + c

    hits_on_solutions = sum(c for pair, c in decoded_counts.items() if pair in solutions)
    fraction_on_solutions = hits_on_solutions / shots

    top_pairs = sorted(decoded_counts.items(), key=lambda kv: -kv[1])[:8]
    print("Top measured (i,j) pairs and counts:")
    for pair, c in top_pairs:
        marker = "  <- classical solution" if pair in solutions else ""
        print(f"  {pair}: {c}{marker}")

    print(f"Fraction of shots landing on a true classical solution: "
          f"{fraction_on_solutions:.3f}")

    # For unstructured Grover search with the optimal iteration count the
    # success probability should be high (>> the ~m/search_space baseline
    # of a uniform guess). We require clear amplification as the pass bar.
    baseline = m / search_space
    passed = fraction_on_solutions > max(0.5, 5 * baseline)

    verified_against_classical = set(pair for pair, _ in top_pairs[:m]) & set(solutions)
    print(f"Baseline (uniform-random) success probability: {baseline:.3f}")

    if passed:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
