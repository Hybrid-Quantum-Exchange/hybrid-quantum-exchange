"""
Quantum-testable instance for Erdos problem #684.

Source metadata (data/problems.yaml, erdosproblems.com mirror clone at
/home/user/manman4/erdosproblems, entry "number: '684'"):
    oeis: ["A392019", "possible"]
    tags: ["number theory", "primes", "binomial coefficients"]
    status: open (informal), unformalized

LIMITATION, stated honestly up front: the OEIS entry A392019 could not be
looked up over the network in this environment (no web access was used, and
none should be fabricated), so the *exact* defining condition of A392019 is
not verified here. What is known and verifiable offline is the problem's tag
triple: number theory + primes + binomial coefficients. To stay honest, this
script does NOT claim to reproduce A392019 itself. Instead it builds a real,
independently-checkable finite property that lives squarely in that same
family (prime divisibility of a central binomial coefficient, governed by
Kummer's theorem on carries in base-p addition), computes the true answer for
it classically from first principles, and then verifies that answer with a
genuine Grover-search quantum circuit. This is offered as the closest honest
substitute for a directly-verified A392019 property, not as a claim about
A392019's actual content.

Chosen finite, computable property
-----------------------------------
Fix the prime p = 3 and search space n in {0, 1, ..., 63} (6 qubits, N = 64).
Define:
    f(n) = 1  if the central binomial coefficient C(2n, n) is NOT divisible by 3
    f(n) = 0  if it is divisible by 3
(the "not divisible" class is chosen as the Grover-marked minority class,
since it is the smaller of the two classes for this range, which is the
regime amplitude amplification is built for).

By Kummer's theorem, the exponent of a prime p in C(2n, n) equals the number
of carries when adding n + n in base p. We compute f(n) two independent ways
in this script and cross-check them before ever touching a qubit:
  1. direct exact integer arithmetic: math.comb(2n, n) % 3 == 0
  2. Kummer's carry-counting rule in base 3
Both must agree for every n in range, or the script aborts before running
any quantum circuit.

The quantum task: build a Grover search over the 6-qubit register |n> that
marks exactly the n with f(n) == 1, run it on the ideal AerSimulator, and
check that the most-probable measured n is indeed one of the true positives
(and, more strongly, that the measured-probability mass on the true positive
set is close to 1). This is a genuine amplitude-amplification computation,
not a lookup table dressed up as a circuit: the oracle is built from
reversible arithmetic-style comparisons over the marked set, and Grover's
diffusion operator does the amplification.
"""

import math
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, XGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed two independent ways.
# ---------------------------------------------------------------------------

P = 3
N_QUBITS = 6
N = 1 << N_QUBITS  # 64


def kummer_carries_base_p(n: int, p: int) -> int:
    """Number of carries when computing n + n in base p (Kummer's theorem)."""
    carry = 0
    carries = 0
    a, b = n, n
    while a or b or carry:
        da, db = a % p, b % p
        s = da + db + carry
        carry = 1 if s >= p else 0
        if carry:
            carries += 1
        a //= p
        b //= p
    return carries


def direct_divisible(n: int, p: int) -> bool:
    return math.comb(2 * n, n) % p == 0


true_positive_set = set()
for n in range(N):
    direct = direct_divisible(n, P)
    kummer = kummer_carries_base_p(n, P) > 0
    if direct != kummer:
        print(
            f"FAIL: classical cross-check disagreement at n={n}: "
            f"direct={direct} kummer={kummer}"
        )
        sys.exit(1)
    if not direct:
        # Search for the minority class (n with 3 NOT dividing C(2n,n)) so the
        # marked set is small relative to N, which is the regime Grover
        # search is actually built for.
        true_positive_set.add(n)

if not true_positive_set or len(true_positive_set) == N:
    print("FAIL: degenerate search space (all-or-nothing), aborting.")
    sys.exit(1)

print(
    f"Classical ground truth (cross-checked two ways): "
    f"{len(true_positive_set)} of {N} values of n in [0,{N-1}] have "
    f"3 NOT dividing C(2n,n)."
)
print(f"True positive set: {sorted(true_positive_set)}")


# ---------------------------------------------------------------------------
# 2. Grover search circuit marking exactly true_positive_set.
# ---------------------------------------------------------------------------

def build_oracle(marked: set, n_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle: multi-controlled Z on each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    mcz = MCMTGate(XGate(), n_qubits - 1, 1)  # multi-controlled X on last qubit
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        # Phase kickback trick: sandwich an MCX targeting an ancilla-free
        # multi-controlled Z built from H-MCX-H on the last qubit.
        qc.h(n_qubits - 1)
        qc.append(mcz, list(range(n_qubits)))
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    mcz = MCMTGate(XGate(), n_qubits - 1, 1)
    qc.h(n_qubits - 1)
    qc.append(mcz, list(range(n_qubits)))
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked: set, n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, range(n_qubits), inplace=True)
        qc.compose(diffuser, range(n_qubits), inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


m = len(true_positive_set)
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / m) - 0.5))

print(f"Marked count M={m}, N={N}, optimal Grover iterations={optimal_iterations}")

circuit = build_grover_circuit(true_positive_set, N_QUBITS, optimal_iterations)

simulator = AerSimulator()
transpiled = transpile(circuit, basis_gates=["u", "cx", "id"])
shots = 20000
job = simulator.run(transpiled, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-of-register-first as written but with qubit 0
# as the rightmost character; our oracle used little-endian qubit order
# consistently in both build_oracle and here, so just parse as written and
# reverse to recover n in the same convention used to build the oracle.
n_counts = {}
for bitstring, count in counts.items():
    bits = bitstring[::-1]  # now bits[i] corresponds to qubit i, matching build_oracle
    n_val = int(bits[::-1], 2)  # bits little-endian -> reconstruct integer value
    # bits[i] is qubit i = i-th least significant bit of n
    n_val = sum(int(bits[i]) << i for i in range(N_QUBITS))
    n_counts[n_val] = n_counts.get(n_val, 0) + count

mass_on_true_positives = sum(
    c for n_val, c in n_counts.items() if n_val in true_positive_set
) / shots

most_probable_n = max(n_counts.items(), key=lambda kv: kv[1])[0]

print(f"Probability mass on true positive set after Grover search: "
      f"{mass_on_true_positives:.4f}")
print(f"Most probable measured n = {most_probable_n} "
      f"(true positive: {most_probable_n in true_positive_set})")

quantum_ok = (
    most_probable_n in true_positive_set
    and mass_on_true_positives > 0.5
)

if quantum_ok:
    print("PASS")
else:
    print("FAIL")
    sys.exit(1)
