"""
Erdos problem #254 -- quantum-testable lane (limitation noted below).

Source check (read-only clone at /home/user/manman4/erdosproblems):
  data/problems.yaml, entry "number: \"254\"" (line ~4176) reads:
    prize: no
    status: open, last_update 2025-08-31
    oeis: ["N/A"]
    tags: ["number theory"]
  No OEIS sequence id is recorded for problem #254 in this dataset, and no
  separate per-problem statement file exists in the clone (searched for any
  file containing "254" under the repo; none found). So there is no OEIS
  sequence to derive a property from, and per the task instructions ("If
  ... no OEIS id ... write the script anyway with your best honest attempt,
  note the limitation clearly") this script does NOT claim to test anything
  specific to problem #254's actual open conjecture. That conjecture's
  content is unknown to this script.

LIMITATION: what follows is a best-honest-attempt substitute, not a test of
problem #254 itself. Its tag is "number theory", so the substitute property
is a small, genuinely finite/computable number-theory search: primality
over a small finite range. This has real mathematical content and is
verified classically from first principles (trial division, no external
data, no OEIS lookup), but it is NOT derived from problem #254's statement,
because that statement / OEIS id is not available in the read-only clone
used here.

Property tested: "which integer n in [0, 7] is the LARGEST prime" (N = 8,
3 qubits, single marked item). Classical answer (trial division, computed
below): the primes in [0,7] are {2, 3, 5, 7}, so the largest is 7.
(Note: an earlier draft of this script marked ALL primes at once (M=4 out
of N=8, i.e. M/N = 1/2); that is a known Grover degenerate case -- when
exactly half the space is marked the state is already at the theta=45deg
equilibrium and a Grover iteration only adds a global-ish phase, so
probability of the marked subspace stays at 0.5 forever and never
amplifies. Verified numerically during development (statevector inspection
showed the post-oracle and post-diffusion amplitudes identical). Switched
to a single-marked-item instance to get genuine amplification.)

Quantum method: Grover's search algorithm on 3 qubits (search space size
N = 8) with a single marked item (M = 1, the largest prime, 7). The oracle
flips the phase of |7>; the diffusion operator amplifies it. With
M/N = 1/8, theta = arcsin(sqrt(1/8)) ~= 0.361 rad, and the optimal number
of iterations round(pi/(4*theta) - 0.5) = 2 brings the marked-state
probability close to 1 on the ideal simulator, recovering n = 7 as the
overwhelmingly most likely measurement outcome -- verified against the
classical trial-division answer.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: which n in [0, N-1] are prime, by trial division.
# ---------------------------------------------------------------------------
def is_prime_trial_division(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 3
N = 2 ** N_QUBITS  # 8

classical_primes = sorted(n for n in range(N) if is_prime_trial_division(n))
largest_prime = max(classical_primes)
print(f"Classical (trial division) primes in [0, {N - 1}]: {classical_primes}")
print(f"Classical largest prime in [0, {N - 1}]: {largest_prime}")


# ---------------------------------------------------------------------------
# 2. Grover oracle: flips the phase of each basis state |n> with n prime.
# ---------------------------------------------------------------------------
def apply_oracle(qc: QuantumCircuit, marked_values, n_qubits: int) -> None:
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")  # MSB..LSB matches qubit n-1..0
        # Flip 0-bits to 1 so a multi-controlled Z fires exactly on `value`.
        for i, b in enumerate(bits):
            qubit = n_qubits - 1 - i
            if b == "0":
                qc.x(qubit)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i, b in enumerate(bits):
            qubit = n_qubits - 1 - i
            if b == "0":
                qc.x(qubit)


def apply_diffuser(qc: QuantumCircuit, n_qubits: int) -> None:
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


# ---------------------------------------------------------------------------
# 3. Build and run the Grover circuit for the requested number of iterations.
# ---------------------------------------------------------------------------
marked_values = [largest_prime]
M = len(marked_values)
theta = np.arcsin(np.sqrt(M / N))
optimal_iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
print(f"M={M} marked out of N={N}; optimal Grover iterations = {optimal_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(optimal_iterations):
    apply_oracle(qc, marked_values, N_QUBITS)
    apply_diffuser(qc, N_QUBITS)
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB..LSB matching qubit order (q_{n-1}...q_0).
most_common_bitstring = max(counts, key=counts.get)
most_common_value = int(most_common_bitstring, 2)
most_common_fraction = counts[most_common_bitstring] / shots

print(f"Grover most-frequent measured value: {most_common_value} "
      f"({most_common_fraction:.3f} of {shots} shots)")
print(f"Shot distribution: {counts}")

# Genuine amplification check: the marked item must dominate the
# distribution (well above the 1/N = 12.5% uniform baseline), and it must
# equal the classically-computed largest prime.
verified = (
    most_common_value == largest_prime
    and most_common_fraction > 0.5
)
print("PASS" if verified else "FAIL")
if not verified:
    raise SystemExit(1)
