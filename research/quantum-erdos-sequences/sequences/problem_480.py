"""
Erdos problem #480 -- quantum-testable sequence lane.

Source check (read-only clone at /home/user/manman4/erdosproblems):
  data/problems.yaml, entry `number: "480"` reads:
      oeis: ["N/A"]
      tags: ["number theory"]
      status: proved (Lean)
  There is no OEIS sequence id attached to problem 480 in the dataset, and no
  problem-statement text file for it in the repo (only the yaml metadata
  record exists). Per the task instructions ("if after reasonable effort no
  genuine quantum circuit can be constructed for this problem's sequence ...
  write the script anyway with your best honest attempt, note the limitation
  clearly"), this script does NOT fabricate an OEIS id or a problem-480-
  specific property -- there isn't one to derive from the source data.

LIMITATION: this script does not test anything specific to Erdos problem
#480. Instead, as an honest best-effort fallback, it builds and runs a real
Grover-search quantum circuit for a genuine, independently-checkable, small
finite number-theoretic property in the same tag ("number theory") that
problem 480 carries: "which n in [0, 15] are prime?". The classical answer
is computed from first principles (trial division) in this script, and the
quantum circuit's measurement distribution is compared against it. This is
disclosed as a substitute test, not a verification of problem 480 itself.

Property under test: primality of n for n in {0, 1, ..., 15} (4-bit search
space), i.e. membership of n in the set {2, 3, 5, 7, 11, 13}.

Circuit: Grover search over 4 qubits with an oracle built from a classically
precomputed marked-set (phase flip on |n> for each prime n), the standard
diffusion operator, and enough Grover iterations for the 6-out-of-16 marked
fraction. Run on AerSimulator (ideal, no noise).
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_primes(limit: int) -> list[int]:
    """Trial-division primality test, computed from first principles."""
    primes = []
    for n in range(limit):
        if n < 2:
            continue
        is_prime = True
        for d in range(2, int(math.isqrt(n)) + 1):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    return primes


def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle: multi-controlled Z on each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        # Flip qubits that are 0 in this basis state so the controlled-Z
        # fires exactly on |m>.
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


def run_grover(n_qubits: int, marked: list[int], shots: int = 4096):
    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    n_items = 2 ** n_qubits
    n_marked = len(marked)
    # Optimal number of Grover iterations for this marked fraction.
    theta = math.asin(math.sqrt(n_marked / n_items))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main() -> bool:
    n_qubits = 4
    limit = 2 ** n_qubits  # search space [0, 15]

    marked = classical_primes(limit)
    print(f"Classical (trial division) primes in [0, {limit - 1}]: {marked}")

    counts, iterations = run_grover(n_qubits, marked, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # Convert bitstrings (Qiskit: qubit 0 is rightmost) to integers.
    int_counts = Counter()
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        int_counts[value] += c

    total_shots = sum(int_counts.values())
    marked_shots = sum(int_counts[v] for v in marked)
    marked_fraction = marked_shots / total_shots

    # Take the top-len(marked) most frequent outcomes as the circuit's
    # "found" set and compare against the classical marked set.
    top_values = [v for v, _ in int_counts.most_common(len(marked))]
    quantum_set = set(top_values)
    classical_set = set(marked)

    print(f"Marked-outcome measurement fraction: {marked_fraction:.3f} "
          f"(uniform-random baseline would be {len(marked) / limit:.3f})")
    print(f"Top {len(marked)} measured outcomes: {sorted(top_values)}")
    print(f"Classical prime set:                 {sorted(classical_set)}")

    passed = (
        quantum_set == classical_set
        and marked_fraction > (len(marked) / limit) * 1.5
    )

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
