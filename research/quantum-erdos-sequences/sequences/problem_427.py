"""
Erdos problem #427 (erdosproblems.com), quantum-testable lane.

Source metadata (from manman4/erdosproblems data/problems.yaml, entry
`number: "427"`): prize "no", status "proved (Lean)", tags ["number theory",
"primes"], oeis: ["N/A"]. There is NO OEIS sequence id attached to this
problem -- the listed value is the literal string "N/A", and no
problem-specific description file exists in the read-only clone either.
Per the task instructions, an honest attempt is written here rather than a
fabricated sequence property, and the limitation is noted explicitly: this
script does NOT test membership in any actual Erdos-problem-427 OEIS
sequence, because none exists in the source data.

Given the problem's tags ("number theory", "primes"), the best small,
finite, computable stand-in property genuinely in that spirit is primality
testing via search:

    Property tested: among the integers 0..63 (6-bit index, N = 64), find
    every n such that n is prime (trial division, the textbook definition,
    which is also literally OEIS A000040's defining property -- "the
    primes"). This is a real, well-defined, finite decision property with
    a small search space (64 candidates, 6 qubits), exactly the flavor of
    "primes"-tagged Erdos number-theory problems.

Classical answer (computed here in the script, from first principles, by
trial division -- no external primality library used): the set of primes
in [0, 63] is enumerated by is_prime_classical() below and used both to
build the Grover oracle and as the ground truth the quantum result is
checked against.

Quantum approach: Grover's search algorithm on 6 qubits encoding n in
0..63. The oracle marks (multi-controlled phase flip) exactly the basis
states whose integer value is prime by the classical trial-division check;
the diffusion operator amplifies those marked amplitudes; the number of
Grover iterations is chosen from the classical count of marked (prime)
states. The ideal AerSimulator counts are checked against the classical
enumeration: every prime must be among the top-measured outcomes and must
outcount every non-prime.

Limitation: this is a faithful small Grover search for primality, not a
test of a specific documented OEIS integer sequence tied to problem #427
(none is listed in the source data for this problem). ran_ok and
verified_against_classical are reported honestly for what this script
actually checks.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_BITS = 6            # n in 0..63
N = 1 << N_BITS         # 64


def is_prime_classical(n: int) -> bool:
    """Trial division primality test, first principles, no libraries."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max: int):
    return [n for n in range(n_max) if is_prime_classical(n)]


def build_oracle_phase(n_bits: int, marked_values, total_qubits: int) -> QuantumCircuit:
    """Multi-controlled-Z oracle: applies -1 phase to each marked basis state."""
    qc = QuantumCircuit(total_qubits, name="oracle")
    for v in marked_values:
        v_bits = format(v, f"0{n_bits}b")[::-1]  # qubit0 = LSB
        pattern = list(v_bits)

        zero_qubits = [q for q, b in enumerate(pattern) if b == "0"]
        for q in zero_qubits:
            qc.x(q)

        qc.h(total_qubits - 1)
        if total_qubits - 1 > 0:
            qc.append(MCXGate(total_qubits - 1), list(range(total_qubits - 1)) + [total_qubits - 1])
        else:
            qc.z(0)
        qc.h(total_qubits - 1)

        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(total_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(total_qubits, name="diffuser")
    qc.h(range(total_qubits))
    qc.x(range(total_qubits))
    qc.h(total_qubits - 1)
    if total_qubits - 1 > 0:
        qc.append(MCXGate(total_qubits - 1), list(range(total_qubits - 1)) + [total_qubits - 1])
    else:
        qc.z(0)
    qc.h(total_qubits - 1)
    qc.x(range(total_qubits))
    qc.h(range(total_qubits))
    return qc


def run():
    total_qubits = N_BITS

    marked = classical_primes(N)
    assert marked, "classical search found no primes — instance is degenerate"
    print(f"Classical answer: primes in 0..{N-1} (trial division): {marked}")

    num_marked = len(marked)
    search_space_size = N

    theta = math.asin(math.sqrt(num_marked / search_space_size))
    optimal_iters = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle_phase(N_BITS, marked, total_qubits)
    diffuser = build_diffuser(total_qubits)

    qc = QuantumCircuit(total_qubits, total_qubits)
    qc.h(range(total_qubits))
    for _ in range(optimal_iters):
        qc.append(oracle.to_instruction(), range(total_qubits))
        qc.append(diffuser.to_instruction(), range(total_qubits))
    qc.measure(range(total_qubits), range(total_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4096, seed_simulator=42).result()
    counts = result.get_counts()

    def bitstring_to_value(bs: str):
        bits = bs[::-1]  # index 0 == qubit0 (LSB)
        return int(bits[::-1], 2)

    value_counts = {}
    for bs, c in counts.items():
        v = bitstring_to_value(bs)
        value_counts[v] = value_counts.get(v, 0) + c

    sorted_values = sorted(value_counts.items(), key=lambda kv: -kv[1])
    print("\nTop measured n values by frequency:")
    for v, c in sorted_values[:num_marked + 5]:
        marker = " <-- PRIME" if v in set(marked) else ""
        print(f"  {v}: {c}{marker}")

    marked_set = set(marked)
    top_n = sorted_values[:num_marked]
    top_values = set(v for v, _ in top_n)

    all_marked_on_top = top_values == marked_set
    min_marked_count = min(value_counts.get(v, 0) for v in marked_set)
    max_unmarked_count = max(
        (c for v, c in value_counts.items() if v not in marked_set), default=0
    )
    dominance_ok = min_marked_count > max_unmarked_count

    verified = all_marked_on_top and dominance_ok

    print(f"\nGrover iterations used: {optimal_iters}")
    print(f"Marked (prime) values: {sorted(marked_set)}")
    print(f"Top-{num_marked} measured values match prime set exactly: {all_marked_on_top}")
    print(f"Every marked value outcounts every unmarked value: {dominance_ok}")

    if verified:
        print("\nPASS")
    else:
        print("\nFAIL")

    return verified


if __name__ == "__main__":
    ok = run()
    if not ok:
        raise SystemExit(1)
