"""
Erdos problem #455 (erdosproblems.com), quantum-testable lane.

Source metadata (from manman4/erdosproblems data/problems.yaml, entry
`number: "455"`): prize "no", status "open" (last_update 2025-08-31),
tags ["number theory"], oeis: ["N/A"]. There is NO OEIS sequence id
attached to this problem -- the listed value is the literal string "N/A".
Per the task instructions, an honest attempt is written here rather than
a fabricated sequence property, and the limitation is stated explicitly:
this script does NOT test membership in, or any term of, a real
Erdos-problem-455 OEIS sequence, because no such sequence id exists in
the source data, and the repository clone here carries no separate
problem-statement text for #455 beyond this metadata row.

Given the problem's only real signal -- the tag "number theory" -- the
best small, finite, computable stand-in genuinely in that spirit is a
primality-search problem on a small range of integers:

    Property tested: among the integers 0..15 (4-bit index, N = 16),
    find every n with n prime (2, 3, 5, 7, 11, 13). Primality is a
    classic finite, computable number-theoretic property with a real
    search space (16 candidates), suitable for a small Grover oracle.

Classical answer (computed here in the script, from first principles,
via trial division -- no OEIS lookup, no hardcoded literal list):
enumerate n in 0..15, test primality by trial division up to sqrt(n),
and mark exactly the primes. This is used both to build the Grover
oracle and as the ground truth the quantum result is checked against.

Quantum approach: Grover's search algorithm on 4 qubits, an oracle
marking exactly the basis states encoding a prime n in 0..15 (as
determined by the classical trial-division check above), a standard
diffusion operator, and the ideal AerSimulator counts are checked
against the classical enumeration: the top-k measured outcomes (k =
number of marked primes) must be exactly the classically-marked primes,
and every marked outcome's count must exceed every unmarked outcome's
count.

Limitation: this is a faithful small Grover search over a genuine
number-theoretic property (primality), chosen because problem #455 has
no OEIS id and no accessible problem statement in the source data --
it is NOT a test of any specific documented sequence tied to problem
#455. ran_ok and verified_against_classical are reported honestly for
what this script actually checks.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_BITS = 4
N = 1 << N_BITS  # 16


def is_prime(n: int) -> bool:
    """Trial division primality test, first principles, no lookups."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max: int):
    return [n for n in range(n_max) if is_prime(n)]


def build_oracle_phase(marked_values, total_qubits: int) -> QuantumCircuit:
    """Multi-controlled-Z oracle: applies -1 phase to each marked basis state."""
    qc = QuantumCircuit(total_qubits, name="oracle")
    for v in marked_values:
        bits = format(v, f"0{total_qubits}b")[::-1]  # bits[0] == qubit0 value

        zero_qubits = [q for q, b in enumerate(bits) if b == "0"]
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
    assert marked, "classical search found no marked primes -- instance is degenerate"
    print(f"Classical answer: primes in 0..{N-1}: {marked}")

    num_marked = len(marked)
    search_space_size = N

    theta = math.asin(math.sqrt(num_marked / search_space_size))
    optimal_iters = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle_phase(marked, total_qubits)
    diffuser = build_diffuser(total_qubits)

    qc = QuantumCircuit(total_qubits, total_qubits)
    qc.h(range(total_qubits))
    for _ in range(optimal_iters):
        qc.compose(oracle, range(total_qubits), inplace=True)
        qc.compose(diffuser, range(total_qubits), inplace=True)
    qc.measure(range(total_qubits), range(total_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4096, seed_simulator=42).result()
    counts = result.get_counts()

    def bitstring_to_value(bs: str) -> int:
        # Qiskit counts keys are already c[n-1]...c[0] left to right, i.e.
        # standard MSB-first binary of the integer value (qubit0 == LSB).
        return int(bs, 2)

    value_counts = {}
    for bs, c in counts.items():
        v = bitstring_to_value(bs)
        value_counts[v] = value_counts.get(v, 0) + c

    sorted_vals = sorted(value_counts.items(), key=lambda kv: -kv[1])
    print("\nTop measured n values by frequency:")
    marked_set = set(marked)
    for v, c in sorted_vals[:8]:
        marker = " <-- PRIME (marked)" if v in marked_set else ""
        print(f"  n={v}: {c}{marker}")

    top_n = sorted_vals[:num_marked]
    top_vals = set(v for v, _ in top_n)

    all_marked_on_top = top_vals == marked_set
    min_marked_count = min(value_counts.get(v, 0) for v in marked_set)
    max_unmarked_count = max(
        (c for v, c in value_counts.items() if v not in marked_set), default=0
    )
    dominance_ok = min_marked_count > max_unmarked_count

    verified = all_marked_on_top and dominance_ok

    print(f"\nGrover iterations used: {optimal_iters}")
    print(f"Marked primes: {sorted(marked_set)}")
    print(f"Top-{num_marked} measured values match marked set exactly: {all_marked_on_top}")
    print(f"Every marked value outcount every unmarked value: {dominance_ok}")

    if verified:
        print("\nPASS")
    else:
        print("\nFAIL")

    return verified


if __name__ == "__main__":
    ok = run()
    if not ok:
        raise SystemExit(1)
