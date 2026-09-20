"""
Erdos problem #402 -- quantum-testable sequence entry.

LIMITATION (read first): In the source data
(/home/user/manman4/erdosproblems/data/problems.yaml, entry "number: \"402\"")
problem #402 carries oeis: ["N/A"] and only the generic tag ["number theory"];
no OEIS sequence id and no problem statement text are available in that record.
Per the task instructions, this means there is no real sequence-specific
property of problem #402 that can honestly be derived and tested here -- doing
so would require fabricating a property with no connection to the actual
problem. Rather than fake that connection, this script is an honest fallback:
it builds and runs a genuine, self-contained quantum circuit on a small,
finite, computable number-theory property (the tag problem #402 does carry),
completely independent of any OEIS sequence, and is clearly labeled as such
rather than being presented as a verified property of problem #402 itself.

Chosen property (classical, finite, computable):
  "Which n in {0, 1, ..., 15} (4 qubits) are prime?"
  The primes in that range are {2, 3, 5, 7, 11, 13} (6 of 16), computed here
  from first principles by trial division -- no OEIS lookup, no hard-coded
  literal list taken on faith.

Quantum approach: Grover's search algorithm on 4 qubits, using a classically
-derived phase oracle that flags exactly the marked (prime) computational
basis states. One Grover iteration (near-optimal for M=6 marked items out of
N=16: optimal count = round(pi/(4*asin(sqrt(M/N))) - 0.5) = 1) is applied on
the ideal AerSimulator; the circuit is expected to amplify the prime states
so that a measurement sample lands on a prime with much higher-than-uniform
probability, and the highest-count outcomes should be exactly the prime set.
(Note: with exactly M=N/2 marked items -- e.g. 4 of 8 -- the mean amplitude
after the oracle is exactly zero and a single Grover diffusion step provably
does not amplify at all; N=16 with M=6 avoids that degenerate ratio.)

Honest reporting: ran_ok / verified_against_classical reflect whether this
substitute circuit actually ran and actually matched the classical primality
answer -- NOT whether problem #402's own (unavailable) sequence was verified.
"""

import sys

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_primes_below(n_bound: int) -> set:
    """Trial-division primality check, computed from first principles."""
    primes = set()
    for n in range(n_bound):
        if n < 2:
            continue
        is_prime = True
        for d in range(2, int(n ** 0.5) + 1):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.add(n)
    return primes


def build_oracle(qc: QuantumCircuit, marked: set, n_qubits: int) -> None:
    """Flip the phase of exactly the basis states in `marked` (0..2^n-1)."""
    for m in marked:
        bits = format(m, f"0{n_qubits}b")
        # Flip qubits where the target bit is 0, so the controlled-Z
        # condition (all-ones control) matches state `m`.
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(n_qubits - 1 - i)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(n_qubits - 1 - i)


def build_diffuser(qc: QuantumCircuit, n_qubits: int) -> None:
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


def main() -> bool:
    n_qubits = 4
    n_states = 2 ** n_qubits  # 16

    classical_answer = classical_primes_below(n_states)
    expected = {2, 3, 5, 7, 11, 13}
    if classical_answer != expected:
        print(f"FAIL: classical primality computation mismatch: "
              f"{classical_answer} != {expected}")
        return False

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    # One Grover iteration is optimal here: with M=4 marked out of N=8,
    # the optimal iteration count is floor(pi/4 * sqrt(N/M)) = floor(pi/4*sqrt(2)) = 1.
    build_oracle(qc, classical_answer, n_qubits)
    build_diffuser(qc, n_qubits)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    shots = 4096
    job = sim.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # counts keys are bitstrings c2 c1 c0 (qiskit little-endian bit order)
    def bitstring_to_int(bs: str) -> int:
        return int(bs, 2)

    outcome_counts = {}
    for bitstring, c in counts.items():
        val = bitstring_to_int(bitstring)
        outcome_counts[val] = outcome_counts.get(val, 0) + c

    marked_prob = sum(outcome_counts.get(v, 0) for v in classical_answer) / shots
    n_marked = len(classical_answer)
    # Top-N (N = number of marked items) most frequent outcomes should be
    # exactly the prime set.
    top_n = set(sorted(outcome_counts, key=lambda v: -outcome_counts[v])[:n_marked])

    print(f"Classical primes in [0,{n_states}): {sorted(classical_answer)}")
    print(f"Quantum outcome counts: { {k: outcome_counts.get(k, 0) for k in range(n_states)} }")
    print(f"Probability mass on marked (prime) states: {marked_prob:.3f}")
    print(f"Top-{n_marked} measured outcomes: {sorted(top_n)}")

    verified = (top_n == classical_answer) and (marked_prob > n_marked / n_states)

    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
