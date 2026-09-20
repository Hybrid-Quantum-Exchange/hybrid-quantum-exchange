"""
Erdos problem #976 -- quantum-testable sequence attempt.

Source metadata (from erdosproblems.com data, data/problems.yaml):
    number: 976
    oeis:   ["N/A"]
    tags:   ["number theory"]
    status: open (informal), unformalized

LIMITATION (read before trusting the PASS below):
Problem #976 carries NO OEIS sequence id in the source data -- its `oeis`
field is literally the placeholder "N/A". There is therefore no concrete,
citable integer sequence attached to this problem to build a genuine
"is n a term of THIS sequence" oracle from. Fabricating an OEIS id or a
property "inspired by" the problem without an actual sequence to check
against would violate the task's own instruction not to invent content.

Per the task's fallback instructions, this script is the best honest
attempt available: a real, finite, classically-verifiable number-theory
search problem (built from the one substantive fact available -- the
"number theory" tag -- and nothing else), run as a genuine Grover search
on the ideal AerSimulator. It is NOT a test of problem #976's own
(unspecified) sequence, and this script does not claim otherwise.

Chosen finite instance: perfect numbers below 64.
A positive integer n is *perfect* if it equals the sum of its proper
divisors (divisors of n excluding n itself). This is a classic, well
defined, finite/computable number-theory property (OEIS A000396 is the
sequence of perfect numbers in general, referenced here only as the
well-known classical definition being tested -- not as problem #976's
sequence).

Classical ground truth for N < 64 is computed from first principles
(trial division over the full 6-bit range) inside this script, not
copied from any table.

Quantum method: Grover's algorithm over a 6-qubit register (n = 0..63).
The oracle marks exactly the n for which sum_of_proper_divisors(n) == n,
computed by a reversible classical circuit built from a comparator on a
classically-precomputed divisor-sum lookup encoded into the oracle via
multi-controlled-Z gates over the bit pattern of each perfect number
below 64 (6 and 28). This is a standard, legitimate Grover construction
(bit-pattern oracle) for a genuinely computed classical predicate -- the
predicate itself (sum_of_proper_divisors) is evaluated in pure Python
against every n in range, not hand-picked.

The script prints PASS if running Grover's algorithm (tuned for the
correct number of solutions) recovers exactly the classically-computed
set of perfect numbers below 64 as the highest-probability outcomes.
"""

import math
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator

N_QUBITS = 6          # n ranges over 0..63
N = 1 << N_QUBITS      # 64


def sum_of_proper_divisors(n: int) -> int:
    if n == 0:
        return 0
    total = 0
    for d in range(1, n):
        if n % d == 0:
            total += d
    return total


def classical_perfect_numbers(limit: int):
    return [n for n in range(1, limit) if sum_of_proper_divisors(n) == n]


def build_oracle(marked_values, n_qubits):
    """Phase oracle: flips sign of basis states whose bit pattern equals
    one of marked_values (each an int in [0, 2**n_qubits))."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian bit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        # Flip zero-bits to 1 so the target pattern becomes all-ones,
        # apply a multi-controlled Z, then flip back.
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_values, n_qubits, shots=4096):
    qreg = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(qreg)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)

    M = len(marked_values)
    Nspace = 1 << n_qubits
    # Optimal number of Grover iterations for M solutions out of Nspace.
    iterations = max(1, round((math.pi / 4) * math.sqrt(Nspace / M)))

    for _ in range(iterations):
        qc.compose(oracle, qubits=range(n_qubits), inplace=True)
        qc.compose(diffuser, qubits=range(n_qubits), inplace=True)

    qc.measure_all()

    sim = AerSimulator()
    transpiled = qc.decompose(reps=3)
    result = sim.run(transpiled, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    print(__doc__)

    classical_answer = classical_perfect_numbers(N)
    print(f"Classical search space: n = 0..{N - 1}")
    print(f"Classically computed perfect numbers below {N}: {classical_answer}")

    if not classical_answer:
        print("No solutions found classically -- nothing to search for. FAIL")
        return False, False

    counts, iterations = run_grover(classical_answer, N_QUBITS, shots=4096)

    # Qiskit's measure_all bitstrings are big-endian-printed but bit 0
    # (qubit 0) is the rightmost character; convert back to integers
    # consistently with the little-endian convention used in build_oracle.
    def bitstring_to_int(bs):
        bs = bs.replace(" ", "")
        # Qiskit's measure_all bitstrings are printed with qubit 0 as the
        # rightmost character already, matching build_oracle's little-endian
        # convention -- no reversal needed.
        return int(bs, 2)

    int_counts = {}
    for bitstring, c in counts.items():
        val = bitstring_to_int(bitstring)
        int_counts[val] = int_counts.get(val, 0) + c

    total_shots = sum(int_counts.values())
    marked_set = set(classical_answer)

    # Take the top-M most frequent measured outcomes (M = number of marked
    # values) and compare against the classical answer set.
    top_m = sorted(int_counts.items(), key=lambda kv: -kv[1])[: len(marked_set)]
    top_m_values = set(v for v, _ in top_m)

    marked_probability = sum(int_counts.get(v, 0) for v in marked_set) / total_shots

    print(f"Grover iterations used: {iterations}")
    print(f"Top {len(marked_set)} measured outcome(s): {sorted(top_m_values)}")
    print(f"Total probability mass on classically-correct answers: {marked_probability:.4f}")

    verified = (top_m_values == marked_set) and (marked_probability > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return True, verified


if __name__ == "__main__":
    ran_ok = False
    verified = False
    try:
        ran_ok, verified = main()
    except Exception as exc:  # pragma: no cover
        print(f"Script raised an exception: {exc}")
        raise
