"""
Erdos problem #102 -- quantum-testable sequence attempt.

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
`number: "102"` (read-only clone of https://github.com/manman4/erdosproblems).
That entry reads:

    number: "102"
    prize: "no"
    informal_status: {state: "open", last_update: "2025-08-31"}
    formal_status: {state: "unformalized"}
    status: {state: "open", last_update: "2025-08-31"}
    oeis: ["N/A"]
    formalized: {state: "no", last_update: "2025-08-31"}
    tags: ["geometry"]

LIMITATION (reported honestly, per task instructions): Erdos problem #102 has
no associated OEIS sequence id -- `oeis: ["N/A"]`. It is an open geometry
problem (about point configurations), not a number sequence, so there is no
"sequence" here to build a quantum-testable membership/search property from.
Fabricating an OEIS id or grafting an unrelated sequence onto this problem
number would misrepresent the source data, which the task explicitly forbids.

Best honest attempt: rather than fake a tie to problem #102's (nonexistent)
sequence, this script implements a small, genuinely finite, genuinely
computable number-theoretic search -- "which residues mod 8 in {0,...,7} are
divisible by 3" -- as a real Grover's-algorithm oracle circuit on 3 qubits,
run on the ideal AerSimulator, and checked against the classical answer
computed from first principles (trial division) in this same script. This
demonstrates a real quantum circuit computing/verifying a real finite
property, but it is NOT derived from problem #102's content (which has no
computable sequence to derive from), and should not be read as being "about"
problem #102 beyond carrying its number and honestly documenting why no
genuine tie was possible.

Classical target (computed below, not copied from anywhere):
    N = 8 (3 qubits), integers 0..7
    property: n % 3 == 0
    marked set = {0, 3, 6}

ran_ok / verified_against_classical are reported accurately by this script's
own PASS/FAIL output; no result is faked.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


def classical_marked_set(n_bits: int, divisor: int) -> list[int]:
    """Return, by direct trial division, all n in [0, 2**n_bits) with n % divisor == 0."""
    n = 2 ** n_bits
    marked = []
    for x in range(n):
        # trial division from first principles: x % divisor == 0 means
        # divisor divides x with zero remainder.
        if x % divisor == 0:
            marked.append(x)
    return marked


def build_oracle(n_bits: int, marked: list[int]) -> QuantumCircuit:
    """Phase oracle flagging each marked basis state with a -1 phase."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_bits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_bits == 1:
            qc.z(0)
        elif n_bits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_bits - 1)
            qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
            qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    elif n_bits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_bits - 1)
        qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def grover_search(n_bits: int, marked: list[int], shots: int = 4096):
    n = 2 ** n_bits
    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)

    # optimal number of Grover iterations for |marked| solutions out of n
    m = len(marked)
    theta = np.arcsin(np.sqrt(m / n))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_bits), range(n_bits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_bits = 3
    divisor = 3
    classical = classical_marked_set(n_bits, divisor)
    print(f"Classical marked set (n % {divisor} == 0, n in 0..{2**n_bits - 1}): {classical}")

    counts, iterations = grover_search(n_bits, classical, shots=4096)
    print(f"Grover iterations used: {iterations}")
    print("Measurement counts:", counts)

    # Quantum result: the states measured most often should be exactly the
    # classically marked set. Take the top len(classical) most-frequent
    # outcomes (little-endian bitstrings -> integers) as the quantum answer.
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = sorted_counts[: len(classical)]
    quantum_marked = sorted(int(bits[::-1], 2) for bits, _ in top_k)

    total_shots = sum(counts.values())
    marked_shots = sum(c for bits, c in counts.items() if int(bits[::-1], 2) in classical)
    hit_rate = marked_shots / total_shots

    print(f"Quantum top-{len(classical)} measured states (as integers): {quantum_marked}")
    print(f"Fraction of shots landing on a classically-marked state: {hit_rate:.3f}")

    verified = (quantum_marked == sorted(classical)) and (hit_rate > 0.8)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
