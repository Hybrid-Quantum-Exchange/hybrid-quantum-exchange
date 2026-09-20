"""
Erdos problem #529 -- quantum-testable sequence lane.

Source of truth: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: '529'":

    number: "529"
    prize: "no"
    informal_status: {state: "open", last_update: "2025-08-31"}
    formal_status: {state: "unformalized"}
    status: {state: "open", last_update: "2025-08-31"}
    oeis: ["N/A"]
    formalized: {state: "no", last_update: "2025-08-31"}
    tags: ["geometry", "probability"]

LIMITATION (read before trusting the PASS below): problem #529 has NO OEIS
sequence associated with it ("oeis: [\"N/A\"]") and no formal statement is
present anywhere in the erdosproblems repository clone available here (only
the metadata block above -- the problem's English statement/page is not in
this checkout). Its tags ("geometry", "probability") indicate a continuous
geometric-probability question, not an integer sequence, so there is no
"small, finite, computable property of the sequence" to derive: there is no
sequence. Per the task instructions for this case, this script is my best
honest attempt rather than a faked pass against problem #529 itself.

What this script actually does: since no real property of problem #529 is
available, it runs a genuine, self-contained finite computable problem and
verifies it with a real quantum circuit (Grover search on AerSimulator),
exactly as the framework does for other lanes, so this file still exercises
a real Qiskit circuit rather than nothing. The chosen finite instance:

    Search space: integers n in [0, 15] (4 qubits).
    Property tested: n is a perfect square (n in {0,1,4,9}) AND n is even.
    That intersection, computed classically from first principles below,
    is {0, 4}.

This is NOT a property of Erdos problem #529 or of any OEIS sequence tied
to it -- there is none available -- and must not be read as verifying
anything about problem #529's actual (open) mathematical content. It exists
only to keep this lane runnable and honest about the gap, per instructions.

Fields to report: ran_ok reflects whether this script executes without
error; verified_against_classical reflects whether the Grover circuit's
most-likely measured outcomes match the classical set computed below --
NOT whether anything about problem #529 was verified, since problem #529
supplies no OEIS id and no derivable finite property.
"""

import sys
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_QUBITS = 4  # search space {0, ..., 15}


def classical_marked_set():
    """Compute, from first principles, all n in [0,15] that are BOTH a
    perfect square and even."""
    marked = []
    for n in range(2 ** N_QUBITS):
        is_square = any(k * k == n for k in range(2 ** N_QUBITS))
        is_even = (n % 2 == 0)
        if is_square and is_even:
            marked.append(n)
    return sorted(marked)


def build_oracle(marked_values, n_qubits):
    """Phase oracle flipping the sign of each marked computational basis
    state (given as little-endian integers)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
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
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked_values, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    classical = classical_marked_set()
    N = 2 ** N_QUBITS
    M = len(classical)
    assert classical == [0, 4], f"Unexpected classical marked set: {classical}"

    # Optimal number of Grover iterations for N states, M marked.
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))

    qc = build_grover_circuit(classical, N_QUBITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (qiskit: classical bit c[N-1]...c[0]) to little-endian ints.
    int_counts = {}
    for bitstring, c in counts.items():
        # Qiskit's bitstring is c[n-1]...c[0] (MSB first, matching our
        # little-endian qubit-index convention used in build_oracle), so
        # parsing it directly as a binary integer recovers the same value.
        value = int(bitstring, 2)
        int_counts[value] = int_counts.get(value, 0) + c

    ranked = sorted(int_counts.items(), key=lambda kv: -kv[1])
    top_values = {v for v, _ in ranked[:M]}
    top_prob = sum(c for v, c in int_counts.items() if v in classical) / shots

    print(f"Erdos problem #529: no OEIS id available (oeis: ['N/A']); "
          f"tags={{'geometry','probability'}} -> not a discrete sequence.")
    print(f"Fallback finite instance run instead (NOT problem #529 content): "
          f"n in [0,{N-1}], property = perfect-square AND even.")
    print(f"Classical marked set: {classical}")
    print(f"Grover iterations used: {iterations}")
    print(f"Top measured outcomes (by count): {ranked[:6]}")
    print(f"Probability mass on classically-marked states: {top_prob:.3f}")

    verified = (top_values == set(classical)) and (top_prob > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
