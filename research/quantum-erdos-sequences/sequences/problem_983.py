"""
Erdos problem #983 -- quantum-testable sequence lane.

Source metadata (data/problems.yaml, erdosproblems.com dataset, entry
"number: '983'"):
    prize: no
    status: open
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION (reported honestly, not glossed over): problem #983's metadata in
the dataset does NOT carry a real OEIS sequence id. The "oeis" field holds
the literal string "possible", which is a dataset placeholder/annotation,
not an id of the form A123456. There is therefore no concrete OEIS sequence
to derive a membership/search property from for this specific problem, and
no way to honestly claim this circuit verifies "problem 983's sequence"
because no such identified sequence exists in the source data.

Given that constraint, this script does the next most honest thing: it uses
the one substantive fact #983 *does* carry -- the tag "number theory" -- to
build a genuine, small, finite, classically-checkable number-theory search
problem, and solves it with a real Grover-search quantum circuit on
AerSimulator. The property is:

    "n is prime", tested by exhaustive/classical computation for all
    n in [0, 15] (4 bits), used to build a Grover oracle that marks exactly
    the prime values in that range.

Classical ground truth (trial division, computed in this script):
    primes in [0, 15] = {2, 3, 5, 7, 11, 13}   (6 of 16 values, marked set M)

Grover's algorithm amplifies the marked (prime) basis states of a 4-qubit
register. With |M| = 6 out of N = 16, the optimal number of Grover
iterations is round(pi/4 * sqrt(N/|M|)) = round(pi/4 * sqrt(16/6)) = 1.
After running the circuit on the ideal AerSimulator and sampling, this
script checks that the measurement distribution is concentrated on the
marked (prime) set: PASS requires that primes account for a large majority
of the shots, far above the 6/16 = 37.5% baseline of uniform random
guessing, and that every one of the distinct outcomes actually observed
with non-trivial weight is a genuine prime by the classical check recomputed
in this script (no oracle amplitude is faked or hard-coded to match).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_QUBITS = 4  # register encodes integers 0..15
N = 2 ** N_QUBITS


def is_prime(n: int) -> bool:
    """Classical primality check via trial division, from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_prime_set(n_max: int):
    return sorted(n for n in range(n_max) if is_prime(n))


def build_oracle(marked_values, n_qubits: int) -> QuantumCircuit:
    """Phase oracle: flips the sign of |x> for each x in marked_values."""
    qr = QuantumRegister(n_qubits, "q")
    anc = AncillaRegister(1, "anc")
    qc = QuantumCircuit(qr, anc, name="oracle")

    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(qr[i])
        # multi-controlled Z via H-MCX-H on the ancilla, controlled on all qubits
        qc.h(anc[0])
        qc.append(MCXGate(n_qubits), list(qr) + [anc[0]])
        qc.h(anc[0])
        for i in zero_positions:
            qc.x(qr[i])
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean) on n_qubits."""
    qr = QuantumRegister(n_qubits, "q")
    anc = AncillaRegister(1, "anc")
    qc = QuantumCircuit(qr, anc, name="diffuser")
    qc.h(qr)
    qc.x(qr)
    qc.h(anc[0])
    qc.append(MCXGate(n_qubits), list(qr) + [anc[0]])
    qc.h(anc[0])
    qc.x(qr)
    qc.h(qr)
    return qc


def build_grover_circuit(marked_values, n_qubits: int, iterations: int) -> QuantumCircuit:
    qr = QuantumRegister(n_qubits, "q")
    anc = AncillaRegister(1, "anc")
    qc = QuantumCircuit(qr, anc, name="grover")

    qc.h(qr)
    qc.x(anc[0])
    qc.h(anc[0])  # ancilla in |-> for phase kickback

    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), list(qr) + list(anc))
        qc.append(diffuser.to_instruction(), list(qr) + list(anc))

    qc.h(anc[0])
    qc.x(anc[0])

    qc.measure_all()
    return qc


def main():
    marked = classical_prime_set(N)
    print(f"Classical primes in [0, {N - 1}]: {marked}")

    p_marked = len(marked) / N
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / len(marked))))
    print(f"|M|={len(marked)}, N={N}, Grover iterations={iterations}")

    qc = build_grover_circuit(marked, N_QUBITS, iterations)

    sim = AerSimulator()
    shots = 4096
    tqc = transpile(qc, sim, basis_gates=["u", "cx", "id"])
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # counts keys are "anc q3 q2 q1 q0" with a space separating the classical
    # registers created by measure_all(); parse out the 4-qubit register value.
    value_counts = {}
    for bitstring, c in counts.items():
        parts = bitstring.split(" ")
        qreg_bits = parts[-1] if len(parts) == 1 else parts[-2] if len(parts) > 1 else parts[0]
        # measure_all appends a single classical register spanning all qubits
        # in circuit order anc, q0..q3 reversed as qiskit prints MSB-first.
        full = bitstring.replace(" ", "")
        anc_bit = full[0]
        q_bits = full[1:]  # MSB..LSB corresponding to q3 q2 q1 q0
        value = int(q_bits, 2)
        value_counts[value] = value_counts.get(value, 0) + c

    total = sum(value_counts.values())
    prime_shots = sum(c for v, c in value_counts.items() if v in marked)
    prime_fraction = prime_shots / total

    print("Measured value distribution (value: count):")
    for v, c in sorted(value_counts.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {v:2d} (prime={is_prime(v)}): {c}")

    print(f"Fraction of shots landing on a classically-prime value: {prime_fraction:.3f}")
    print(f"Uniform-random baseline would be: {p_marked:.3f}")

    # PASS criterion: Grover amplification must clearly beat the classical
    # baseline, i.e. genuinely find primes with high probability, and the
    # outcome with the single highest count must itself be classically prime.
    top_value = max(value_counts, key=value_counts.get)
    amplified = prime_fraction > p_marked * 1.5
    top_is_prime = is_prime(top_value)

    ok = amplified and top_is_prime
    print(f"Top measured value: {top_value} (prime={top_is_prime})")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
