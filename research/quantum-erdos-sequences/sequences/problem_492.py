"""
Erdos problem #492 -- quantum-testable companion script.

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
`number: "492"` (tags: ["number theory"], status: disproved 2025-08-31,
oeis: ["N/A"]).

LIMITATION (read before trusting the "verified" claim below): problem #492's
metadata record carries no OEIS sequence id ("N/A") and no numeric parameter
of its own that reduces to a small finite search/decision instance. There is
therefore no specific sequence belonging to problem 492 for a quantum circuit
to compute or verify. Rather than fabricate a value or silently borrow an
unrelated OEIS id and pretend it is "the" sequence for #492, this script
falls back to the one honest, well-defined, finite, computable property its
metadata *does* give: the problem's own `tags` field, `["number theory"]`.
The chosen instance below (primality of the integers 0..15) is a genuine,
independently-verifiable number-theory property with real mathematical
content -- it is NOT copied from any OEIS entry for problem 492, because none
exists. Treat this file as a "no genuine sequence-specific circuit possible"
case per the task's own fallback instructions: the circuit and the PASS/FAIL
check below are real and exercised on the ideal simulator, but they verify a
generic number-theory property (primality), not a property that is specific
to Erdos problem #492 itself.

Classical property tested
--------------------------
Grover's algorithm is used to search the 4-qubit computational basis
{0, 1, ..., 15} for the states |n> whose integer value n is PRIME. The
classical primality of every n in [0, 15] is computed from first principles
by this script (trial division), independently of any quantum step, and used
both to build the oracle and to check the quantum output.

Circuit
-------
- 4 qubits, computational basis states 0..15.
- Oracle: phase-flips exactly the basis states corresponding to primes in
  [0, 15], built as a sequence of multi-controlled-Z operations against the
  classically-computed prime list (not hard-coded from any external table).
- Diffuser: the standard Grover inversion-about-the-mean operator.
- Iterated ceil(pi/4 * sqrt(N/M)) times for N = 16 basis states and M =
  number of primes in range, then measured 4000 shots on AerSimulator.

PASS/FAIL
---------
PASS if, in simulation, the sampled outcomes are overwhelmingly concentrated
(>= 80% of shots) on the classically-correct set of primes in [0, 15]. (With
M=6 marked states out of N=16 and an integer iteration count, Grover's
theoretical maximum success probability here is ~0.84-0.85, not 1.0 -- the
80% bar reflects that ceiling rather than an arbitrarily loosened check.)
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_primes_upto_15():
    """Trial-division primality test for n in [0, 15], computed from scratch."""
    primes = []
    for n in range(16):
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


def build_oracle(n_qubits, marked_values):
    """Phase-oracle that flips the sign of each marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all qubits (controls = all but last, target = last)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits, marked_values):
    N = 2 ** n_qubits
    M = len(marked_values)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4000).result()
    counts = result.get_counts()
    return counts


def main():
    n_qubits = 4
    classical_primes = classical_primes_upto_15()
    print(f"Erdos problem #492: tags=['number theory'], oeis=['N/A'] (no sequence-specific OEIS id).")
    print(f"Fallback instance: primality search over n in [0, 15].")
    print(f"Classical primes in [0, 15] (trial division): {classical_primes}")

    counts = run_grover(n_qubits, classical_primes)

    total_shots = sum(counts.values())
    hits = 0
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        if n in classical_primes:
            hits += c
    hit_fraction = hits / total_shots

    print(f"Quantum measurement counts: {counts}")
    print(f"Fraction of shots landing on a classically-verified prime: {hit_fraction:.4f}")

    threshold = 0.80
    verified = hit_fraction >= threshold

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
