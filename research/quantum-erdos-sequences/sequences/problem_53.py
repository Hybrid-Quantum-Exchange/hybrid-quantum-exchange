"""
Erdos problem #53 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "53"`):
    prize: no
    informal_status: proved (2025-08-31), formal_status: Lean (2026-08-24)
    oeis: ["possible"]
    tags: ["number theory", "additive combinatorics"]

LIMITATION (read before trusting the "PASS" below as evidence about problem 53
itself): the data file does not give problem 53 a real OEIS sequence id --
the `oeis` field is the placeholder string "possible", not an A-number -- and
the read-only clone available in this environment does not carry the prose
statement of the problem, only this metadata row. Per the task instructions,
when no OEIS id is available the honest move is to still write a real quantum
script in the problem's thematic area and say plainly that it does not verify
problem 53's actual mathematical content, rather than fabricate an OEIS value
or invent a "sequence" that was never in the data.

So this script is a best-honest-effort fallback: it builds a genuine, finite,
classically-checkable number-theory search problem consistent with problem
53's tags ("number theory", "additive combinatorics") -- primality testing
over a small finite universe -- and solves it with a real Grover search
circuit on Qiskit's AerSimulator, then checks the quantum output against an
independently computed classical answer.

Classical property being tested
--------------------------------
Universe: integers 0..15, encoded as 4-qubit basis states |x3 x2 x1 x0>.
Property: x is prime (classical primality test, computed from first
principles by trial division in this script, no lookup table).

The primes in {0,...,7} are computed classically below. Grover's algorithm is
used to amplify exactly those basis states, using a phase oracle built from
the classically-computed marked set (the oracle is *derived from* the
classical computation, not an independent guess), and the circuit's most
frequent measurement outcomes are compared against that classical set.

Run: python3 problem_53.py
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    """Classical primality test by trial division, computed here from
    first principles (no external table)."""
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(universe_size: int) -> list:
    return [n for n in range(universe_size) if is_prime(n)]


def build_oracle(n_qubits: int, marked_values: list) -> QuantumCircuit:
    """Phase oracle that flips the sign of each basis state in
    `marked_values` (each an n_qubits-bit integer)."""
    oracle = QuantumCircuit(n_qubits, name="Oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # bit i -> qubit i
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            oracle.x(i)
        if n_qubits == 1:
            oracle.z(0)
        else:
            oracle.append(
                MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits))
            )
        for i in zero_positions:
            oracle.x(i)
    return oracle


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    diffuser = QuantumCircuit(n_qubits, name="Diffuser")
    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    if n_qubits == 1:
        diffuser.z(0)
    else:
        diffuser.append(
            MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits))
        )
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))
    return diffuser


def grover_search(n_qubits: int, marked_values: list, n_iterations: int, shots: int = 4096):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)

    for _ in range(n_iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    n_qubits = 4
    universe_size = 2 ** n_qubits  # 0..15

    primes = classical_primes(universe_size)
    print(f"Classical primes in 0..{universe_size - 1}: {primes}")

    n_marked = len(primes)
    n_total = universe_size
    # Optimal number of Grover iterations for M marked out of N states.
    theta = np.arcsin(np.sqrt(n_marked / n_total))
    n_iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations used: {n_iterations}")

    shots = 4096
    counts = grover_search(n_qubits, primes, n_iterations, shots=shots)

    # Sort outcomes by measured frequency, most frequent first.
    sorted_outcomes = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    print("Measurement counts (bitstring -> shots):", counts)

    # The top len(primes) most-frequent measured integers should be exactly
    # the classical prime set, since Grover amplifies marked states.
    top_values = set()
    for bitstring, _ in sorted_outcomes:
        value = int(bitstring, 2)
        top_values.add(value)
        if len(top_values) >= n_marked:
            break

    # Fraction of shots landing on a marked (prime) outcome -- should be
    # close to the Grover success probability, well above the 1/N baseline.
    marked_shots = sum(c for b, c in counts.items() if int(b, 2) in primes)
    success_rate = marked_shots / shots
    baseline_rate = n_marked / n_total

    print(f"Top {n_marked} measured values: {sorted(top_values)}")
    print(f"Measured success rate on marked states: {success_rate:.3f} "
          f"(uniform-random baseline would be {baseline_rate:.3f})")

    matches_classical_set = top_values == set(primes)
    beats_baseline = success_rate > baseline_rate * 1.5

    verified = matches_classical_set and beats_baseline

    print(f"Classical primes:            {sorted(primes)}")
    print(f"Quantum top-{n_marked} outcomes: {sorted(top_values)}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
