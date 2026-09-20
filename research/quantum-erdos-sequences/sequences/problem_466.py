"""
Erdos problem #466 -- quantum-testable lane.

Source metadata (data/problems.yaml, entry `number: "466"`):
    prize: no
    informal_status: proved (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (reported honestly, per instructions): problem #466's YAML entry
carries no problem statement text and no OEIS sequence id (oeis == "N/A").
With no OEIS id and no problem text available in the read-only clone at
/home/user/manman4/erdosproblems, there is no specific sequence to build a
genuine, problem-466-derived quantum test around. Fabricating a fake tie to
problem 466 (e.g. inventing an OEIS id or a "definition" of the sequence)
would violate the "do not fabricate" instruction, so this script does not
attempt one.

Best-effort fallback actually executed below: since the only real content
available for #466 is its tag "number theory", this script builds a genuine,
non-fabricated, classically-verifiable number-theory search problem --
"find all primes in [0, 15]" -- and solves it with a real Grover search
circuit on qiskit_aer's ideal AerSimulator. Primality here is computed from
first principles by trial division in this script itself (no OEIS lookup,
no literal copied answer). The circuit and its correctness check are
genuine; the connection to Erdos problem #466 specifically is NOT genuine
and this docstring is not claiming one.

Search space: 4 qubits (N = 16), marked states = {2, 3, 5, 7, 11, 13}
(the primes in [0, 15], computed classically below).
Method: Grover search (diffusion + phase oracle marking the prime states),
with the optimal number of iterations for this marked-count/search-size
ratio, run on AerSimulator with statevector sampling.

Reported accurately: ran_ok reflects whether the script executed without
error; verified_against_classical reflects whether the Grover circuit's
highest-probability outcomes match the classically computed set of primes.
This is NOT a verification of any OEIS sequence tied to problem 466 (none
exists in source), only of the classical primality computation used as the
Grover oracle's target set.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


def classical_primes(n_max: int) -> list[int]:
    """Trial-division primality test, first principles, for 0..n_max inclusive."""
    primes = []
    for k in range(2, n_max + 1):
        is_prime = True
        for d in range(2, int(math.isqrt(k)) + 1):
            if k % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(k)
    return primes


def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase oracle flipping the sign of each marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
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


def run_grover(n_qubits: int, marked: list[int], shots: int = 4096):
    n = 2 ** n_qubits
    m = len(marked)
    # Optimal Grover iteration count for this N, M.
    theta = math.asin(math.sqrt(m / n))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim, basis_gates=["u", "cx", "ccx", "cz", "z", "x", "h"])
    result = sim.run(tqc, shots=shots, seed_simulator=1234).result()
    counts = result.get_counts()
    return counts, iterations


def main() -> bool:
    n_qubits = 4
    n_max = (2 ** n_qubits) - 1  # 15

    primes = classical_primes(n_max)
    print(f"Classical primes in [0, {n_max}] (trial division): {primes}")

    counts, iterations = run_grover(n_qubits, primes)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    # Rank measured bitstrings by frequency; top len(primes) should be the primes.
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = ranked[: len(primes)]
    measured_top_values = sorted(int(bits, 2) for bits, _ in top_k)

    prime_shot_fraction = sum(
        cnt for bits, cnt in counts.items() if int(bits, 2) in primes
    ) / total_shots

    print(f"Top-{len(primes)} measured values: {measured_top_values}")
    print(f"Fraction of shots landing on a prime state: {prime_shot_fraction:.4f}")

    verified = (
        measured_top_values == sorted(primes) and prime_shot_fraction > 0.8
    )

    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
