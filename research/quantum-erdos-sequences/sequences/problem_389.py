"""
Erdos problem #389 (Erdos-Straus divisibility problem), OEIS A375071.

A375071(n) = smallest positive integer k such that
    Product_{i=1..k} (n+i)   divides   Product_{i=1..k} (n+k+i)
(or 0 if no such k exists). Equivalently, writing the two products as
factorial ratios:
    L(n,k) = (n+k)! / n!            [= Product_{i=1..k} (n+i)]
    R(n,k) = (n+2k)! / (n+k)!       [= Product_{i=1..k} (n+k+i)]
a(n) is the smallest k >= 1 with L(n,k) | R(n,k).

This script fixes n = 1 and searches k over the small finite range
0..7 (3 qubits), which is exactly the search space a 3-qubit Grover
circuit can address. The classical property tested is:

    P(k) := ( (2k+1)! / (k+1)! )  is divisible by  (k+1)!         for k in 1..7

(k=0 is excluded as a trivial/undefined case and never marked).

We first compute P(k) classically for every k in 0..7 using exact
integer arithmetic (math.factorial), from first principles -- no OEIS
value is copied in. OEIS lists A375071(1) = 5; that is a claim we
verify here, not an input we assume.

We then build a genuine Grover search circuit over the 3-qubit space
{0,...,7} whose oracle marks exactly the k for which P(k) is True
(computed classically above, encoded as a multi-controlled-Z pattern
on the corresponding basis state(s)), run it on the ideal AerSimulator,
and check that the state(s) Grover amplifies match the classically
marked set -- in particular that the unique marked state is k=5,
confirming a(1)=5 as the smallest (and here, only) k in range solving
the divisibility condition.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_QUBITS = 3          # search space k = 0..7
N = 1                  # fixed Erdos-Straus parameter n


def classical_predicate(n: int, k: int) -> bool:
    """P(k): does (n+k)!/n! divide (n+2k)!/(n+k)! ?  (k=0 -> False, excluded)."""
    if k == 0:
        return False
    L = math.factorial(n + k) // math.factorial(n)
    R = math.factorial(n + 2 * k) // math.factorial(n + k)
    return R % L == 0


def compute_marked_states(n: int, n_qubits: int):
    marked = []
    for k in range(2 ** n_qubits):
        if classical_predicate(n, k):
            marked.append(k)
    return marked


def classical_answer(n: int, n_qubits: int) -> int:
    """The smallest k in [1, 2**n_qubits - 1] with P(k) True (0 if none)."""
    for k in range(1, 2 ** n_qubits):
        if classical_predicate(n, k):
            return k
    return 0


def oracle_for_marked(marked_states, n_qubits):
    """Phase-flip oracle marking each integer in marked_states (little-endian)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def diffuser(n_qubits):
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


def build_grover_circuit(marked_states, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = oracle_for_marked(marked_states, n_qubits)
    diff = diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diff.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    marked = compute_marked_states(N, N_QUBITS)
    answer = classical_answer(N, N_QUBITS)
    print(f"Classical scan over k=0..{2**N_QUBITS - 1} for n={N}:")
    for k in range(2 ** N_QUBITS):
        print(f"  k={k}: P(k)={classical_predicate(N, k)}")
    print(f"Marked states (P(k)=True): {marked}")
    print(f"Classical answer a({N}) restricted to this range = {answer}  (OEIS A375071({N}) = 5)")

    if not marked:
        print("FAIL: no marked states found classically; nothing to search for.")
        return False

    M = len(marked)
    Nspace = 2 ** N_QUBITS
    iterations = max(1, round((math.pi / 4) * math.sqrt(Nspace / M)))

    qc = build_grover_circuit(marked, N_QUBITS, iterations)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=2048).result()
    counts = result.get_counts()

    # Qiskit bit-string is big-endian over classical bits in register order;
    # our circuit measures qubit i into classical bit i, and Qiskit prints
    # classical bit (n-1) ... bit 0, so reverse to get little-endian int.
    def bits_to_int(bstr):
        return int(bstr[::-1], 2)

    dist = {}
    for bstr, cnt in counts.items():
        val = bits_to_int(bstr)
        dist[val] = dist.get(val, 0) + cnt

    most_likely = max(dist, key=dist.get)
    total_shots = sum(dist.values())
    marked_prob = sum(dist.get(k, 0) for k in marked) / total_shots

    print(f"Grover iterations used: {iterations}")
    print(f"Measurement distribution (value: count): {dict(sorted(dist.items()))}")
    print(f"Most likely measured value: {most_likely}")
    print(f"Total probability mass on marked states: {marked_prob:.3f}")

    quantum_found_marked_state = most_likely in marked
    quantum_matches_classical_answer = most_likely == answer

    ok = quantum_found_marked_state and quantum_matches_classical_answer and marked_prob > 0.8

    print()
    if ok:
        print(f"PASS: Grover search amplified k={most_likely}, matching the classical "
              f"answer a({N})={answer} for Erdos problem #389 / OEIS A375071.")
    else:
        print("FAIL: Grover result did not match the classical answer.")
    return ok


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
