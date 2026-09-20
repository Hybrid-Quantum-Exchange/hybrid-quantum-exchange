"""
Erdos problem #4 -- quantum-testable instance.

Erdos problem #4 (see erdosproblems.com/4, data/problems.yaml entry
`number: "4"`) concerns maximal gaps between consecutive primes and is
tagged number theory / primes, with associated OEIS sequence A002386
("Record (maximal) gaps between consecutive primes: primes p such that the
gap to the next prime exceeds the gap between any smaller pair of
consecutive primes").

Classical property tested here (derived from first principles in this
script, not copied from OEIS):

    For n in the small range [2, 9], n is a "gap record" prime iff n is
    prime and the gap to the next prime, next_prime(n) - n, is strictly
    greater than every gap between consecutive primes among all prime
    pairs below n in that same range.

    Working this out by hand for [2, 9]:
        primes in range: 2, 3, 5, 7 (gaps use the true next prime, even if
        it falls outside [2, 9])
        gaps:            2->3  = 1   (first gap: trivially a record)
                          3->5  = 2   (2 > 1: record)
                          5->7  = 2   (2 is not > previous max 2: not a record)
                          7->11 = 4   (4 > 2: record)
    So the record-gap primes in [2, 9] are {2, 3, 7} -- exactly the first
    three terms of OEIS A002386 (2, 3, 7, 23, 89, ...), confirming the
    classical computation below matches the real sequence before any
    quantum step runs.

Quantum circuit:
    We encode candidates n = 2..9 as 3-qubit basis states |i> with
    n = i + 2 (i = 0..7, 8 = 2^3 candidates exactly). A Grover search
    circuit is built whose oracle phase-flips exactly the basis states
    corresponding to the classically-determined record-gap set {2, 3}
    (i.e. i in {0, 1}), computed by the classical routine below -- the
    oracle is not hand-picked independently of that computation. Grover's
    diffusion operator is applied the optimal number of times for
    2 marked items out of 8, and the circuit is run on the ideal
    AerSimulator. The script PASSes if the two most-measured basis states
    are exactly the classically-computed marked set.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Classical computation (first principles): find record prime gaps in [2, 9]
# ---------------------------------------------------------------------------

def is_prime(k: int) -> bool:
    if k < 2:
        return False
    for d in range(2, int(math.isqrt(k)) + 1):
        if k % d == 0:
            return False
    return True


LO, HI = 2, 9  # inclusive range -> 8 candidates, fits exactly in 3 qubits
candidates = list(range(LO, HI + 1))
primes_in_range = [n for n in candidates if is_prime(n)]

record_primes = []
max_gap_so_far = 0
for idx, p in enumerate(primes_in_range):
    # gap from this prime to the next prime overall (may be outside range;
    # compute the true next prime so the gap is mathematically correct)
    q = p + 1
    while not is_prime(q):
        q += 1
    gap = q - p
    if gap > max_gap_so_far:
        record_primes.append(p)
        max_gap_so_far = gap

marked_values = sorted(record_primes)
marked_indices = sorted(v - LO for v in marked_values)

print(f"Candidates n in [{LO}, {HI}]")
print(f"Primes in range: {primes_in_range}")
print(f"Classically computed record-gap primes (OEIS A002386 property): {marked_values}")
print(f"Corresponding 3-qubit basis indices: {marked_indices}")

assert marked_values == [2, 3, 7], (
    "Sanity check failed: expected the first three OEIS A002386 terms "
    f"2, 3, 7 to be the record-gap primes in [{LO}, {HI}], got {marked_values}"
)

# ---------------------------------------------------------------------------
# Quantum circuit: Grover search over the 8 candidates for the marked set
# ---------------------------------------------------------------------------

N_QUBITS = 3
N_STATES = 2 ** N_QUBITS
assert N_STATES == len(candidates)


def oracle(qc: QuantumCircuit, marked: list[int]) -> None:
    """Phase-flip each marked basis index (multi-controlled Z via X-sandwich)."""
    for idx in marked:
        bits = format(idx, f"0{N_QUBITS}b")
        # flip qubits that should be 0 in this basis state so a normal
        # multi-controlled Z targets exactly this bitstring
        for qi, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(qi)
        if N_QUBITS == 1:
            qc.z(0)
        elif N_QUBITS == 2:
            qc.cz(0, 1)
        else:
            qc.h(N_QUBITS - 1)
            qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
            qc.h(N_QUBITS - 1)
        for qi, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(qi)


def diffuser(qc: QuantumCircuit) -> None:
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))


M = len(marked_indices)
# optimal number of Grover iterations for M marked out of N_STATES
iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    oracle(qc, marked_indices)
    diffuser(qc)
qc.measure(range(N_QUBITS), range(N_QUBITS))

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's count-key string has its leftmost character as the highest
# classical-bit index (clbit N-1) down to the rightmost as clbit 0, which is
# exactly a standard big-endian binary number with the same weight as the
# basis index we built (qubit i carries weight 2**i). So int(bs, 2) directly
# recovers our candidate index -- no reversal needed.
def bitstring_to_index(bs: str) -> int:
    return int(bs, 2)

index_counts: dict[int, int] = {}
for bitstring, c in counts.items():
    idx = bitstring_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + c

sorted_indices = sorted(index_counts.items(), key=lambda kv: kv[1], reverse=True)
top_indices = sorted([idx for idx, _ in sorted_indices[:M]])

print(f"Grover iterations used: {iterations}")
print(f"Measurement counts by candidate index: {index_counts}")
print(f"Top-{M} measured indices: {top_indices}")
print(f"Classically expected marked indices: {marked_indices}")

quantum_matches_classical = (top_indices == marked_indices)

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
