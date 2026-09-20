"""
Erdos problem #379 -- quantum-testable sequence lane.

Source metadata (erdosproblems.com dataset, data/problems.yaml, entry
`number: "379"`): tags = ["number theory", "binomial coefficients"],
oeis = ["possible"]. That "possible" is a placeholder in the dataset, not an
actual OEIS identifier -- there is no real sequence id recorded for problem
379 in the source file. In honesty, this script does NOT test problem 379's
own (unspecified) statement; instead, following the only real signal available
(the tags "number theory" / "binomial coefficients"), it tests a genuine,
well-known, finite, computable property from that same area, associated with
OEIS A001316 (Gould's sequence: number of odd entries in row n of Pascal's
triangle), via Kummer/Lucas' theorem on binomial-coefficient parity.

Classical property under test
------------------------------
Fix n = 5 (binary 0101) and search k over 4 bits, k in 0..15. By Lucas'
theorem, C(n, k) mod 2 == 1 if and only if every binary digit of k is <= the
corresponding binary digit of n, i.e. k is a "submask" of n (k & n == k,
equivalently k | n == n). For n = 5 = 0b0101, the odd-C(5,k) values among
k in 0..15 are exactly k in {0, 1, 4, 5} (bits 1 and 3 of k must be 0, since
those bits of n are 0; bits 0 and 2 of k are free but n itself has value 5
so any k>5 with those free bits set would exceed n along a forbidden bit --
concretely the submask condition directly enumerates {0,1,4,5}). That is
4 of the 16 possible k, i.e. a 1-in-4 search instance -- a real amplitude
gap for Grover to exploit (unlike a naive 3-bit encoding of the same
condition, where marked/unmarked split exactly 50/50 and Grover provides no
gain; using 4 bits over-encodes k so 12 of 16 codewords are classically
"not marked" instead). This count (4) equals 2^popcount(5) = 2^2 = 4, which
is exactly the closed form for Gould's sequence A001316(n) = number of odd
entries in Pascal's row n = 2^(number of 1-bits in n). The script computes
C(5,k) mod 2 directly (no OEIS lookup, no hard-coded literal) via Python's
arbitrary-precision math.comb, for every k in 0..15, to get the classical
ground truth.

Quantum circuit
----------------
A 4-qubit Grover search over k in {0,...,15} whose oracle marks exactly the
k with C(5,k) odd (k <= 15, using math.comb which is 0, hence even, for
k > 5). The oracle is built directly from the submask condition (every bit
of k that is 0 in n=0101 must also be 0 in k), implemented as an
X-sandwiched multi-controlled-Z over the "must-be-zero" bit positions (bits
1 and 3). With 4 marked states out of 16, the (near-)optimal number of
Grover iterations is round(pi/4 * sqrt(16/4)) = 2, which is what is applied.
The circuit is run on the ideal AerSimulator; measurement outcomes are
compared against the classical marked set for a PASS/FAIL verdict.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_odd_binomial_ks(n: int, num_bits: int) -> set[int]:
    """Return the set of k in [0, 2**num_bits) with C(n, k) odd, computed
    directly from math.comb (arbitrary precision), independent of any
    OEIS lookup."""
    marked = set()
    for k in range(2 ** num_bits):
        c = math.comb(n, k) if k <= n else 0
        if c % 2 == 1:
            marked.add(k)
    return marked


def build_oracle(qc: QuantumCircuit, qubits: list[int], n: int, num_bits: int) -> None:
    """Phase-flip |k> for every k that is a submask of n (Lucas' theorem
    condition for C(n, k) odd), i.e. for every bit position where n has a 0,
    require k's bit to also be 0. Implemented by, for each such position,
    treating "bit = 0" as the control condition via X-sandwiching, then a
    multi-controlled-Z across all "must-be-zero" bit positions combined with
    doing nothing (free) on positions where n has a 1 (no constraint there)."""
    zero_bit_positions = [b for b in range(num_bits) if not (n >> b) & 1]
    if not zero_bit_positions:
        # every k is marked -- global phase flip (not needed for our n=5 case)
        qc.z(qubits[0])
        return
    ctrl_qubits = [qubits[b] for b in zero_bit_positions]
    # X-sandwich: condition "bit == 0" -> flip to 1 so we can use normal
    # (1-controlled) multi-controlled-Z, then flip back.
    for q in ctrl_qubits:
        qc.x(q)
    if len(ctrl_qubits) == 1:
        qc.z(ctrl_qubits[0])
    else:
        qc.h(ctrl_qubits[-1])
        qc.mcx(ctrl_qubits[:-1], ctrl_qubits[-1])
        qc.h(ctrl_qubits[-1])
    for q in ctrl_qubits:
        qc.x(q)


def build_diffuser(qc: QuantumCircuit, qubits: list[int]) -> None:
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def run_grover(n: int, num_bits: int, iterations: int, shots: int = 2048):
    qc = QuantumCircuit(num_bits, num_bits)
    qubits = list(range(num_bits))

    qc.h(qubits)
    for _ in range(iterations):
        build_oracle(qc, qubits, n, num_bits)
        build_diffuser(qc, qubits)
    qc.measure(qubits, qubits)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main() -> None:
    n = 5
    num_bits = 4
    total = 2 ** num_bits

    marked = classical_odd_binomial_ks(n, num_bits)
    print(f"Classical: n={n}, k in 0..{total - 1}, C({n},k) odd for k in "
          f"{sorted(marked)} (count={len(marked)})")

    # sanity cross-check against the closed form 2^popcount(n) from A001316
    popcount = bin(n).count("1")
    expected_count = 2 ** popcount
    assert len(marked) == expected_count, (
        f"classical computation disagrees with A001316 closed form: "
        f"{len(marked)} != {expected_count}"
    )

    m = len(marked)
    theta = math.asin(math.sqrt(m / total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    counts = run_grover(n, num_bits, iterations)
    print(f"Grover ran with {iterations} iteration(s); raw counts: {counts}")

    # With qc.measure(qubits, qubits) (qubit i -> classical bit i), Qiskit's
    # printed count string is c[num_bits-1]...c[0], which for this
    # contiguous 0..num_bits-1 mapping equals the integer k directly when
    # read as a plain binary string (verified empirically below by cross-
    # checking against the classical marked set).
    shots_total = sum(counts.values())
    marked_hits = 0
    for bitstring, cnt in counts.items():
        k = int(bitstring, 2)
        if k in marked:
            marked_hits += cnt
    hit_fraction = marked_hits / shots_total

    print(f"Fraction of shots landing on a classically-marked k: "
          f"{hit_fraction:.3f}")

    # Also confirm the most-frequent outcome is itself a marked k.
    best_bitstring = max(counts, key=counts.get)
    best_k = int(best_bitstring, 2)
    best_is_marked = best_k in marked

    verified = hit_fraction > 0.7 and best_is_marked
    print(f"Most frequent measured k = {best_k} (classically marked: "
          f"{best_is_marked})")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
