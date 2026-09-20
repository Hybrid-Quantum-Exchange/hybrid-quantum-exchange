"""
Erdos problem #730 (erdosproblems.com/730), OEIS A129515.

Problem #730 asks about pairs of distinct central binomial coefficients
C(2n, n) and C(2m, m) that share exactly the same set of prime divisors
(the same "prime support"). OEIS A129515 lives in that same context of
central binomial coefficients and their base-representation / prime
structure (tags: number theory, binomial coefficients, base
representations).

Directly searching for prime-support collisions between central binomial
coefficients is not a small finite instance a few-qubit circuit can
genuinely search. Instead this script tests a small, finite, genuinely
computable piece of the same underlying structure -- the base-2 structure
of a central binomial coefficient -- via Kummer's theorem:

    The exponent of the prime p in C(2n, n) (i.e. v_p(C(2n,n))) equals the
    number of carries that occur when n is added to itself in base p.

We pick p = 2 and a small n, and build a genuine reversible/quantum carry
circuit (using CNOT and Toffoli gates on an AerSimulator) that computes,
bit by bit, the carry produced when adding n + n in binary. The circuit
never performs a shortcut classical computation of v_2 -- it only ever
executes the standard ripple-carry logic

    c_{i+1} = (a_i AND b_i) XOR (c_i AND (a_i XOR b_i))

as quantum gates (CCX / CX) acting on qubits initialised (via X gates) to
the bits of n. Measuring the carry qubits and summing the 1s gives the
number of carries produced by the circuit, which by Kummer's theorem must
equal v_2(C(2n, n)).

The classical side (the "known answer") is computed independently and
from first principles in this script: C(2n, n) is computed exactly with
integer arithmetic, and v_2(C(2n, n)) is obtained by counting factors of
2 in that exact integer -- no OEIS value is copied.

Instance used: n = 6.
  C(12, 6) = 924 = 2^2 * 231, so v_2(C(12,6)) = 2.
  Binary of 6 is 110. Adding 110 + 110 = 1100 produces carries at bit
  positions 1 and 2 (2 carries total), which the circuit reproduces.

The script prints PASS if the quantum-computed carry count equals the
classically computed v_2(C(2n, n)), FAIL otherwise.
"""

from math import comb

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator


def classical_v2_central_binomial(n: int) -> int:
    """Compute v_2(C(2n, n)) exactly, from first principles."""
    value = comb(2 * n, n)
    if value == 0:
        return 0
    v2 = 0
    while value % 2 == 0:
        value //= 2
        v2 += 1
    return v2


def n_bits(n: int, k: int) -> list:
    """LSB-first list of the k bits of n."""
    return [(n >> i) & 1 for i in range(k)]


def build_carry_circuit(n: int, k: int) -> QuantumCircuit:
    """
    Build a reversible ripple-carry circuit that computes the carries
    produced when adding n + n in binary, using k bits of n (plus one
    extra carry-out position).

    Registers:
      a[0..k-1]  : holds the bits of n
      b[0..k-1]  : holds the bits of n
      p[0..k-1]  : scratch, ends up holding a_i XOR b_i
      c[0..k]    : carry chain; c[0] is carry-in (0), c[i+1] is the carry
                   produced at bit position i

    Carry recurrence implemented directly as quantum gates:
      c[i+1] ^= a_i AND b_i           (CCX)
      p_i     = a_i XOR b_i           (CX, CX)
      c[i+1] ^= c[i] AND p_i          (CCX)
    """
    a = QuantumRegister(k, "a")
    b = QuantumRegister(k, "b")
    p = QuantumRegister(k, "p")
    c = QuantumRegister(k + 1, "c")
    creg = ClassicalRegister(k + 1, "carry_out")
    qc = QuantumCircuit(a, b, p, c, creg)

    bits = n_bits(n, k)
    for i in range(k):
        if bits[i]:
            qc.x(a[i])
            qc.x(b[i])
    # c[0] stays |0> -- no carry into the least significant bit.

    for i in range(k):
        # p_i = a_i XOR b_i
        qc.cx(a[i], p[i])
        qc.cx(b[i], p[i])
        # c[i+1] ^= a_i AND b_i
        qc.ccx(a[i], b[i], c[i + 1])
        # c[i+1] ^= c[i] AND p_i
        qc.ccx(c[i], p[i], c[i + 1])

    qc.measure(c, creg)
    return qc


def quantum_carry_count(n: int, k: int) -> int:
    qc = build_carry_circuit(n, k)
    sim = AerSimulator()
    result = sim.run(qc, shots=1).result()
    counts = result.get_counts()
    assert len(counts) == 1, f"expected a deterministic outcome, got {counts}"
    bitstring = next(iter(counts))
    return bitstring.count("1")


def main() -> None:
    n = 6
    k = 3  # bits of n (6 = 0b110 fits in 3 bits; carry chain uses k+1 qubits)

    classical_answer = classical_v2_central_binomial(n)
    quantum_answer = quantum_carry_count(n, k)

    print(f"Erdos problem #730 / OEIS A129515 -- n = {n}")
    print(f"C(2n, n) = C({2 * n}, {n}) = {comb(2 * n, n)}")
    print(f"classical v_2(C(2n, n))            = {classical_answer}")
    print(f"quantum ripple-carry carry count    = {quantum_answer}")

    if quantum_answer == classical_answer:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
