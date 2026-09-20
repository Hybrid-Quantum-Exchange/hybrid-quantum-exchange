"""
Erdos problem #889 -- quantum-testable sequence lane.

Source data: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"889\"" (tags: ["number theory"], status: open, prize: "no").

LIMITATION (read before trusting the "verified against classical" claim):
Problem #889's YAML entry lists `oeis: ["possible"]`. That is a literal
placeholder string used elsewhere in this dataset to mean "an OEIS entry
may exist but has not been identified/linked yet" -- it is NOT an actual
OEIS sequence id (compare, e.g., a real entry such as "A048892" seen on
other problems in the same file). There is also no problem statement text
available in the cloned repo for #889 beyond the YAML metadata. Because of
that, there is no real OEIS sequence to derive a property from for this
problem, and fabricating one would violate the task's own instructions.

Per the task's fallback instruction ("write the script anyway with your
best honest attempt, note the limitation clearly ... report ran_ok /
verified_against_classical accurately rather than faking a pass"), this
script instead does the following honestly-labelled substitute:

  It builds a REAL, self-contained Grover-search circuit for a genuine,
  independently-checkable finite number-theory property in the same
  family as the problem's tag ("number theory"): quadratic residuosity
  mod a small prime. Concretely, for p = 7 and the domain x in {0,...,7}
  (3 qubits), it searches for x such that x is a nonzero quadratic
  residue mod p, i.e. x in QR(7) = {1, 2, 4} (since 1^2=1, 2^2=4, 3^2=2,
  4^2=2, 5^2=4, 6^2=1, all mod 7). This is computed from first principles
  classically in `classical_quadratic_residues` below -- no OEIS value is
  copied.

  This verifies that Grover's algorithm, run against a correctly built
  oracle for a genuine small arithmetic property, converges on the right
  answer set on the ideal AerSimulator. It does NOT verify anything about
  Erdos problem #889 itself, because #889 has no identifiable OEIS
  sequence in the source data to test against. That gap is intentional
  and reported truthfully via `verified_against_classical` /
  problem-level applicability rather than being hidden.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


def classical_quadratic_residues(p: int, domain_size: int):
    """Classically compute {x in [0, domain_size) : x is a nonzero QR mod p}."""
    residues = set()
    for r in range(1, p):
        residues.add((r * r) % p)
    return sorted(x for x in range(domain_size) if x in residues)


def build_oracle(marked_states, n_qubits):
    """Phase-flip oracle: applies -1 to each computational basis state in
    `marked_states` (each an int in [0, 2**n_qubits)), leaves all others
    unchanged. Built directly from the classical list, so it is exact."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")
        # Flip qubits that are 0 in this state so the multi-controlled-Z
        # fires exactly on |state>.
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
            qc.append(mcz, list(range(n_qubits)))
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(i)
    return qc


def build_diffuser(n_qubits):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
        qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_states, n_qubits, shots=4096):
    N = 2 ** n_qubits
    M = len(marked_states)
    if M == 0 or M == N:
        raise ValueError("Grover needs 0 < M < N marked states")

    # Optimal number of Grover iterations for amplitude amplification.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    oracle = build_oracle(marked_states, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    qc = qc.decompose().decompose().decompose()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    p = 7
    n_qubits = 3
    domain_size = 2 ** n_qubits  # 8

    marked_states = classical_quadratic_residues(p, domain_size)
    print(f"Classical nonzero quadratic residues mod {p} in [0, {domain_size}): "
          f"{marked_states}")

    counts, iterations = run_grover(marked_states, n_qubits)
    print(f"Grover iterations used: {iterations}")

    # Convert bitstring keys (qiskit orders classical register 'c2c1c0')
    # to integers.
    int_counts = Counter()
    for bitstring, c in counts.items():
        int_counts[int(bitstring, 2)] += c

    total_shots = sum(int_counts.values())
    marked_shots = sum(c for x, c in int_counts.items() if x in marked_states)
    marked_fraction = marked_shots / total_shots

    # Most-frequent measured outcome should be a genuine marked state.
    most_common_state, most_common_count = int_counts.most_common(1)[0]

    print(f"Fraction of shots landing on a quadratic residue: "
          f"{marked_fraction:.3f}")
    print(f"Most frequent measured state: {most_common_state} "
          f"(count {most_common_count}/{total_shots})")

    grover_success = (
        most_common_state in marked_states and marked_fraction > 0.8
    )

    if grover_success:
        print("PASS: Grover search converged on the classically-computed "
              "quadratic-residue set.")
    else:
        print("FAIL: Grover search did not converge on the classically-"
              "computed quadratic-residue set.")

    print()
    print("NOTE ON SCOPE: this PASS/FAIL is for the quadratic-residuosity "
          "Grover search instance only. It does NOT confirm anything about "
          "Erdos problem #889 itself -- that problem's source metadata has "
          "no real OEIS id (only the placeholder 'possible') and no cached "
          "statement text in this clone, so no property of its actual "
          "sequence could be identified or tested. See the module "
          "docstring for the full explanation.")

    return grover_success


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
