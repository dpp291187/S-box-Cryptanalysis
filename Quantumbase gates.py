import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer


def circuit_cnot():
    """
    CNOT circuit (2 qubits):
      - qubit0 = control
      - qubit1 = target
    Operation: target <- target ⊕ control.
    """
    qc = QuantumCircuit(2, name="CNOT")
    qc.cx(0, 1)
    return qc


def circuit_toffoli():
    """
    Toffoli circuit (3 qubits):
      - qubit0, qubit1 = controls
      - qubit2 = target
    Operation: target <- target ⊕ (control0 AND control1).
    """
    qc = QuantumCircuit(3, name="Toffoli")
    qc.ccx(0, 1, 2)
    return qc


def circuit_not():
    """
    NOT circuit (1 qubit):
      - qubit0
    Operation: qubit0 <- NOT(qubit0).
    """
    qc = QuantumCircuit(1, name="NOT")
    qc.x(0)
    return qc


def save_pdf(qc, filename, figsize=(3, 3), dpi=300):
    """
    Save the diagram of the quantum circuit 'qc' to a file named 'filename',
    with size 'figsize' and resolution 'dpi'.
    """
    style = {
        "figure.figsize": figsize,
        "font_size": 12,
        # Turn off LaTeX for simplicity; enable if needed:
        "text.usetex": False
    }
    fig = circuit_drawer(qc, output="mpl", style=style)
    plt.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def print_ops_info(qc, circuit_name):
    """
    Print resource information of the circuit:
      - Number of each gate type,
      - Total gate count,
      - Circuit depth.
    """
    ops = qc.count_ops()
    size = qc.size()
    depth = qc.depth()
    print(f"=== {circuit_name} ===")
    for gate, count in ops.items():
        print(f"  {gate}: {count}")
    print(f"sum: {size}")
    print(f"depth: {depth}\n")


def main():
    # Create three circuits
    cnot_qc = circuit_cnot()
    toff_qc = circuit_toffoli()
    not_qc = circuit_not()

    # Save each circuit to a file
    save_pdf(cnot_qc,  "cnot_circuit.png", figsize=(2, 2), dpi=300)
    save_pdf(toff_qc,  "toffoli_circuit.png", figsize=(3, 3), dpi=300)
    save_pdf(not_qc,   "not_circuit.png",    figsize=(1, 1), dpi=300)

    # Print resource information
    print_ops_info(cnot_qc,  "CNOT circuit")
    print_ops_info(toff_qc,  "Toffoli circuit")
    print_ops_info(not_qc,   "NOT circuit")

    # Print text diagrams for reference
    print("=== CNOT circuit (text) ===")
    print(cnot_qc.draw("text"))
    print("=== Toffoli circuit (text) ===")
    print(toff_qc.draw("text"))
    print("=== NOT circuit (text) ===")
    print(not_qc.draw("text"))


if __name__ == "__main__":
    main()
