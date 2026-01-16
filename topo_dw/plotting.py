import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_spectrum(kx, E, title, filename):
    plt.figure(figsize=(6,4))
    for n in range(E.shape[1]):
        plt.plot(kx, E[:, n], 'k-', lw=0.4)
    plt.axhline(0, color='r', lw=0.8)
    plt.title(title)
    plt.xlabel("kx")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_interface_modes(ys, Dens, Es, title, filename):
    plt.figure(figsize=(6,4))
    for i in range(Dens.shape[0]):
        plt.plot(ys, Dens[i], label=f"mode {i}, E={Es[i]:.3f}")
    plt.title(title)
    plt.xlabel("y")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_spectral_flow(kx, E_flow, title, filename):
    plt.figure(figsize=(6,4))
    for i in range(E_flow.shape[1]):
        plt.plot(kx, E_flow[:, i], lw=1.2)
    plt.title(title)
    plt.xlabel("kx")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
