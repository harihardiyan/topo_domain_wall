import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_spectrum_png(kxs, E, filename, title):
    plt.figure(figsize=(8, 5))
    plt.plot(kxs, E, color="black", linewidth=0.5)
    plt.xlabel("kx")
    plt.ylabel("Energy")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_interface_modes_png(ys, Es, Dens, filename, title):
    plt.figure(figsize=(8, 5))
    for i in range(len(Es)):
        plt.plot(ys, Dens[i], label=f"Mode {i}, E={Es[i]:.3f}")
    plt.xlabel("y")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_spectral_flow_png(kxs, E_flow, filename, title):
    plt.figure(figsize=(8, 5))
    for i in range(E_flow.shape[1]):
        plt.plot(kxs, E_flow[:, i], label=f"Mode {i}")
    plt.xlabel("kx")
    plt.ylabel("Energy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
