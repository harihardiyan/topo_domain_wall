from topo_dw import (
    mass_step_profile_y, 
    spectrum_2d_wilson, 
    interface_modes_2d, 
    spectral_flow_2d_wilson,
    plot_spectrum, 
    plot_interface_modes, 
    plot_spectral_flow
)

def main():
    Ny, Nkx, r = 60, 201, 1.0
    m_step = mass_step_profile_y(Ny, -1.0, 1.0)

    # 1. Spectrum
    kx_step, E_step = spectrum_2d_wilson(m_step, Ny, Nkx=Nkx, r=r)
    plot_spectrum(kx_step, E_step, "2D Wilson–Dirac Spectrum", "wilson2d_spectrum.png")

    # 2. Interface Modes
    ys, Es, Dens = interface_modes_2d(m_step, Ny, kx_target=0.0, n_modes=4, r=r)
    plot_interface_modes(ys, Dens, Es, "Interface Mode Profiles", "wilson2d_interface_modes.png")

    # 3. Spectral Flow
    kx_f, E_f = spectral_flow_2d_wilson(m_step, Ny, Nkx=51, n_modes=4, r=r)
    plot_spectral_flow(kx_f, E_f, "Spectral Flow", "wilson2d_spectral_flow.png")

    print("Success: Generated PNGs in the root folder.")

if __name__ == "__main__":
    main()
