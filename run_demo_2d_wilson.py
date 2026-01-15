from topo_dw import (
    mass_step_profile_y,
    mass_kink_profile_y,
    spectrum_2d_wilson,
    interface_modes_2d,
    spectral_flow_2d_wilson,
    plot_spectrum_png,
    plot_interface_modes_png,
    plot_spectral_flow_png,
)

def main():
    Ny = 60
    Nkx = 201
    r = 1.0

    m_top = -1.0
    m_bottom = 1.0

    # Step mass
    m_step = mass_step_profile_y(Ny, m_top, m_bottom)
    kx_step, E_step = spectrum_2d_wilson(m_step, Ny, Nkx=Nkx, r=r)
    ys_step, Es_step, Dens_step = interface_modes_2d(m_step, Ny, kx_target=0.0, n_modes=4, r=r)
    kx_flow_step, E_flow_step = spectral_flow_2d_wilson(m_step, Ny, Nkx=51, n_modes=4, r=r)

    print("Interface energies (step):", Es_step)

    plot_spectrum_png(kx_step, E_step, "wilson2d_step_spectrum.png",
                      title="2D Wilson–Dirac Spectrum (Step Mass)")
    plot_interface_modes_png(ys_step, Es_step, Dens_step,
                             "wilson2d_step_interface_modes.png",
                             title="Interface Modes (Step Mass)")
    plot_spectral_flow_png(kx_flow_step, E_flow_step,
                           "wilson2d_step_spectral_flow.png",
                           title="Spectral Flow (Step Mass)")

    # Kink mass
    m_kink = mass_kink_profile_y(Ny, m0=1.0, width=5.0)
    kx_kink, E_kink = spectrum_2d_wilson(m_kink, Ny, Nkx=Nkx, r=r)
    ys_kink, Es_kink, Dens_kink = interface_modes_2d(m_kink, Ny, kx_target=0.0, n_modes=4, r=r)
    kx_flow_kink, E_flow_kink = spectral_flow_2d_wilson(m_kink, Ny, Nkx=51, n_modes=4, r=r)

    print("Interface energies (kink):", Es_kink)

    plot_spectrum_png(kx_kink, E_kink, "wilson2d_kink_spectrum.png",
                      title="2D Wilson–Dirac Spectrum (Kink Mass)")
    plot_interface_modes_png(ys_kink, Es_kink, Dens_kink,
                             "wilson2d_kink_interface_modes.png",
                             title="Interface Modes (Kink Mass)")
    plot_spectral_flow_png(kx_flow_kink, E_flow_kink,
                           "wilson2d_kink_spectral_flow.png",
                           title="Spectral Flow (Kink Mass)")

if __name__ == "__main__":
    main()
