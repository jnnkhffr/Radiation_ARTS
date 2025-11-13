"""Muss noch überarbeitet werden VMR passt noch nicht ganz so. """
import numpy as np
import matplotlib.pyplot as plt

def climate_column(Ts, Tcp, RH, N=100):
    """
    Idealized single-column atmosphere with a moist-adiabatic troposphere,
    a cold-point tropopause, and constant stratospheric temperature/humidity VMR.

    Inputs:
      Ts  : surface temperature [K]
      Tcp : cold-point tropopause temperature [K]
      RH  : tropospheric relative humidity (0-1)
      N   : number of pressure levels (default 100)

    Outputs:
      p : pressure profile [hPa] (surface to upper atmosphere)
      T : temperature profile [K]
      x : water-vapor volume mixing ratio profile [mol/mol] (dimensionless)
    """

    # Constants
    g = 9.80665                 # gravity [m/s^2]
    Rd = 287.05                 # gas constant for dry air [J/(kg·K)]
    Rv = 461.5                  # gas constant for water vapor [J/(kg·K)]
    cpd = 1004.0               # specific heat at constant pressure (dry air) [J/(kg·K)]
    epsilon = Rd / Rv           # ε = Rd/Rv
    e0 = 611.21                 # reference saturation vapor pressure [Pa]
    T0 = 273.15                 # reference temperature [K]
    Lv = 2.260e6               # latent heat of vaporization [J/kg] (2260 kJ/kg)

    # Pressure levels: logarithmically spaced from 1000 hPa to 1 hPa
    p = np.logspace(np.log10(1000e2), np.log10(1e2), N)  # from 1000 hPa to 1 hPa
    ln_p = np.log(p)

    # Allocate arrays
    T = np.full(N, np.nan)   # Temperature [K]
    x = np.full(N, np.nan)   # Water vapor volume mixing ratio, x = e/p [dimensionless]

    # Helper functions
    def esat(TK):
        # Clausius–Clapeyron saturation vapor pressure
        return e0 * np.exp(Lv / Rv * (1.0 / T0 - 1.0 / TK))

    def rsat(TK, pPa):
        # Saturation mass mixing ratio r_s = ε es / (p - es)
        es = esat(TK)
        return epsilon * es / (pPa - es)

    def specific_humidity_from_r(r):
        # q = r / (1 + r) Wandelt den Massenmischungsanteil r in die spezifische Feuchte q um.
        # q ist der Anteil der Wasserdampfmasse an der Gesamtmasse der feuchten Luft.
        return r / (1.0 + r)

    def moist_lapse_rate(TK, pPa):
        # Γ_m(T, p) [K/m], using saturated rs
        rs = rsat(TK, pPa)
        numerator = g * (1.0 + (Lv * rs) / (Rd * TK))
        denominator = cpd + (Lv**2 * rs * epsilon) / (Rv * TK**2)
        return numerator / denominator

    def virtual_temperature(TK, q):
        # Tv = T * [1 + (Rv/Rd - 1) q]
        return TK * (1.0 + (Rv / Rd - 1.0) * q)

    # surface temperature
    T[0] = Ts

    # Upward integration along moist adiabat until T reaches Tcp
    tropopause_index = None
    for i in range(N - 1):
        # Saturation quantities at (Ti, pi)
        es_i = esat(T[i])
        rs_i = epsilon * es_i / (p[i] - es_i)        # saturation mass mixing ratio
        qs_i = specific_humidity_from_r(rs_i)        # saturation specific humidity

        # Tropospheric vapor partial pressure and specific humidity from RH
        e_i = RH * es_i
        r_i = epsilon * e_i / (p[i] - e_i)
        q_i = specific_humidity_from_r(r_i)

        # Moist adiabatic lapse rate using saturated rs
        Gamma_m_i = moist_lapse_rate(T[i], p[i])     # [K/m]

        # Virtual temperature from actual q (enforced by RH)
        Tv_i = virtual_temperature(T[i], q_i)

        # Hydrostatic height increment and temperature update
        dlnp = ln_p[i + 1] - ln_p[i]                 # negative upward
        dz = -(Rd * Tv_i / g) * dlnp                 # [m], positive upward
        T[i + 1] = T[i] - Gamma_m_i * dz

        # Stop integration once T drops to Tcp; define tropopause
        if T[i + 1] <= Tcp and tropopause_index is None:
            tropopause_index = i + 1
            # Set remaining temperatures (stratosphere) to Tcp
            T[tropopause_index:] = Tcp
            break

    # If we never reached Tcp within N levels, set tropopause at top level
    if tropopause_index is None:
        tropopause_index = N - 1
        T[tropopause_index] = max(T[tropopause_index], Tcp)  # ensure not warmer than Tcp
        T[tropopause_index:] = Tcp

    # Build water-vapor VMR x = e/p
    # Troposphere: RH controls humidity; Stratosphere: fixed to tropopause value
    for i in range(N):
        if i <= tropopause_index:
            es_i = esat(T[i])
            e_i = RH * es_i
            x[i] = e_i / p[i]
        else:
            x[i] = x[tropopause_index]

    return p, T, x

def plot_profiles(p, T, x, Ts=None, Tcp=None, RH=None):
    """
    Plot temperature and water-vapor VMR profiles versus pressure.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    p_hPa = p * 1e-2

    # Temperature vs pressure
    axs[0].plot(T, p_hPa, color='tab:red', lw=2)
    #axs[0].invert_yaxis()
    axs[0].set_ylabel('Pressure [hPa]')
    axs[0].set_xlabel('Temperature [K]')
    axs[0].grid(True, ls='--', alpha=0.5)
    axs[0].set_ylim(1000, 1)

    # VMR vs pressure
    axs[1].plot(x, p_hPa, color='tab:blue', lw=2)
    #axs[1].invert_yaxis()
    axs[1].set_xlabel('Water vapor VMR x [mol/mol]')
    axs[1].grid(True, ls='--', alpha=0.5)
    axs[1].set_ylim(1000, 1)

    # Title
    if Ts is not None and Tcp is not None and RH is not None:
        fig.suptitle(f'Idealized single-column: Ts={Ts:.1f} K, Tcp={Tcp:.1f} K, RH={RH:.2f}', y=0.98)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Choose inputs
    Ts_example = 300.0   # K
    Tcp_example = 200.0  # K
    RH_example = 0.8     # dimensionless

    p, T, x = climate_column(Ts_example, Tcp_example, RH_example, N=100)
    plot_profiles(p, T, x, Ts=Ts_example, Tcp=Tcp_example, RH=RH_example)
