import numpy as np
import matplotlib.pyplot as plt
def Physics_driven_block(Vi, R_load, D1, D2, D_phi):
    # Define parameters
    Ls = 80e-6  # Transformer leakage inductor
    C2 = 100e-6  # DC Output Link Capacitance
    R_load = 15  # Load resistance
    R_T = 0.085  # AC Inductor Resistance
    Vi = 100  # DC Input Voltage
    n = 1  # Transformer Turns Ratio
    f = 20000  # Switching Frequency
    T = 1 / f  # Period
    T_hs = T / 2  # Half period

    D1 = 0.2212  # Duty cycle for primary side
    D2 = 0.2212  # Duty cycle for secondary side
    D_phi = 0.3957  # Phase shift between primary and secondary

    # Define time step and simulation time
    dt = 1e-7
    t_end = 0.5  # Simulation time
    t = np.arange(0, t_end, dt)  # Time vector

    # Initialize primary side bridge voltage Vab
    Vab = np.zeros_like(t)
    # Initialize secondary side bridge voltage Vcd
    Vcd = np.zeros_like(t)

    # Initialize v_C2 and i_L
    v_C2 = np.zeros_like(t)
    i_L = np.zeros_like(t)

    # Record primary side bridge voltage Vab waveform
    for i in range(len(t)):
        t_mod = t[i] % T  # Get the current time point within the period
        if D1 * T_hs <= t_mod < T_hs:
            Vab[i] = 1
        elif (1 + D1) * T_hs <= t_mod < T:
            Vab[i] = -1

        if (D1 + D_phi) * T_hs <= t_mod < (D1 + D_phi + 1 - D2) * T_hs:
            Vcd[i] = 1
        elif (t_mod < (D1 + D_phi - D2) * T_hs or ((1 + D1 + D_phi) * T_hs <= t_mod < T)):
            Vcd[i] = -1

        # 4th order Runge-Kutta method for v_C2 and i_L
        if i > 1:
            k1_vC2 = (n * Vcd[i - 1] * i_L[i - 1] - v_C2[i - 1] / R_load) / C2
            k1_iL = (Vab[i - 1] * Vi - n * Vcd[i - 1] * v_C2[i - 1] - R_T * i_L[i - 1]) / Ls

            k2_vC2 = (n * Vcd[i - 1] * (i_L[i - 1] + k1_iL * dt / 2) - (v_C2[i - 1] + k1_vC2 * dt / 2) / R_load) / C2
            k2_iL = (Vab[i - 1] * Vi - n * Vcd[i - 1] * (v_C2[i - 1] + k1_vC2 * dt / 2) - R_T * (
                        i_L[i - 1] + k1_iL * dt / 2)) / Ls

            k3_vC2 = (n * Vcd[i - 1] * (i_L[i - 1] + k2_iL * dt / 2) - (v_C2[i - 1] + k2_vC2 * dt / 2) / R_load) / C2
            k3_iL = (Vab[i - 1] * Vi - n * Vcd[i - 1] * (v_C2[i - 1] + k2_vC2 * dt / 2) - R_T * (
                        i_L[i - 1] + k2_iL * dt / 2)) / Ls

            k4_vC2 = (n * Vcd[i - 1] * (i_L[i - 1] + k3_iL * dt) - (v_C2[i - 1] + k3_vC2 * dt) / R_load) / C2
            k4_iL = (Vab[i - 1] * Vi - n * Vcd[i - 1] * (v_C2[i - 1] + k3_vC2 * dt) - R_T * (
                        i_L[i - 1] + k3_iL * dt)) / Ls

            v_C2[i] = v_C2[i - 1] + (k1_vC2 + 2 * k2_vC2 + 2 * k3_vC2 + k4_vC2) * dt / 6
            i_L[i] = i_L[i - 1] + (k1_iL + 2 * k2_iL + 2 * k3_iL + k4_iL) * dt / 6

    tp = np.arange(1, len(t) // 2 + 1)

    # Record only waveform change points
    record_t = []
    record_v_C2 = []
    record_i_L = []

    # Record initial values
    record_t.append(t[0])
    record_v_C2.append(v_C2[0])
    record_i_L.append(i_L[0])

    # Record the points where Vab and Vcd change
    for i in tp[1:]:
        if Vab[i] != Vab[i - 1] or Vcd[i] != Vcd[i - 1]:
            record_t.append(t[i])
            record_v_C2.append(v_C2[i])
            record_i_L.append(i_L[i])

    record_t = np.array(record_t)
    record_v_C2 = np.array(record_v_C2)
    record_i_L = np.array(record_i_L)

    # 绘制record_t, record_v_C2, record_i_L
    plt.figure()
    plt.plot(record_t, record_v_C2, 'b', label='v_C2')
    plt.plot(record_t, record_i_L, 'r', label='i_L')
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Recorded Waveform Data')
    plt.show()

    return record_t, record_v_C2, record_i_L
