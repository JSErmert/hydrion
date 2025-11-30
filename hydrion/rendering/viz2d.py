import matplotlib.pyplot as plt
import numpy as np

def plot_episode_timeseries(history):
    if len(history) == 0:
        print("Empty history.")
        return

    T = np.arange(len(history))

    def arr(key, default=0.0):
        return np.array([h.get(key, default) for h in history], dtype=float)

    flow      = arr("flow")
    press     = arr("pressure")
    clog      = arr("clog")
    E_norm    = arr("E_norm")
    turb      = arr("sensor_turbidity")
    scatter   = arr("sensor_scatter")
    C_out     = arr("C_out")
    eff       = arr("particle_capture_eff")

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(T, flow, label="Flow")
    axs[0].plot(T, press, label="Pressure")
    axs[0].set_title("Hydraulics")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(T, clog, label="Clog")
    axs[1].plot(T, E_norm, label="E_norm")
    axs[1].set_title("Clogging + Electrostatics")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(T, C_out, label="C_out")
    axs[2].plot(T, eff, label="Particle Capture Eff")
    axs[2].set_title("Particle System")
    axs[2].legend()
    axs[2].grid()

    axs[3].plot(T, turb, label="Turbidity")
    axs[3].plot(T, scatter, label="Scatter")
    axs[3].set_title("Optical Sensors")
    axs[3].legend()
    axs[3].grid()

    plt.tight_layout()
    plt.show()
