"""
Joshua Mayersky
BTO ferroelectric polarization simulation based on landau khalatnikov equations
Tae Kwon Song, "Landau-Khalatnikov Simulations for Ferroelectric Switching in Ferroelectric Random Access Memory Application", Journal of the Korean Physical Society, Vol. 46, No. 1, January 2005, pp. 59

To use: change fitting parameters epsilon, g_2, g_4, and g_6 as desired. Other parameters to play with are applied votlage, film thickness, and the time scales for pulses and the time step for calculating new values
"""

import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import cycle
import pyswarms as ps
from scipy import stats
import statistics as st


"""Plot Fonts and Text size"""
plt.rcParams['font.size'] = 12
plt.rcParams['font.serif'] = "Times New Roman"


"""Thickness of FE material"""
thickness_nm = 105.7 * math.pow(10, -9)
thickness_cm = thickness_nm / math.pow(10, -2)


"""Function that returns a list of values in a particular range"""
def create_range(start, end, step):
    new_list = []
    if start < end:
        while start <= (end - step):
            new_list.append(start)
            start += step
        return new_list
    else:
        while start >= (end - step):
            new_list.append(start)
            start += step
        return new_list


"""Function that is used to plot data and save the files to various folders"""
def plotting(y_axis, x_axis, y_label, x_label, title, scale, legend, yaxislim, ymax, ymin):
    print("Now Plotting...")
    x_color_plot = []
    y_color_plot = []
    color_legend = []
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    colors = ["blue", "green", "red", "black", "magenta", "orange", "cyan"]
    colorcycler = cycle(colors)

    fig, ax = plt.subplots()
    for j in range(0, len(x_axis)):
        ax.plot(x_axis[j], y_axis[j], label="{}".format(legend[j]))
    for k in range(0, len(x_color_plot)):
        ax.plot(x_color_plot[k], y_color_plot[k], linestyle=next(linecycler), color=next(colorcycler), linewidth=1, label="{}".format(color_legend[k]))

    ax.set_xlabel(x_label)
    ax.legend(loc='lower right')
    if yaxislim:
        axes = plt.gca()
        axes.set_ylim(ymin, ymax)
    if scale == "linear":
        ax.set_ylabel("{}".format(y_label))
    else:
        ax.set_ylabel(y_label)
        ax.set_yscale("log")

    fn = "{}".format(title)

    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    folder_name = plots_folder

    plt.savefig(folder_name + "/{}.png".format(fn), format="png")
    plt.close()


"""Landau-Devonshire function that returns the delta polarization for each time step"""
def l_d_function(e_field, epsilon, old_polarization, g_2, g_4, g_6):
    dp_dt = ((0.5 * epsilon * e_field) - (g_2 * old_polarization) - (
                g_4 * math.pow(old_polarization, 3)) - (g_6 * math.pow(old_polarization, 5)))
    return dp_dt


"""Runge-Kutta method for estimating the value of a function that changes with time and it's current value"""
def runge_kutta(epsilon, measured_voltage, measured_time, p_0, time_step, g_2, g_4, g_6):
    # print("Now running Runge-Kutta...")
    p = []
    v_g = []
    i = 0
    h = time_step

    for index, t in enumerate(measured_time):
        if index == 0:
            v_g.append(measured_voltage[0])
            p.append(p_0)
        else:
            """This is the core of the R-K method"""
            e_field_0 = measured_voltage[index - 1] / thickness_cm / math.pow(10, 3)
            e_field_1 = ((measured_voltage[index - 1] / thickness_cm / math.pow(10, 3)) + (measured_voltage[index] / thickness_cm / math.pow(10, 3))) / 2
            e_field_2 = measured_voltage[index] / thickness_cm / math.pow(10, 3)

            k_1 = l_d_function(e_field_0, epsilon, p[i - 1], g_2, g_4, g_6)
            k_2 = l_d_function(e_field_1, epsilon, p[i - 1] + (h * 0.5 * k_1), g_2, g_4, g_6)
            k_3 = l_d_function(e_field_1, epsilon, p[i - 1] + (h * 0.5 * k_2), g_2, g_4, g_6)
            k_4 = l_d_function(e_field_2, epsilon, p[i - 1] + (h * k_3), g_2, g_4, g_6)
            p_next = p[i - 1] + ((1 / 6) * h * (k_1 + (2 * k_2) + (2 * k_3) + k_4))
            v_g.append(e_field_2 * thickness_cm * math.pow(10, 3))  # gate voltage (V)
            p.append(p_next)

    return p, v_g


def runge_kutta_with_args(particle_array, measured_voltage, measured_time, measured_polarization, time_step):
    fitness_list = []
    for x in particle_array:
        polarization, vg_sweep_plot = runge_kutta(x[0], measured_voltage, measured_time, measured_polarization[0], time_step, x[1], x[2], x[3])
        mse = find_MSE(measured_polarization, polarization)
        fitness_list.append(mse)
    fitness_list = np.array(fitness_list)
    return fitness_list


def read_raw_measured():
    raw = pd.read_csv("./raw_measured/PCM fitting/0min/20V.csv", header=None, names=['v', 't', 'p'])
    polarization = raw.p
    voltage = raw.v
    time = raw.t
    return polarization, voltage, time


def find_MSE(actual, pred): # weighted via a cosine distribution with period based on the length of the applied voltage
    actual, pred = np.array(actual), np.array(pred)
    summed = 0
    #mse = np.square(np.subtract(actual, pred)).mean()
    root = (len(actual) / 4) / math.pi
    for index, (act_val, pred_val) in enumerate(zip(actual, pred)):
        weight = (math.cos(index/root) + 3) / 4
        #print(index, weight)
        sub = ((act_val - pred_val) ** 2) / weight
        summed += sub
    mse = summed / len(actual)
    #mse = np.square(np.subtract(actual, pred)).mean()
    return mse


def find_NMSE(actual, pred):    # normalized MSE, but it's weird
    actual, pred = np.array(actual), np.array(pred)
    nmse = (np.square(np.subtract(actual, pred)) / np.square(actual-actual.mean())).mean()
    return nmse


if __name__ == '__main__':
    """Time step for simulations (should match the keithley time step"""
    time_step = 5 * math.pow(10, -9)  # time step

    """Voltage definitions"""
    v_end = 20

    """"Run the PSO or only do manual fitting and plotting"""
    run_PSO = True
    #run_PSO = False

    """Fitting parameters"""
    # uC*kV = mJ
    ep_num = 8  #1
    g2_num = 1  #1
    g4_num = 1  #4
    g6_num = 1  #1

    ep_pow = 4  #5
    g2_pow = 5  #6
    g4_pow = 3  #3
    g6_pow = 0  #0
    epsilon = ep_num * math.pow(10, ep_pow)
    g2 = g2_num * math.pow(10, g2_pow)  # Jm/C^2 - what it's supposed to be, it's actually in mJ*cm / uC^2
    g4 = g4_num * math.pow(10, g4_pow)  # Jm^5/C^4 - what it's supposed to be, it's actually in mJ*cm^5 / uC^4
    g6 = g6_num * math.pow(10, g6_pow)  # Jm^9/C^6 - what it's supposed to be, it's actually in mJ*cm^9 / uC^6
    #epsilon = 103706.55950438
    #g2 = 2936641.85135359
    #g4 = 8672.14477492694
    #g6 = 37.9593037423401

    ep_min = 1 * math.pow(10, 3)
    g2_min = 1 * math.pow(10, 4)  # Jm/C^2 - what it's supposed to be, it's actually in mJ*cm / uC^2
    g4_min = 1 * math.pow(10, 2)  # Jm^5/C^4 - what it's supposed to be, it's actually in mJ*cm^5 / uC^4
    g6_min = 1 * math.pow(10, -1)  # Jm^9/C^6 - what it's supposed to be, it's actually in mJ*cm^9 / uC^6

    ep_max = 1 * math.pow(10, 5)
    g2_max = 1 * math.pow(10, 6)  # Jm/C^2 - what it's supposed to be, it's actually in mJ*cm / uC^2
    g4_max = 1 * math.pow(10, 4)  # Jm^5/C^4 - what it's supposed to be, it's actually in mJ*cm^5 / uC^4
    g6_max = 1 * math.pow(10, 2)  # Jm^9/C^6 - what it's supposed to be, it's actually in mJ*cm^9 / uC^6

    legend_polarization_meta_plot = []
    polarization_meta_plot = []
    vg_sweep_meta_plot = []

    """Read in the raw data for the PV sweep"""
    measured_polarization, measured_voltage, measured_time = read_raw_measured()
    polarization_meta_plot.append(measured_polarization)
    vg_sweep_meta_plot.append(measured_voltage)
    legend_polarization_meta_plot.append("Measured")

    #print(len(measured_voltage), len(measured_polarization), len(measured_time))

    Vmax = max(measured_voltage)
    Vmin = min(measured_voltage)

    """Look at the manually tuned version of the parameters"""
    polarization_man, vg_sweep_plot_man = runge_kutta(epsilon, measured_voltage, measured_time, measured_polarization[0], time_step, g2, g4, g6)

    polarization_meta_plot.append(polarization_man)
    vg_sweep_meta_plot.append(vg_sweep_plot_man)
    #legend_polarization_meta_plot.append("Simulated-Manual")
    legend_polarization_meta_plot.append("Simulated")

    mse = find_MSE(measured_polarization, polarization_man)
    print("Final cost:", mse)

    if run_PSO == True:
        """PSO portion of the code"""
        options = {'c1': 2, 'c2': 2, 'w': 0.7}
        min_bound = np.array([ep_min, g2_min, g4_min, g6_min])
        max_bound = np.array([ep_max, g2_max, g4_max, g6_max])
        bounds = (min_bound, max_bound)
        kwargs = {"measured_voltage": measured_voltage, 'measured_time': measured_time, "measured_polarization": measured_polarization, "time_step": time_step}
        optimizer = ps.single.GlobalBestPSO(n_particles=1000, dimensions=4, options=options, bounds=bounds)
        cost, pos = optimizer.optimize(runge_kutta_with_args, 30, **kwargs)

        """Re-run the RK method with the optimized parameters for plotting"""
        polarization, vg_sweep_plot = runge_kutta(pos[0], measured_voltage, measured_time, measured_polarization[0], time_step, pos[1], pos[2], pos[3])

        polarization_meta_plot.append(polarization)
        vg_sweep_meta_plot.append(vg_sweep_plot)
        legend_polarization_meta_plot.append("Simulated-PSO")

        """Saving simulated data to a csv file"""
        df = pd.DataFrame({"VG": vg_sweep_plot, "Polarization": polarization})
        df.to_csv("P-V ep_{} g2_{} g4_{} g6_{} Vprog_{}V.csv".format(pos[0], pos[1], pos[2], pos[3], v_end), index=False)

        """Saving the simulation values and MSE and R2 values"""
        slope, intercept, r, p, se = stats.linregress(measured_polarization, polarization)
        r2 = r ** 2
        mse = find_MSE(measured_polarization, polarization)
        nmse = find_NMSE(measured_polarization, polarization)
        df_2 = pd.DataFrame([{"MSE": mse, "NMSE": nmse, "R2": r2, "Epsilon": pos[0], "g2": pos[1], "g4": pos[2], "g6": pos[3]}])
        df_2.to_csv("{}V_PSO_fitting_parameters.csv".format(v_end), index=False)

        """Plot all three versions together"""
        plotting(polarization_meta_plot, vg_sweep_meta_plot, "Polarization ($μC/cm^2$)", "Voltage ($V$)",
                 "Polarization Hysteresis Loops {}V".format(v_end), "linear",
                 legend_polarization_meta_plot, True, 35, -35)
    else:
        """Plot all both manual fitting and measured data together"""
        plotting(polarization_meta_plot, vg_sweep_meta_plot, "Polarization ($μC/cm^2$)", "Voltage ($V$)",
                 "Polarization Hysteresis Loops {}V".format(v_end), "linear",
                 legend_polarization_meta_plot, True, 35, -35)

    """Saving simulated data to a csv file"""
    #df = pd.DataFrame({"VG": vg_sweep_plot_man, "Polarization": polarization_man})
    #df.to_csv("P-V ep_{} g2_{} g4_{} g6_{} Vprog_{}V.csv".format(epsilon, g2, g4, g6, v_end), index=False)

    """Saving the simulation values and MSE and R2 values"""
    #slope, intercept, r, p, se = stats.linregress(measured_polarization, polarization_man)
    #r2 = r ** 2
    #mse = find_MSE(measured_polarization, polarization_man)
    #nmse = find_NMSE(measured_polarization, polarization_man)
    #df_2 = pd.DataFrame(
    #    [{"MSE": mse, "NMSE": nmse, "R2": r2, "Epsilon": epsilon, "g2": g2, "g4": g4, "g6": g6}])
    #df_2.to_csv("{}V_PSO_fitting_parameters.csv".format(v_end), index=False)


