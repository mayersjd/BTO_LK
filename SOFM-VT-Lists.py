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

"""Plot Fonts and Text size"""
plt.rcParams['font.size'] = 12
plt.rcParams['font.serif'] = "Times New Roman"


"""Global material constants/parameters"""
k = 1.38 * math.pow(10, -23)  # Boltzmann constant (J/K)
T = 300  # room temperature (K)
q = 1.602 * math.pow(10, -19)  # Charge of electron (C) or the electron volt conversion of J to eV (J)
na = 5.533 * math.pow(10, 19)  # p-doping concentration of the substrate, used in nmos (1/cm^3)
nd = 1 * math.pow(10, 15)  # n-doping concentration of the substrate, used in pmos (1/cm^3)
ni = 1 * math.pow(10, 10)  # intrinsic carrier concentration of silicon at 300K (1/cm^3)
chi_si = 4.05  # electron affinity of Si (eV)
eg_si = 1.12  # band-gap of Si at 300K (eV)
phi_m = 4.32  # metal work function of electrode (eV)
ep_naught = 8.854 * math.pow(10, -14)  # permittivity of free space (F/cm)
mu_n = 1400  # mobility of electrons in si at 300K (cm^2/V*s)
mu_p = 450  # mobility of holes in si at 300K (cm^2/V*s)
q_it = 0  # trapped charges at interface, ignoring for now
ep_si = 11.9  # dielectric constant of silicon
ep_ox = 2500  # dielectric constant of ferroelectric oxide
# area = math.pow(0.015, 2)  # area of devices in cm^2
l = 45 * math.pow(10, -6)  # length of the transistor (60um)
w = 67.5 * math.pow(10, -6)  # width of the transistor (90um)
dopant = 0  # variable for dopant concentration, either na or nd (1/cm^3)
mobility = 0  # variable for carrier mobility, either mu_n or mu_p (cm^2/V*s)

"""Thickness of FE material"""
thickness_nm = 100 * math.pow(10, -9)
thickness_cm = thickness_nm / math.pow(10, -2)

"""Calculated material parameters"""
phi_f = 0  # fermi level in si
phi_si = 0  # fermi work function of n-si
v_fb = (phi_m - phi_si)  # flatband voltage
phi_ms = v_fb  # difference in workfunctions of metal and silicon is the same as the flatband voltage
# c_ox = ep_naught * ep_ox * area / thickness_cm  # oxide capacitance (F)
c_ox_p = ep_naught * ep_ox / thickness_cm  # oxide capacitance (F/cm^2)

"""Fitting parameters for L-K equation"""
epsilon = 0
g2 = 0
g4 = 0
g6 = 0

"""Applied voltage"""
drain_bias = 1  # drain bias (V)
v_d = 0
e_prog = 0
e_read = 0

"""Other global variables for referencing"""
device_type = ""
idvg_meta_plot = []
vg_meta_plot = []
legend_idvg_plot = []
number_of_pulses = 0
number_of_periods = 0
last_pulse = ""
plot_epsilon = 0
plot_g2 = 0
plot_g4 = 0
plot_g6 = 0
vt_list = []
vt_neg_list = []
vt_pos_list = []
g6_list = []    #for plotting
g4_list = []    #for plotting
g2_list = []    #for plotting
ep_list = []    #for plotting
vprog_list = []
pprog_list = []
g6_values = []  #for looping

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
def plotting(y_axis, x_axis, y_label, x_label, title, scale, legend, with_legend, multi_plot, yaxislim, ymax, ymin,
             multi_axis, y_axis2, y_label2, thresholdVoltageShift):
    print("Now Plotting...")
    # Plotting all the data and saving them to individual files
    i = 1
    x_color_plot = []
    y_color_plot = []
    color_legend = []
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    colors = ["blue", "green", "red", "black", "magenta", "orange", "cyan"]
    colorcycler = cycle(colors)
    #plt.rcParams.update({'font.size': 12})
    #plt.rcParams.update({'axes.formatter.limits': [0, 0]})

    # Selections for showing 16 bit
    """
    g6_select_list = [0, 7, 11, 15, 19, 23, 27, 31]
    g6_select_list_for_vt = [0, 1, 14, 15, 22, 23, 30, 31, 38, 39, 46, 47, 54, 55, 62, 63]
    g6_select_list_for_vt_pos = [0, 14, 22, 30, 38, 46, 54, 62]
    """
    # Selections for showing 8bit
    g6_select_list = [1, 11, 19, 31]
    g6_select_list_for_vt = [2, 3, 22, 23, 38, 39, 62, 63]
    g6_select_list_for_vt_pos = [2, 22, 38, 62]

    fig, ax = plt.subplots()
    if multi_axis:
        ax2 = ax.twinx()

    if multi_plot:
        if multi_axis:
            if with_legend:
                ax.plot(x_axis, y_axis, linestyle=next(linecycler), color=next(colorcycler), label="{}".format(legend[0]))
                for j in range(0, len(y_axis2)):
                    ax2.plot(x_axis, y_axis2[j], linestyle=next(linecycler), color=next(colorcycler), label="{}".format(legend[j+1]))
            else:
                ax.plot(x_axis, y_axis)
                for j in range(0, len(y_axis2)):
                    ax2.plot(x_axis, y_axis2[j])
        else:
            if not thresholdVoltageShift:
                for j in range(0, len(x_axis)):
                    if with_legend:
                        ax.plot(x_axis[j], y_axis[j], label="{}".format(legend[j]))
                    else:
                        #if j % 4 != 0:    #100 for "analog" states
                        if j not in g6_select_list:
                            ax.plot(x_axis[j], y_axis[j], '0.7', linewidth=0.15)
                        else:
                            x_color_plot.append(x_axis[j])
                            y_color_plot.append(y_axis[j])
                            # color_legend.append(legend[j])
                            color_legend.append("{}".format(j+1))
                            i += 1
                            # ax.plot(x_axis[j], y_axis[j], linewidth=3)
                for k in range(0, len(x_color_plot)):
                    ax.plot(x_color_plot[k], y_color_plot[k], linestyle=next(linecycler), color=next(colorcycler), linewidth=1, label="{}".format(color_legend[k]))
            else:
                for j in range(0, len(x_axis)):
                    if with_legend:
                        ax.plot(x_axis[j], y_axis[j], label="{}".format(legend[j]))
                    else:
                        #if j % 8 > 1: #200 for "analog"
                        if j not in g6_select_list_for_vt:
                            ax.plot(x_axis[j], y_axis[j], '0.7', linewidth=0.15)
                        else:
                            #if j % 8 == 0: #200 for "analog"
                            if j in g6_select_list_for_vt_pos:
                                #color_legend.append("$g_6$-{}-Positive Pulse".format(i))
                                color_legend.append("{}, $V_T$, $P_{}$".format(int(j/2+1), "r+"))
                            else:
                                #color_legend.append("$g_6$-{}-Negative Pulse".format(i))
                                color_legend.append("{}, $V_T$, $P_{}$".format(int((j-1)/2+1), "r-"))
                                i += 1
                            x_color_plot.append(x_axis[j])
                            y_color_plot.append(y_axis[j])
                            # color_legend.append(legend[j])
                            # ax.plot(x_axis[j], y_axis[j], linewidth=3)
                for k in range(0, len(x_color_plot)):
                    ax.plot(x_color_plot[k], y_color_plot[k], linestyle=next(linecycler), color=next(colorcycler), linewidth=1, label="{}".format(color_legend[k]))

    else:
        if with_legend:
            ax.plot(x_axis, y_axis, label="{}".format(legend))
        else:
            ax.plot(x_axis, y_axis)

    ax.set_xlabel(x_label)
    if multi_axis:
        ax2.set_ylabel(y_label2)
        # ax2.legend(loc='lower left')  #for pmos IDVG plotting
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='lower left')
    else:
        ax.legend(loc='lower right')
        #ax.legend(loc='lower left')  #for pmos IDVG plotting
    if yaxislim:
        axes = plt.gca()
        axes.set_ylim(ymin, ymax)
    if scale == "linear":
        ax.set_ylabel("{}".format(y_label))
    else:
        # ax.set_ylabel("Log {}".format(y_label))
        ax.set_ylabel(y_label)
        ax.set_yscale("log")


    fn = "{}".format(title)

    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    plt.savefig(plots_folder + "/{}.png".format(fn), format="png", dpi=600)
    plt.close()


"""Function that returns the e-field at a given time value based on what type of pulse/signal is being sent"""
def e_of_t(amplitude, time, period):
    e_field = (2 * amplitude / math.pi) * np.arcsin(np.sin(2 * math.pi * time / period))
    return e_field


"""Landau-Devonshire function that returns the delta polarization for each time step"""
def l_d_function(e_field, epsilon, old_polarization, g_2, g_4, g_6):
    dp_dt = ((0.5 * epsilon * e_field) - (g_2 * old_polarization) - (
                g_4 * math.pow(old_polarization, 3)) - (g_6 * math.pow(old_polarization, 5)))
    return dp_dt


"""Runge-Kutta method for estimating the value of a function that changes with time and it's current value"""
def runge_kutta(epsilon, e_amp, time_range, time_period, time_step, g_2, g_4, g_6, length):
    # print("Now running Runge-Kutta...")
    p = []
    q_ox = []
    v_t = []
    v_g = []
    i_plot = []
    i = 0
    h = time_step
    delta_p = 0
    p_next = delta_p
    e_plot = []
    time_counter = 0

    for t in time_range:
        if time_counter >= length * 2:
            trad_Id_Vg_calc(v_t[-1])
            idvg, vg = trad_Id_Vg_calc(v_t[-1])
            idvg_meta_plot.append(idvg)
            vg_meta_plot.append(vg)
            legend_idvg_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(v_t[-1], 3)))

            idvg_meta_meta_plot.append(idvg)
            vg_meta_meta_plot.append(vg)
            legend_idvg_meta_meta_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(v_t[-1], 3)))

            vt_list.append(v_t[-1])
            if len(vt_list) % 2 == 0:
                vt_neg_list.append(v_t[-1])
                vprog_list.append(e_amp * thickness_cm * math.pow(10, 3) * -1)
            else:
                vt_pos_list.append(v_t[-1])
                vprog_list.append(e_amp * thickness_cm * math.pow(10, 3))
            g6_list.append(g_6)
            g4_list.append(g_4)
            g2_list.append(g_2)
            ep_list.append(epsilon)

            time_counter = 0
        if i != 0:
            """This is the core of the R-K method"""
            e_field_0 = e_of_t(e_amp, (t - h), time_period)
            e_field_1 = e_of_t(e_amp, (t - (h / 2)), time_period)
            e_field_2 = e_of_t(e_amp, t, time_period)

            k_1 = l_d_function(e_field_0, epsilon, p[i - 1], g_2, g_4, g_6)
            k_2 = l_d_function(e_field_1, epsilon, p[i - 1] + (h * 0.5 * k_1), g_2, g_4, g_6)
            k_3 = l_d_function(e_field_1, epsilon, p[i - 1] + (h * 0.5 * k_2), g_2, g_4, g_6)
            k_4 = l_d_function(e_field_2, epsilon, p[i - 1] + (h * k_3), g_2, g_4, g_6)
            p_next = p[i - 1] + ((1 / 6) * h * (k_1 + (2 * k_2) + (2 * k_3) + k_4))
            e_plot.append(e_field_2)
            v_g.append(e_field_2 * thickness_cm * math.pow(10, 3))  # gate voltage (V)
        else:
            e_plot.append(0)
            v_g.append(0)
        p.append(p_next)

        q_ox.append(p[-1] * math.pow(10, -6))  # polarization (charge) value in C/cm^2
        v_t.append(vt_naught - (q_ox[-1] / c_ox_p))  # threshold voltage change


        if device_type == "nmos":
            if v_g[-1] < v_t[-1]:
                i_d = 0
            elif v_g[-1] - v_t[-1] > v_d:  # linear, mobility controls nmos or pmos
                i_d = (w * c_ox_p * mobility / l) * (v_d * (v_g[-1] - v_t[-1]) - math.pow(v_d, 2) / 2)
            elif v_g[-1] - v_t[-1] <= v_d:  # saturation, mobility controls nmos or pmos
                i_d = (w * c_ox_p * mobility / l) * (math.pow(v_g[-1] - v_t[-1], 2) / 2)
        else:
            if v_g[-1] > v_t[-1]:
                i_d = 0
            elif abs(v_g[-1] - v_t[-1]) > abs(v_d):  # linear, mobility controls nmos or pmos
                i_d = -(w * c_ox_p * mobility / l) * (v_d * (v_g[-1] - v_t[-1]) - math.pow(v_d, 2) / 2)
            elif abs(v_g[-1] - v_t[-1]) <= abs(v_d):  # saturation, mobility controls nmos or pmos
                i_d = -(w * c_ox_p * mobility / l) * (math.pow(v_g[-1] - v_t[-1], 2) / 2)

        i_plot.append(i_d)
        i += 1
        time_counter += time_step
    return p, v_g


def trad_Id_Vg_calc(vt):
    # Section of loops and whatnot to look at traditional Id-Vg curves for nmos and pmos devices
    v_g_plot = []
    idvg_plot = []
    if device_type == "nmos":
        for v_g in create_range(0, 5, 0.1):
            if v_g < vt:
                id_vg = 0
            elif v_g - vt > v_d:  # linear, mobility controls nmos or pmos
                id_vg = (w * c_ox_p * mobility / l) * (v_d * (v_g - vt) - math.pow(v_d, 2) / 2)
            elif v_g - vt <= v_d:  # saturation, mobility controls nmos or pmos
                id_vg = (w * c_ox_p * mobility / l) * (math.pow(v_g - vt, 2) / 2)
            v_g_plot.append(v_g)
            idvg_plot.append(id_vg)
    else:
        for v_g in create_range(0, -5, -0.1):
            if v_g > vt:
                id_vg = 0
            elif abs(v_g - vt) > abs(v_d):  # linear, mobility controls nmos or pmos
                id_vg = abs(-(w * c_ox_p * mobility / l) * (v_d * (v_g - vt) - math.pow(v_d, 2) / 2))
            elif abs(v_g - vt) <= abs(v_d):  # saturation, mobility controls nmos or pmos
                id_vg = abs(-(w * c_ox_p * mobility / l) * (math.pow(v_g - vt, 2) / 2))
            v_g_plot.append(v_g)
            idvg_plot.append(id_vg)

    return idvg_plot, v_g_plot


if __name__ == '__main__':
    """Wave type"""
    wave_type = "triangle"

    #prog_type = "amplitude"
    prog_type = "pulse"

    """nmos or pmos"""
    device_type = "nmos"    # p-doped silicon, with a n-channel (electron is dominant carrier)
    #device_type = "pmos"  # n-doped silicon, with a p-channel (hole is dominant carrier)

    if device_type == "nmos":
        phi_f = (k * T / q) * math.log(na / ni)  # fermi level in p-si
        phi_si = chi_si + eg_si / 2 + phi_f  # fermi work function of p-si
        dopant = na
        mobility = mu_n
        v_d = 1 * drain_bias
        v_fb = (phi_m - phi_si)  # flatband voltage
        phi_ms = v_fb  # difference in workfunctions of metal and silicon is the same as the flatband voltage
        sqrt_input = 4 * ep_si * ep_naught * q * dopant * phi_f
        vt_naught = v_fb + (2 * phi_f) + (math.sqrt(sqrt_input) / c_ox_p)  # initial threshold voltage of the transistor
    else:
        phi_f = (k * T / q) * math.log(nd / ni)  # fermi level in n-si
        phi_si = chi_si + eg_si / 2 - phi_f  # fermi work function of n-si
        dopant = nd
        mobility = mu_p
        v_d = -1 * drain_bias
        v_fb = (phi_m - phi_si)  # flatband voltage
        phi_ms = v_fb  # difference in workfunctions of metal and silicon is the same as the flatband voltage
        sqrt_input = 4 * ep_si * ep_naught * q * dopant * abs(phi_f)
        vt_naught = v_fb - abs((2 * phi_f)) + (math.sqrt(sqrt_input) / c_ox_p)  # initial threshold voltage of the transistor

    print("Vt_naught:", vt_naught)

    if prog_type == "amplitude":    #amplitude programming
        # For measured data, time_step = 5 * math.pow(10, -9)
        time_step = 5 * math.pow(10, -9)  # time
    else:   # Pulse type programming
        time_step = 2.5 * math.pow(10, -12)  # time

    """Triangular wave definitions"""
    time_start = 0
    if prog_type == "amplitude":  # amplitude programming
        time_length = 1 * math.pow(10, -6)  # time pulse (amount of time for each segment of the triangular wave)
    else:  # Pulse type programming
        time_length = 5 * math.pow(10, -10)  # time

    time_period = 4 * time_length  # Total time period for full triangular pulse is 4*time_pulse

    """Pulse definitions"""
    if prog_type == "amplitude":    # amplitude programming
        time_pulse_start = 1 * math.pow(10, -6)  # pulse start time
        time_pulse_length = 1 * math.pow(10, -6)  # pulse length time
    else:   #pulse programming
        time_pulse_start = 5 * math.pow(10, -10)  # pulse start time
        time_pulse_length = 5 * math.pow(10, -10)  # pulse length time
    #time_pulse_period = 4 * time_pulse_length  # pulse period
    number_of_periods = 1
    #number_of_pulses = number_of_periods * 2
    #time_pulse_scale = 0.5

    """Voltage definitions"""
    v_start = 0
    if prog_type == "amplitude":
        v_end = 40  # 3 for pulses, 40 for amplitude
    else:
        v_end = 3  # 3 for pulses, 40 for amplitude
    v_end_max = 20
    v_start_kv = v_start / math.pow(10, 3)
    v_end_kv = v_end / math.pow(10, 3)
    """E-field variables, in units of kV/cm"""
    e_start = v_start_kv / thickness_cm
    e_end = v_end_kv / thickness_cm

    t_plot = []
    e_plot = []

    time_range = create_range(time_start, (time_period * number_of_periods) + time_length, time_step)

    for t in time_range:
        t_plot.append(t / math.pow(10, -9))

    """Fitting parameters"""
    number_of_states = 32   # 360 states for "analog"
    e_pend = []
    if prog_type == "amplitude":   #amplitude programming
        e_pend = create_range(e_start, e_end, (e_end / number_of_states))
    else:   # pulse programming
        for i in range(number_of_states):
            e_pend.append(e_end)

    legend_polarization_meta_plot = []
    polarization_meta_plot = []
    vg_sweep_meta_plot = []
    idvg_meta_meta_plot = []
    vg_meta_meta_plot = []
    legend_idvg_meta_meta_plot = []

    # Loop for generating the plots for the SOFM+FeFET paper, looks at a decay in fitting parameters to generate different PE curves
    state_iter = 1
    ep_values = []
    g2_values = []
    g4_values = []
    while state_iter <= number_of_states:
        """These equations were sort of arbitrarily determined/fitted to some measured data then modified to fit the VT scale"""
        if prog_type == "amplitude":  # amplitude programming
            ep_values.append((33248.6837825599 / state_iter) * math.log10(state_iter / 300) + (3.5 * 33248.6837825599))
            g2_values.append((1460107.51924403 / state_iter) * math.log10(state_iter / 10) + (2 * 1460107.51924403))
            g4_values.append(1 * 24766.2159790265 * math.exp(-2 * state_iter) + (2354.61569279783 * (number_of_states / state_iter)))
            g6_values.append(90 * 354325.344921231 * math.exp(-4.5 * state_iter) + (1.28651793543193 * (number_of_states / state_iter)))
        else:   # pulse programming
            ep_constant = 5 * math.pow(10, 8)
            g2_constant = 2 * math.pow(10, 9)
            g4_constant_1 = 1 * math.pow(10, 12)
            g4_constant_2 = 1 * math.pow(10, 7)
            g6_constant_1 = 5 * math.pow(10, 11)
            g6_constant_2 = 1 * math.pow(10, 4)
            ep_values.append((ep_constant / state_iter) * math.log10(state_iter / 300) + (3.5 * ep_constant))
            g2_values.append((g2_constant / state_iter) * math.log10(state_iter / 10) + (2 * g2_constant))
            g4_values.append(g4_constant_1 * math.exp(-2 * state_iter) + (g4_constant_2 * (number_of_states / state_iter)))
            g6_values.append(g6_constant_1 * math.exp(-4.5 * state_iter) + (g6_constant_2 * (number_of_states / state_iter)))

        state_iter += 1

    pulse_plot = []
    g6_plot = []
    g4_plot = []
    g2_plot = []
    ep_plot = []
    v_plot = []
    p_plot = []
    pulse_count = 1

    for index, (ep_iter, g2_iter, g4_iter, g6_iter, e_field_iter) in enumerate(zip(ep_values, g2_values, g4_values, g6_values, e_pend)):
        idvg_meta_plot = []
        vg_meta_plot = []
        legend_idvg_plot = []

        idvg, vg = trad_Id_Vg_calc(vt_naught)
        idvg_meta_plot.append(idvg)
        vg_meta_plot.append(vg)
        legend_idvg_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(vt_naught, 3)))

        polarization, vg_sweep_plot = runge_kutta(ep_iter, e_field_iter, time_range, time_period, time_step, g2_iter, g4_iter, g6_iter, time_pulse_length)

        polarization_meta_plot.append(polarization)
        vg_sweep_meta_plot.append(vg_sweep_plot)
        legend_polarization_meta_plot.append("g6_{}E15".format(plot_g6))

        # Lists for plotting
        pulse_plot.append(pulse_count)
        g6_plot.append(g6_iter * math.pow(10, 15))
        g4_plot.append(g4_iter * math.pow(10, 11))
        g2_plot.append(g2_iter * math.pow(10, 7))
        ep_plot.append(ep_iter)
        v_plot.append(e_field_iter * thickness_cm * math.pow(10, 3))
        p_plot.append(index)
        pprog_list.append(index)
        pprog_list.append(index)
        pulse_count += 1

    # Polarization Hysteresis loop plotting
    plotting(polarization_meta_plot, vg_sweep_meta_plot, "Polarization ($μC/cm^2$)", "Voltage ($V$)",
             "{} prog-Polarization Hysteresis Loops".format(prog_type), "linear", legend_polarization_meta_plot, False, True, True, 35, -35, False, [], "", False)

    # Stuff for SOFM+FeFET paper (Fitting parameter change with programming amplitude/pulses)
    if prog_type == "amplitude":    #amplitude programming
        plotting(g6_plot, v_plot, "$g_6$ Value ($Jm^9$/$C^6$)", "Programming Voltage", "{} prog-g6 Parameter Change".format(prog_type), "log",
                 ["$g_6$ Values", "$V_T$, $P_{}$".format("r+"), "$V_T$, $P_{}$".format("r-")],
                 True, True, False, 0, 0, True, [vt_pos_list, vt_neg_list], "Threshold Voltage ($V$)", False)

        plotting(g4_plot, v_plot, "$g_4$ Value ($Jm^5$/$C^4$)", "Programming Voltage", "{} prog-g4 Parameter Change".format(prog_type), "linear",
                 ["$g_4$ Values", "$V_T$, $P_{}$".format("r+"), "$V_T$, $P_{}$".format("r-")],
                 True, True, False, 0, 0, True, [vt_pos_list, vt_neg_list], "Threshold Voltage ($V$)", False)
        plotting(g2_plot, v_plot, "$g_2$ Value ($Jm$/$C^2$)", "Programming Voltage", "{} prog-g2 Parameter Change".format(prog_type), "linear",
                 ["$g_2$ Values", "$V_T$, $P_{}$".format("r+"), "$V_T$, $P_{}$".format("r-")],
                 True, True, False, 0, 0, True, [vt_pos_list, vt_neg_list], "Threshold Voltage ($V$)", False)
        plt.rcParams.update({'axes.formatter.limits': [0, 3]})
        plotting(ep_plot, v_plot, "$ep$ Value (Unitless)", "Programming Voltage", "{} prog-ep Parameter Change".format(prog_type), "linear",
                 ["$ep$ Values", "$V_T$, $P_{}$".format("r+"), "$V_T$, $P_{}$".format("r-")],
                 True, True, False, 0, 0, True, [vt_pos_list, vt_neg_list], "Threshold Voltage ($V$)", False)
    else:   # pulse programming
        plotting(g6_plot, p_plot, "$g_6$ Value ($Jm^9$/$C^6$)", "Programming Pulse", "{} prog-g6 Parameter Change".format(prog_type), "log",
                 ["$g_6$ Values", "$V_T$, $P_{}$".format("r+"), "$V_T$, $P_{}$".format("r-")],
                 True, True, False, 0, 0, True, [vt_pos_list, vt_neg_list], "Threshold Voltage ($V$)", False)

        plotting(g4_plot, p_plot, "$g_4$ Value ($Jm^5$/$C^4$)", "Programming Pulse", "{} prog-g4 Parameter Change".format(prog_type), "linear",
                 ["$g_4$ Values", "$V_T$, $P_{}$".format("r+"), "$V_T$, $P_{}$".format("r-")],
                 True, True, False, 0, 0, True, [vt_pos_list, vt_neg_list], "Threshold Voltage ($V$)", False)
        plotting(g2_plot, p_plot, "$g_2$ Value ($Jm$/$C^2$)", "Programming Pulse", "{} prog-g2 Parameter Change".format(prog_type), "linear",
                 ["$g_2$ Values", "$V_T$, $P_{}$".format("r+"), "$V_T$, $P_{}$".format("r-")],
                 True, True, False, 0, 0, True, [vt_pos_list, vt_neg_list], "Threshold Voltage ($V$)", False)
        plt.rcParams.update({'axes.formatter.limits': [0, 3]})
        plotting(ep_plot, p_plot, "$ep$ Value (Unitless)", "Programming Pulse", "{} prog-ep Parameter Change".format(prog_type), "linear",
                 ["$ep$ Values", "$V_T$, $P_{}$".format("r+"), "$V_T$, $P_{}$".format("r-")],
                 True, True, False, 0, 0, True, [vt_pos_list, vt_neg_list], "Threshold Voltage ($V$)", False)

    # Threshold Voltage Shift (ID-VG) plotting
    if device_type == "nmos":
        plotting(idvg_meta_meta_plot, vg_meta_meta_plot, "Current ($A$)", "Gate Voltage ($V$)",
                 "{} prog-{} Threshold Voltage Shift".format(prog_type, device_type), "log", legend_idvg_meta_meta_plot,
                 False, True, True, math.pow(10, 0), math.pow(10, -4), False, [], "", True)
    else:
        plotting(idvg_meta_meta_plot, vg_meta_meta_plot, "Abs Current (|$A$|)", "Gate Voltage ($V$)",
                 "{} prog-{} Threshold Voltage Shift".format(prog_type, device_type), "log", legend_idvg_meta_meta_plot,
                 False, True, True, math.pow(10, 0), math.pow(10, -4), False, [], "", True)

    # Formatting data for writing to CSV file
    vt_list_array = np.array(vt_list)
    g6_list_array = np.array(g6_list)
    g4_list_array = np.array(g4_list)
    g2_list_array = np.array(g2_list)
    ep_list_array = np.array(ep_list)
    vprog_list_array = np.array(vprog_list)
    pprog_list_array = np.array(pprog_list)

    # saving VT lists to csv files
    if prog_type == "amplitude":    #amplitude programming
        df = pd.DataFrame({"VT": vt_list_array, "g6 value": g6_list_array, "g4 value": g4_list_array, "g2 value": g2_list_array, "ep value": ep_list_array, "programming Voltage": vprog_list_array})
        df.to_csv("vt list {} Vprog_{}V-{}.csv".format(device_type, v_end, prog_type), index=False)
    else:   # pulse programming
        df = pd.DataFrame({"VT": vt_list_array, "g6 value": g6_list_array, "g4 value": g4_list_array, "g2 value": g2_list_array, "ep value": ep_list_array, "programming Voltage": vprog_list_array, "programming pulse": pprog_list_array})
        df.to_csv("vt list {} Vprog_{}V-{}.csv".format(device_type, v_end, prog_type), index=False)
