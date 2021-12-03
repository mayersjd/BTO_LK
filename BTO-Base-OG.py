"""
Joshua Mayersky
BTO ferroelectric polarization simulation based on landau khalatnikov equations
Tae Kwon Song, "Landau-Khalatnikov Simulations for Ferroelectric Switching in Ferroelectric Random Access Memory Application", Journal of the Korean Physical Society, Vol. 46, No. 1, January 2005, pp. 59

To use: change fitting parameters epsilon, g_2, g_4, and g_6 as desired. Other parameters to play with are applied votlage, film thickness, and the time scales for pulses and the time step for calculating new values
"""

import math
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from itertools import cycle

"""Plot Fonts and Text size"""
plt.rcParams['font.size'] = 12
plt.rcParams['font.serif'] = "Times New Roman"


"""Global material constants/parameters"""
# k = 8.62 * math.pow(10, -5)   # Boltzmann constant (eV/K)
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
gamma = 0  # variable used to control if the vt is increased or decreased based on the polarization. Positive polarization will decrease Vt in an nmos, but increase Vt in a pmos, adn vice versa
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
def plotting(y_axis, x_axis, y_label, x_label, title, scale, plot_g2, plot_g4, plot_g6, plot_epsilon, e_amp,
             multi_folder, legend, with_legend, multi_plot, yaxislim, ymax, ymin, multi_axis, y_axis2, y_label2, thresholdVoltageShift):
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
    #selections for showing 8bit
    g6_select_list = [1, 11, 19, 31]
    g6_select_list_for_vt = [2, 3, 22, 23, 38, 39, 62, 63]
    g6_select_list_for_vt_pos = [2, 22, 38, 62]

    #g6_select_list = [0, 8, 16, 24, 31]
    #g6_select_list_for_vt = [0, 1, 16, 17, 32, 33, 48, 49, 62, 63]
    #g6_select_list_for_vt_pos = [0, 16, 32, 48, 62]

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
        # ax2.legend(loc='lower left')
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='lower left')
    else:
        ax.legend(loc='lower right')
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

    # fig.tight_layout()

    # plt.title(fn)

    if multi_folder:
        """Folders for saving plots"""
        plots_folder = "plots"
        epsilon_folder = "ep_{}E13".format(plot_epsilon)
        g2_folder = "g2_{}E13".format(plot_g2)
        g4_folder = "g4_{}E11".format(plot_g4)
        g6_folder = "g6_{}E10".format(plot_g6)
        eamp_folder = "Eamp_{}kvcm-1".format(e_amp)

        eamp_comparison_folder = os.path.join(plots_folder, "Eamp comparisons", epsilon_folder, g2_folder, g4_folder,
                                              g6_folder)
        if not os.path.exists(eamp_comparison_folder):
            os.makedirs(eamp_comparison_folder)

        """These can be commented/uncommented to create folders specific to what parameter is being varied"""
        g6_comparison_folder = os.path.join(plots_folder, "g6 comparisons", epsilon_folder, g2_folder, g4_folder)
        # if not os.path.exists(g6_comparison_folder):
        # os.makedirs(g6_comparison_folder)

        g4_comparison_folder = os.path.join(plots_folder, "g4 comparisons", epsilon_folder, g2_folder, g6_folder)
        # if not os.path.exists(g4_comparison_folder):
        # os.makedirs(g4_comparison_folder)

        g2_comparison_folder = os.path.join(plots_folder, "g2 comparisons", epsilon_folder, g4_folder, g6_folder)
        # if not os.path.exists(g2_comparison_folder):
        # os.makedirs(g2_comparison_folder)

        ep_comparison_folder = os.path.join(plots_folder, "epsilon comparisons", g2_folder, g4_folder, g6_folder)
        # if not os.path.exists(ep_comparison_folder):
        # os.makedirs(ep_comparison_folder)

        plt.savefig(eamp_comparison_folder + "/{}.png".format(fn), format="png", dpi=600)
        # plt.savefig(g6_comparison_folder + "/{}.png".format(fn), format="png")
        # plt.savefig(g4_comparison_folder + "/{}.png".format(fn), format="png")
        # plt.savefig(g2_comparison_folder + "/{}.png".format(fn), format="png")
        # plt.savefig(ep_comparison_folder + "/{}.png".format(fn), format="png")
    else:
        plots_folder = "plots"
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
        folder_name = plots_folder

        plt.savefig(folder_name + "/{}.png".format(fn), format="png", dpi=600)
    plt.close()


"""Function that returns the e-field at a given time value based on what type of pulse/signal is being sent"""
def e_of_t(amplitude, time, period, wave, start, length, period_count, e_amp_step, pulse_counter, scale, pulse_high):
    if wave == "triangle":
        e_field = (2 * amplitude / math.pi) * np.arcsin(np.sin(2 * math.pi * time / period))
    elif wave == "triangle_pulse":
        if time < period:
            e_field = (2 * amplitude / math.pi) * np.arcsin(np.sin(2 * math.pi * time / period))
        elif (start + (period_count - 1) * period <= time) and (time < (start + (period_count - 1) * period + length)):
            if pulse_counter < 5:
                e_field = amplitude
            else:
                e_field = -amplitude
        elif ((start + (period_count - 1) * period + 2 * length) <= time) and (
                time < (start + (period_count - 1) * period + 3 * length)):
            if pulse_counter < 5:
                e_field = amplitude
            else:
                e_field = -amplitude
        else:
            e_field = 0
    elif wave == "pulse_only":
        if (start + period_count * period <= time) and (time < (start + period_count * period + length)):
            if period_count < (number_of_periods / 2):
                e_field = amplitude
                last_pulse = "pos"
            else:
                e_field = -amplitude
                last_pulse = "neg"
        elif ((start + period_count * period + 2 * length) <= time) and (
                time < (start + period_count * period + 3 * length)):
            if period_count < (number_of_periods / 2) - 1:
                e_field = amplitude
                last_pulse = "pos"
            else:
                e_field = -amplitude
                last_pulse = "neg"
        else:
            e_field = 0
    elif wave_type == "pos_pulse_only" or wave_type == "neg_pulse_only":
        if (start + period_count * period <= time) and (time < (start + period_count * period + length)):
            e_field = amplitude
        elif ((start + period_count * period + 2 * length) <= time) and (
                time < (start + period_count * period + 3 * length)):
            e_field = amplitude
        else:
            e_field = 0
    elif wave_type == "read_and_write":
        if (start + period_count * period <= time) and (time < (start + period_count * period + length)):
            if period_count % 2 == 0:
                e_field = amplitude
            else:
                e_field = -amplitude
        elif ((start + period_count * period + 2 * length) <= time) and (
                time < (start + period_count * period + 3 * length)):
            if period_count % 2 == 0:
                # e_field = (amplitude / 10) * (math.ceil(period_count/2) + 1)
                e_field = (amplitude / 10) * 1
                e_field = e_read
            else:
                # e_field = (amplitude / 10) * (math.ceil(period_count/2))
                e_field = e_read
            # if device_type == "pmos":
                # e_field *= -1
        else:
            e_field = 0
    elif wave == "increase_pos_pulse_amp_only":
        if (start + period_count * period <= time) and (time < (start + period_count * period + length)):
            e_field = amplitude
        elif ((start + period_count * period + 2 * length) <= time) and (
                time < (start + period_count * period + 3 * length)):
            e_field = amplitude + e_amp_step
        else:
            e_field = 0
    elif wave == "increase_pos_pulse_length_only":
        # if (start + (pulse_counter * (length * scale + length)) + (pulse_counter * length) <= time) and (time < (start + (pulse_counter * (length * scale + length)) + (pulse_counter * length) + length)):
        if pulse_high:
            e_field = amplitude
        else:
            e_field = 0
    return e_field


"""Landau-Devonshire function that returns the delta polarization for each time step"""
def l_d_function(e_field, epsilon, old_polarization, g_2, g_4, g_6):
    dp_dt = ((0.5 * epsilon * e_field) - (g_2 * old_polarization) - (
                g_4 * math.pow(old_polarization, 3)) - (g_6 * math.pow(old_polarization, 5)))
    return dp_dt


"""Runge-Kutta method for estimating the value of a function that changes with time and it's current value"""
def runge_kutta(epsilon, e_amp, e_plot, t_plot, time_range, time_period, time_step, g_2, g_4, g_6, multi_folder,
                wave_type, start, length, number_of_periods, scale):
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
    period_count = 0
    time_counter = 0
    e_amp_step = e_amp / number_of_periods
    e_amp_read = e_amp / 2
    number_of_pulses = number_of_periods * 2
    pulse_counter = 0
    total_pulsed_time = 0
    pulse_now = False  # I'm going to eventually use this signal for all the square wave pulses, as opposed to just the one at the end, that way I'mm not redundantly figuring out when a signal is supposed to be HIGH or LOW
    idvg_plotted = False
    last_pulse = ""
    g6_initial = g_6

    for t in time_range:
        """This is me experimenting with different wave types"""
        if wave_type == "triangle":
            if time_counter >= length * 2:
                trad_Id_Vg_calc(v_t[-1])
                idvg, vg, vt = trad_Id_Vg_calc(v_t[-1])
                idvg_meta_plot.append(idvg)
                vg_meta_plot.append(vg)
                legend_idvg_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(v_t[-1], 3)))

                idvg_meta_meta_plot.append(idvg)
                vg_meta_meta_plot.append(vg)
                legend_idvg_meta_meta_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(v_t[-1], 3)))

                vt_list.append(vt)
                if len(vt_list) % 2 == 0:
                    vt_neg_list.append(vt)
                    vprog_list.append(e_amp * thickness_cm * math.pow(10, 3) * -1)
                else:
                    vt_pos_list.append(vt)
                    vprog_list.append(e_amp * thickness_cm * math.pow(10, 3))
                g6_list.append(g_6)
                g4_list.append(g_4)
                g2_list.append(g_2)
                ep_list.append(epsilon)
                #pprog_list.append(pprog_list[-1] + 1)

                time_counter = 0
        elif wave_type == "triangle_pulse":
            if time_counter >= time_period:
                period_count += 1
                time_counter = 0
                idvg_plotted = False
            elif (time_counter >= length * 2) and (idvg_plotted != True):
                trad_Id_Vg_calc(v_t[-1])
                idvg, vg, vt = trad_Id_Vg_calc(v_t[-1])
                idvg_meta_plot.append(idvg)
                vg_meta_plot.append(vg)
                legend_idvg_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(v_t[-1], 3)))
                idvg_plotted = True
            if t >= start + total_pulsed_time + pulse_counter * time_pulse_length:
                pulse_counter += 1
                total_pulsed_time += ((pulse_counter - 1) * time_pulse_length * time_pulse_scale) + time_pulse_length
                print("t:", t, "pulse_counter:", pulse_counter)
                if (pulse_counter > 1) and (pulse_counter < 5):
                    g_6 /= 10  # Jm^9/C^6
                    last_pulse = "pos"
                else:
                    if last_pulse == "pos":
                        # g_6 = g6_initial
                        g_6 /= 10  # Jm^9/C^6
                    else:
                        g_6 /= 10  # Jm^9/C^6
                    last_pulse = "neg"
        elif wave_type == "pulse_only":
            if time_counter >= length * 4:
                period_count += 1
                time_counter = 0
                idvg_plotted = False
                trad_Id_Vg_calc(v_t[-1])
                idvg, vg, vt = trad_Id_Vg_calc(v_t[-1])
                idvg_meta_plot.append(idvg)
                vg_meta_plot.append(vg)
                legend_idvg_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(v_t[-1], 3)))
            elif (time_counter >= length * 2) and (idvg_plotted != True):
                trad_Id_Vg_calc(v_t[-1])
                idvg, vg, vt = trad_Id_Vg_calc(v_t[-1])
                idvg_meta_plot.append(idvg)
                vg_meta_plot.append(vg)
                legend_idvg_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(v_t[-1], 3)))
                idvg_plotted = True
            if t >= start + total_pulsed_time + pulse_counter * time_pulse_length:
                pulse_counter += 1
                total_pulsed_time += ((pulse_counter - 1) * time_pulse_length * time_pulse_scale) + time_pulse_length
                if pulse_counter > 1:
                    if pulse_counter < number_of_pulses / 2:
                        g_6 -= 1 * math.pow(10, 8) # Jm^9/C^6
                    else:
                        if last_pulse == "pos":
                            g_6 = g6_initial
                            g_6 -= 1 * math.pow(10, 8)  # Jm^9/C^6
                        else:
                            g_6 -= 1 * math.pow(10, 8)  # Jm^9/C^6
        elif wave_type == "pos_pulse_only" or wave_type == "neg_pulse_only":
            if time_counter >= length * 4:
                period_count += 1
                time_counter = 0
                idvg_plotted = False
                trad_Id_Vg_calc(v_t[-1])
                idvg, vg, vt = trad_Id_Vg_calc(v_t[-1])
                idvg_meta_plot.append(idvg)
                vg_meta_plot.append(vg)
                legend_idvg_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(v_t[-1], 3)))
            elif (time_counter >= length * 2) and (idvg_plotted != True):
                trad_Id_Vg_calc(v_t[-1])
                idvg, vg, vt = trad_Id_Vg_calc(v_t[-1])
                idvg_meta_plot.append(idvg)
                vg_meta_plot.append(vg)
                legend_idvg_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(v_t[-1], 3)))
                idvg_plotted = True
            if t >= start + total_pulsed_time + pulse_counter * time_pulse_length:
                pulse_counter += 1
                total_pulsed_time += ((pulse_counter - 1) * time_pulse_length * time_pulse_scale) + time_pulse_length
                if pulse_counter > 1:
                    if wave_type == "pos_pulse_only":
                        # g_6 /= 10  # Jm^9/C^6
                        g_6 -= 1 * math.pow(10, 14)
                    else:
                        # g_6 *= 10  # Jm^9/C^6
                        g_6 -= 1 * math.pow(10, 14)
                # print("t:", t, "epsilon:", epsilon)
        elif wave_type == "increase_pos_pulse_amp_only":
            if time_counter >= length * 4:
                period_count += 1
                time_counter = 0
                e_amp += 2 * e_amp_step
        elif wave_type == "read_and_write":
            if time_counter >= length * 4:
                period_count += 1
                time_counter = 0
                # if period_count % 2 == 0:
                # e_amp -= e_amp_read
                # else:
                # e_amp += e_amp_read
                idvg_plotted = False
            elif (time_counter >= length * 3.5) and (idvg_plotted != True):
                trad_Id_Vg_calc(v_t[-1])
                idvg, vg, vt = trad_Id_Vg_calc(v_t[-1])
                idvg_meta_plot.append(idvg)
                vg_meta_plot.append(vg)
                legend_idvg_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(v_t[-1], 3)))
                idvg_plotted = True
        elif wave_type == "increase_pos_pulse_length_only":
            if t >= start + total_pulsed_time + pulse_counter * time_pulse_length:
                period_count += 1
                pulse_counter += 1
                total_pulsed_time += ((pulse_counter - 1) * time_pulse_length * time_pulse_scale) + time_pulse_length
                pulse_now = True
                # print("t:", t, "number of pulses:", pulse_counter, "Pulse?", pulse_high)
            elif t > start + total_pulsed_time + pulse_counter * time_pulse_length - time_pulse_length:
                # print("t:", t, "number of pulses:", pulse_counter, "Pulse?", pulse_high)
                pulse_now = False
        if i != 0:
            """This is the core of the R-K method"""
            e_field_0 = e_of_t(e_amp, (t - h), time_period, wave_type, start, length, period_count, e_amp_step,
                               pulse_counter, scale, pulse_now)
            e_field_1 = e_of_t(e_amp, (t - (h / 2)), time_period, wave_type, start, length, period_count, e_amp_step,
                               pulse_counter, scale, pulse_now)
            e_field_2 = e_of_t(e_amp, t, time_period, wave_type, start, length, period_count, e_amp_step, pulse_counter,
                               scale, pulse_now)

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
        v_t.append(vt_naught - (gamma * q_ox[-1] / c_ox_p))  # threshold voltage change


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

    """Changing the scale of the fitting parameters to make the file names and plot titles a little easier to read"""
    """plot_epsilon = math.floor(epsilon / math.pow(10, 13))
    plot_g2 = math.floor(g_2 / math.pow(10, 13))
    plot_g4 = math.floor(g_4 / math.pow(10, 11))
    plot_g6 = math.floor(g_6 / math.pow(10, 10))"""

    if wave_type == "triangle":
        # plot the polarization vs. e-field
        """
        plotting(p, v_g, "Polarization (uC/cm^2)", "Voltage (V)",
                 "P-E Hysteresis ({}) g2_{}E7 g4_{}E9 g6_{}E16 ep_{}E21 Vprog_{}V wave_{}".format(device_type,
                                                                                                        plot_g2,
                                                                                                        plot_g4,
                                                                                                        plot_g6,
                                                                                                        plot_epsilon,
                                                                                                        (e_amp * math.pow(10, 3) * thickness_cm),
                                                                                                        wave_type),
                 "linear", plot_g2, plot_g4, plot_g6, plot_epsilon, e_amp, multi_folder, "", False, True, 50, -50)
                 """
    """
    # Plot the polarization vs. time
    plotting(p, t_plot, "Polarization (uC/cm^2)", "Time (ns)",
             "P-T Hysteresis ({}) g2_{}E13 g4_{}E11 g6_{}E8 ep_{}E13 Vprog_{}V Vread_{}V wave_{}".format(device_type,
                                                                                                   plot_g2, plot_g4,
                                                                                                   plot_g6,
                                                                                                   plot_epsilon,
                                                                                                         (e_amp * math.pow(10, 3) * thickness_cm), (e_read * math.pow(10, 3) * thickness_cm), wave_type),
             "linear", plot_g2, plot_g4, plot_g6, plot_epsilon, e_amp, multi_folder, "", False, False, 0, 0)
    # Plot the e-field vs. time
    plotting(v_g, t_plot, "Voltage (V)", "Time (ns)",
             "Applied E-Field ({}) g2_{}E13 g4_{}E11 g6_{}E8 ep_{}E13 Vprog_{}V Vread_{}V wave_{}".format(device_type,
                                                                                                    plot_g2, plot_g4,
                                                                                                    plot_g6,
                                                                                                    plot_epsilon, (e_amp * math.pow(10, 3) * thickness_cm),
                                                                                                                    (e_read * math.pow(10, 3) * thickness_cm),
                                                                                                    wave_type),
             "linear", plot_g2, plot_g4, plot_g6, plot_epsilon, e_amp, multi_folder, "", False, False, 0, 0)
    plotting(v_t, t_plot, "V_T (V)", "Time (ns)",
             "Threshold Voltage ({}) g2_{}E13 g4_{}E11 g6_{}E8 ep_{}E13 Vprog_{}V Vread_{}V wave_{}".format(device_type,
                                                                                                      plot_g2, plot_g4,
                                                                                                      plot_g6,
                                                                                                      plot_epsilon,
                                                                                                                      (e_amp * math.pow(10, 3) * thickness_cm), (e_read * math.pow(10, 3) * thickness_cm),
                                                                                                      wave_type),
             "linear", plot_g2, plot_g4, plot_g6, plot_epsilon, e_amp, multi_folder, "", False, False, 0, 0)
    plotting(i_plot, t_plot, "Current (A)", "Time (ns)",
             "Drain current ({})- 1 g2_{}E13 g4_{}E11 g6_{}E8 ep_{}E13 Vprog_{}V Vread_{}V wave_{}".format(device_type,
                                                                                                     plot_g2, plot_g4,
                                                                                                     plot_g6,
                                                                                                     plot_epsilon,
                                                                                                                     (e_amp * math.pow(10, 3) * thickness_cm),
                                                                                                                     (e_read * math.pow(10, 3) * thickness_cm),
                                                                                                     wave_type),
             "linear", plot_g2, plot_g4, plot_g6, plot_epsilon, e_amp, multi_folder, "Vd: {}V".format(v_d), False, False, 0, 0)
    plotting(i_plot, v_g, "Current (A)", "Gate Voltage (V)",
             "Drain current ({})- 2 g2_{}E13 g4_{}E11 g6_{}E8 ep_{}E13 Vprog_{}V Vread_{}V wave_{}".format(
                 device_type,
                                                                                                     plot_g2, plot_g4,
                                                                                                     plot_g6,
                                                                                                     plot_epsilon,
                 (e_amp * math.pow(10, 3) * thickness_cm), (e_read * math.pow(10, 3) * thickness_cm), wave_type),
             "linear", plot_g2, plot_g4, plot_g6, plot_epsilon, e_amp, multi_folder, "Vd: {}V".format(v_d), False, False, 0, 0)
    """
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
                id_vg = -(w * c_ox_p * mobility / l) * (v_d * (v_g - vt) - math.pow(v_d, 2) / 2)
            elif abs(v_g - vt) <= abs(v_d):  # saturation, mobility controls nmos or pmos
                id_vg = -(w * c_ox_p * mobility / l) * (math.pow(v_g - vt, 2) / 2)
            v_g_plot.append(v_g)
            idvg_plot.append(id_vg)
    # plotting(idvg_plot, v_g_plot, "Current (uA)", "Gate Voltage (V)", "{} Id-Vg {}V Vt".format(device_type, round(vt, 3)), "linear", 1, 1, 1, 1, 1, False, "Vd:{}".format(v_d), False)

    return idvg_plot, v_g_plot, vt


def trad_Id_Vd_calc(vt):
    # Section of loops and whatnot to look at traditional Id-Vd curves for nmos and pmos devices
    vd_plot = []
    idvd_plot = []
    legend_plot = []
    vd_meta_plot = []
    idvd_meta_plot = []
    if device_type == "nmos":
        for v_g in create_range(0, 5, 1):
            vd_plot = []
            idvd_plot = []
            for vd in create_range(0, 5, 0.1):
                if v_g < vt:
                    id_vd = 0
                elif v_g - vt > vd:  # linear, mobility controls nmos or pmos
                    id_vd = (w * c_ox_p * mobility / l) * (vd * (v_g - vt) - math.pow(vd, 2) / 2)
                elif v_g - vt <= vd:  # saturation, mobility controls nmos or pmos
                    id_vd = (w * c_ox_p * mobility / l) * (math.pow(v_g - vt, 2) / 2)
                vd_plot.append(vd)
                idvd_plot.append(id_vd)
            legend_plot.append("Vg:{}".format(v_g))
            vd_meta_plot.append(vd_plot)
            idvd_meta_plot.append(idvd_plot)
        plotting(idvd_meta_plot, vd_meta_plot, "Current (A)", "Drain Voltage (V)",
                 "{} Id-Vd {}V Vt  g2_{}E13 g4_{}E11 g6_{}E8 ep_{}E13 Eamp_{}kvcm-1 E-read_{}kvcm-1 wave_{}".format(device_type, round(vt, 3), math.floor(g2 / math.pow(10, 13)), math.floor(g4 / math.pow(10, 11)), math.floor(g6 / math.pow(10, 8)), math.floor(epsilon / math.pow(10, 13)), e_prog, e_read, wave_type), "linear", 1, 1, 1, 1, 1, False, legend_plot, True, False, 0, 0)
    else:
        for v_g in create_range(0, -5, -1):
            vd_plot = []
            idvd_plot = []
            for vd in create_range(0, -5, -0.1):
                if v_g > vt:
                    id_vd = 0
                elif abs(v_g - vt) > abs(vd):  # linear, mobility controls nmos or pmos
                    id_vd = -(w * c_ox_p * mobility / l) * (vd * (v_g - vt) - math.pow(vd, 2) / 2)
                elif abs(v_g - vt) <= abs(vd):  # saturation, mobility controls nmos or pmos
                    id_vd = -(w * c_ox_p * mobility / l) * (math.pow(v_g - vt, 2) / 2)
                vd_plot.append(vd)
                idvd_plot.append(id_vd)
            legend_plot.append("Vg:{}".format(v_g))
            vd_meta_plot.append(vd_plot)
            idvd_meta_plot.append(idvd_plot)
        plotting(idvd_meta_plot, vd_meta_plot, "Current (A)", "Drain Voltage (V)",
                 "{} Id-Vd {}V Vt  g2_{}E13 g4_{}E11 g6_{}E8 ep_{}E13 Eamp_{}kvcm-1 E-read_{}kvcm-1 wave_{}".format(device_type, round(vt, 3), math.floor(g2 / math.pow(10, 13)), math.floor(g4 / math.pow(10, 11)), math.floor(g6 / math.pow(10, 8)), math.floor(epsilon / math.pow(10, 13)), e_prog, e_read, wave_type), "linear", 1, 1, 1, 1, 1, False, legend_plot, True, False, 0, 0)

    return


if __name__ == '__main__':
    """Wave type"""
    wave_type = "triangle"
    # wave_type = "triangle_pulse"
    # wave_type = "pulse_only"
    # wave_type = "pos_pulse_only"
    # wave_type = "neg_pulse_only"
    # wave_type = "read_and_write"
    # wave_type = "increase_pos_pulse_amp_only"
    # wave_type = "increase_pos_pulse_length_only"

    """nmos or pmos"""
    device_type = "nmos"    # p-doped silicon, with a n-channel (electron is dominant carrier)
    # device_type = "pmos"  # n-doped silicon, with a p-channel (hole is dominant carrier)

    if device_type == "nmos":
        phi_f = (k * T / q) * math.log(na / ni)  # fermi level in p-si
        phi_si = chi_si + eg_si / 2 + phi_f  # fermi work function of p-si
        dopant = na
        mobility = mu_n
        v_d = 1 * drain_bias
        gamma = 1
    else:
        phi_f = (k * T / q) * math.log(nd / ni)  # fermi level in n-si
        phi_si = chi_si + eg_si / 2 - phi_f  # fermi work function of n-si
        dopant = nd
        mobility = mu_p
        v_d = -1 * drain_bias
        gamma = -1

    v_fb = (phi_m - phi_si)  # flatband voltage
    phi_ms = v_fb  # difference in workfunctions of metal and silicon is the same as the flatband voltage
    sqrt_input = 4 * ep_si * ep_naught * q * dopant * phi_f
    vt_naught = v_fb + (2 * phi_f) + (math.sqrt(sqrt_input) / c_ox_p)  # initial threshold voltage of the transistor
    print("Vt_naught:", vt_naught)

    # For SOFM+FeFET paper, time_step = 2.5 * math.pow(10, -12)
    # For measured data, time_step = 5 * math.pow(10, -9)
    time_step = 5 * math.pow(10, -9)  # time step
    """Triangular wave definitions"""
    time_start = 0
    # For SOFM+FeFET paper, time_length = 5 * math.pow(10, -10)
    # For measured data, time_length = 1 * math.pow(10, -6)
    time_length = 1 * math.pow(10, -6)  # time pulse (amount of time for each segment of the triangular wave)
    time_period = 4 * time_length  # Total time period for full triangular pulse is 4*time_pulse

    """Square wave pulse definitions"""
    # 1E-6 for measured data
    # 5E-10 for measured data
    time_pulse_start = 1 * math.pow(10, -6)  # pulse start time
    if wave_type == "triangle_pulse":
        time_pulse_start += time_period
    # 1E-6 for measured data
    # 5E-10 for measured data
    time_pulse_length = 1 * math.pow(10, -6)  # pulse length time
    time_pulse_period = 4 * time_pulse_length  # pulse period
    if wave_type != "triangle":
        number_of_periods = 20
    else:
        number_of_periods = 1
    number_of_pulses = number_of_periods * 2
    time_pulse_scale = 0.5

    """Voltage definitions"""
    v_start = 0
    v_end = 40  #3 for pulses, 40 for amplitude
    v_end_max = 20
    v_start_kv = v_start / math.pow(10, 3)
    v_end_kv = v_end / math.pow(10, 3)
    """E-field variables, in units of kV/cm"""
    e_start = v_start_kv / thickness_cm
    e_end = v_end_kv / thickness_cm
    #e_pend = v_end_kv / thickness_cm
    #e_prog = e_pend
    #e_read = e_prog / 10
    #e_nend = -v_end_kv / thickness_cm
    #if wave_type == "neg_pulse_only":
        #e_pend *= -1

    t_plot = []
    e_plot = []

    if wave_type == "triangle":
        time_range = create_range(time_start, (time_period * number_of_periods) + time_length, time_step)
        # time_range = create_range(time_start, (time_period * number_of_periods), time_step)
    elif wave_type == "triangle_pulse":
        time_range = create_range(time_start, time_pulse_start + (time_pulse_period * number_of_periods), time_step)
    elif wave_type == "increase_pos_pulse_length_only":
        p = 1
        total_pulse_time = 0
        while p <= number_of_pulses:
            total_pulse_time += ((p - 1) * time_pulse_length * time_pulse_scale) + time_pulse_length
            p += 1
        time_range = create_range(time_start,
                                  time_pulse_start + (time_pulse_length * number_of_pulses) + total_pulse_time,
                                  time_step)
    else:
        time_range = create_range(time_start, time_pulse_start + (time_pulse_period * number_of_periods), time_step)
    # print("time_range:", len(time_range))

    for t in time_range:
        t_plot.append(t / math.pow(10, -9))

    """Fitting parameters"""
    # g_0 = 4.9 * math.pow(10, 8)  # a_11 =? a_11
    epsilon_min = 1 * math.pow(10, 13)  # scaling factor for e_field
    g2_min = -1 * math.pow(10, 14)  # Jm/C^2
    g4_min = -1 * math.pow(10, 13)  # Jm^5/C^4
    # g6_min = 1 * math.pow(10, 13)  # Jm^9/C^6


    epsilon_max = 20 * math.pow(10, 13)
    g2_max = -20 * math.pow(10, 14)
    g4_max = -20 * math.pow(10, 13)
    # g6_max = 20 * math.pow(10, 13)

    # For SOFM+FeFET paper, epsilon = epsilon = 2 * math.pow(10, 22), g2= -1 * math.pow(10, 7), g4 = -1 * math.pow(10, 9), g6 = variable based on g6_max and g6_min
    epsilon = 1 * math.pow(10, 9)
    # epsilon = epsilon_min
    g2 = 1 * math.pow(10, 1)  # Jm/C^2
    # g2 = g2_min
    g4 = 1 * math.pow(10, 1)  # Jm^5/C^4
    # g4 = g4_min
    g6 = 1 * math.pow(10, 8)  # Jm^9/C^6, 1E10 is the LL, 1E18 is the UL for (e=1E16, g2=-1E15, g4=-1E13, t_step = 1E-9, t_length = 1E-6, 1V)
    # 1E16 is the LL, 1E27 is the UL for (e=5E22, g2=-1E7, g4=-1E9, t_step = 1E-13, t_length = 1E-9, 1V)
    # g6 = g6_min

    i = 0
    r = 0
    power_min = 18
    power_max = 25
    power_step = 1
    g6_min = 1 * math.pow(10, power_min)  # Jm^9/C^6
    g6_max = 1 * math.pow(10, power_max)
    number_of_states = 32   #360 states for "analog", 16, 32, or 360
    e_pend = []
    # for the measured data portion
    e_pend = create_range(e_start, e_end, (e_end/number_of_states))
    #For pulses
    """for i in range(number_of_states):
        e_pend.append(e_end)"""


    legend_polarization_meta_plot = []
    polarization_meta_plot = []
    vg_sweep_meta_plot = []
    idvg_meta_meta_plot = []
    vg_meta_meta_plot = []
    legend_idvg_meta_meta_plot = []

    """idvg, vg, vt = trad_Id_Vg_calc(vt_naught)
    idvg_meta_meta_plot.append(idvg)
    vg_meta_meta_plot.append(vg)
    legend_idvg_meta_meta_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(vt_naught, 3)))"""
    # Loop for generating the plots for the SOFM+FeFET Nature paper, looks at a decay in g6 parameter to generate different PE curves
    state_iter = 1
    # g6_conversion_slope = (g6_max - g6_min) / (number_of_states - 1)
    ep_values = []
    g2_values = []
    g4_values = []
    while state_iter <= number_of_states:
        # g6_conversion = g6_conversion_slope * state_iter
        # g6_values.append(g6_max * math.pow(state_iter, -0.2)) # Power decay in g6 corresponds to logarithmic increase in VT
        #g6_values.append(-math.pow(10, 16) * math.log(state_iter, 5) + g6_max)  # Log decay in g6 corresponds to logarithmic increase in VT
        #g6_values.append(g6_max - math.pow(state_iter, 5))
        #g6_values.append(g6_max - math.pow(state_iter, 5))
        # g6_values.append(8 * math.pow(10, 10) * math.pow(state_iter, 5) + math.pow(10, 16))  #
        # g6_values.append(g6_max * math.exp(-1 * state_iter / 2.05)) # exponential decay (decreasing form) in g6 corresponds to exponential increase in VT, 22.5 for "analog" states, 2.05 for 64 states, 1.03 for 32 states, 0.45 and 9E25 for 16 states
        # g6_values.append(g6_max * (1-math.exp(-1 * state_iter/500)) - math.pow(10, 17))   # exponential decay (increasing form) in g6
        # g6_values.append(g6_conversion)
        # g6_values.append(1 * math.pow(10, 81) / math.exp(math.pow(state_iter, 0.04) * 130) + 1.3 * math.pow(10, 18))

        # Equations for amplitude programming
        ep_values.append((33248.6837825599 / state_iter) * math.log10(state_iter / 300) + (3.5 * 33248.6837825599))
        g2_values.append((1460107.51924403 / state_iter) * math.log10(state_iter / 10) + (2 * 1460107.51924403))
        g4_values.append(1 * 24766.2159790265 * math.exp(-2 * state_iter) + (2354.61569279783 * (number_of_states / state_iter)))
        g6_values.append(90 * 354325.344921231 * math.exp(-4.5 * state_iter) + (1.28651793543193 * (number_of_states / state_iter)))
        # Equations for pulse number programming
        """ep_constant = 5 * math.pow(10, 8)
        g2_constant = 2 * math.pow(10, 9)
        g4_constant_1 = 1 * math.pow(10, 12)
        g4_constant_2 = 1 * math.pow(10, 7)
        g6_constant_1 = 1 * math.pow(10, 7)
        g6_constant_1 = 5 * math.pow(10, 11)
        g6_constant_2 = 1 * math.pow(10, 4)
        ep_values.append((ep_constant / state_iter) * math.log10(state_iter / 300) + (3.5 * ep_constant))
        g2_values.append((g2_constant / state_iter) * math.log10(state_iter / 10) + (2 * g2_constant))
        g4_values.append(g4_constant_1 * math.exp(-2 * state_iter) + (g4_constant_2 * (number_of_states / state_iter)))
        g6_values.append(g6_constant_1 * math.exp(-4.5 * state_iter) + (g6_constant_2 * (number_of_states / state_iter)))"""
        state_iter += 1

    # Linear decay in g6 corresponds to polynomial (5th power?) increase in VT or is it actually a power relationship?
    # for power in create_range(power_max, power_min, power_step):
        # for integer in create_range(1, 10, 0.1):
            # g6_values.append(integer * math.pow(10, power))

    # g6_values.append(g6)

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

        plot_epsilon = math.floor(ep_iter / math.pow(10, 21))
        plot_g2 = math.floor(g2_iter / math.pow(10, 6))
        plot_g4 = math.floor(g4_iter / math.pow(10, 8))
        plot_g6 = math.floor(g6_iter / math.pow(10, 15))

        # trad_Id_Vd_calc(vt_naught)

        idvg, vg, vt = trad_Id_Vg_calc(vt_naught)
        idvg_meta_plot.append(idvg)
        vg_meta_plot.append(vg)
        legend_idvg_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(vt_naught, 3)))

        polarization, vg_sweep_plot = runge_kutta(ep_iter, e_field_iter, e_plot, t_plot, time_range, time_period, time_step, g2_iter, g4_iter, g6_iter, False, wave_type, time_pulse_start, time_pulse_length, number_of_periods, time_pulse_scale)
        # polarization, vg_sweep_plot = runge_kutta(epsilon, e_field_iter, e_plot, t_plot, time_range, time_period, time_step, g2, g4, g6, False, wave_type, time_pulse_start, time_pulse_length, number_of_periods, time_pulse_scale)

        polarization_meta_plot.append(polarization)
        vg_sweep_meta_plot.append(vg_sweep_plot)
        legend_polarization_meta_plot.append("g6_{}E15".format(plot_g6))

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

    plotting(polarization_meta_plot, vg_sweep_meta_plot, "Polarization ($μC/cm^2$)", "Voltage ($V$)",
             "Polarization Hysteresis Loops", "linear", 1, 1, 1, 1, 1, False, legend_polarization_meta_plot, False, True, True, 35, -35, False, [], "", False)  #Normally title is :"P-E Hysteresis ({}) g2_{}E6 g4_{}E8 g6_1E{}-1E{} ep_{}E21 Vprog_{}V wave_{}".format(device_type, plot_g2,plot_g4, power_min, power_max,plot_epsilon, v_end, wave_type)

    # Block of code to plot stuff for SOFM+FeFET paper
    #v_plot and "Programming Voltage" for amplitude, p_plot and "Programming Pulse" for pulse number
    plotting(g6_plot, v_plot, "$g_6$ Value ($Jm^9$/$C^6$)", "Programming Voltage", "g6 Parameter Change", "log",
             1, 1, 1, 1, 1, False, ["$g_6$ Values", "$V_T$, $P_{}$".format("r+"), "$V_T$, $P_{}$".format("r-")],
             True, True, False, 0, 0, True, [vt_pos_list, vt_neg_list], "Threshold Voltage ($V$)", False)

    plotting(g4_plot, v_plot, "$g_4$ Value ($Jm^5$/$C^4$)", "Programming Voltage", "g4 Parameter Change", "linear",
             1, 1, 1, 1, 1, False, ["$g_4$ Values", "$V_T$, $P_{}$".format("r+"), "$V_T$, $P_{}$".format("r-")],
             True, True, False, 0, 0, True, [vt_pos_list, vt_neg_list], "Threshold Voltage ($V$)", False)
    plotting(g2_plot, v_plot, "$g_2$ Value ($Jm$/$C^2$)", "Programming Voltage", "g2 Parameter Change", "linear",
             1, 1, 1, 1, 1, False, ["$g_2$ Values", "$V_T$, $P_{}$".format("r+"), "$V_T$, $P_{}$".format("r-")],
             True, True, False, 0, 0, True, [vt_pos_list, vt_neg_list], "Threshold Voltage ($V$)", False)
    plt.rcParams.update({'axes.formatter.limits': [0, 3]})
    plotting(ep_plot, v_plot, "$ep$ Value (Unitless)", "Programming Voltage", "ep Parameter Change", "linear",
             1, 1, 1, 1, 1, False, ["$ep$ Values", "$V_T$, $P_{}$".format("r+"), "$V_T$, $P_{}$".format("r-")],
             True, True, False, 0, 0, True, [vt_pos_list, vt_neg_list], "Threshold Voltage ($V$)", False)


    if device_type == "nmos":
        plotting(idvg_meta_meta_plot, vg_meta_meta_plot, "Current ($A$)", "Gate Voltage ($V$)",
                 "Threshold Voltage Shift",
                 "log", 1, 1, 1, 1, 1, False, legend_idvg_meta_meta_plot, False, True, True, math.pow(10, 0), math.pow(10, -4), False, [], "", True)    #Normally title is: "Meta - {} Id-Vg {}V Vt0  g2_{}E6 g4_{}E8 g6_1E{}-1E{} ep_{}E21 Vprog_{}V wave_{}".format(device_type, round(vt_naught, 3), plot_g2,plot_g4, power_min, power_max,plot_epsilon, v_end, wave_type),
    else:
        plotting(idvg_meta_meta_plot, vg_meta_meta_plot, "Current (A)", "Gate Voltage (V)",
                 "Meta - {} Id-Vg {}V Vt0  g2_{}E6 g4_{}E8 g6_1E{}-1E{} ep_{}E21 Vprog_{}V wave_{}".format(
                     device_type, round(vt_naught, 3), plot_g2,
                     plot_g4, power_min, power_max,
                     plot_epsilon, v_end, wave_type),
                 "log", 1, 1, 1, 1, 1, False, legend_idvg_meta_meta_plot, False, True, True, 0, -0.1, False, [], "")

    vt_list_array = np.array(vt_list)
    g6_list_array = np.array(g6_list)
    g4_list_array = np.array(g4_list)
    g2_list_array = np.array(g2_list)
    ep_list_array = np.array(ep_list)
    vprog_list_array = np.array(vprog_list)
    pprog_list_array = np.array(pprog_list)

    # For measured data amplitude
    #df = pd.DataFrame({"VT": vt_list_array, "g6 value": g6_list_array, "g4 value": g4_list_array, "g2 value": g2_list_array, "ep value": ep_list_array, "programming Voltage": vprog_list_array})
    #df.to_csv("vt list {} g2_{}E6 g4_{}E8 g6_1E{}-1E{} ep_{}E21 Vprog_{}V wave_{}.csv".format(device_type, plot_g2, plot_g4, power_min, power_max, plot_epsilon, v_end, wave_type), index=False)
    # For programming pulses
    df = pd.DataFrame({"VT": vt_list_array, "g6 value": g6_list_array, "g4 value": g4_list_array, "g2 value": g2_list_array, "ep value": ep_list_array, "programming Voltage": vprog_list_array, "programming pulse": pprog_list_array})
    df.to_csv("vt list {} g2_{}E6 g4_{}E8 g6_1E{}-1E{} ep_{}E21 Vprog_{}V wave_{}.csv".format(device_type, plot_g2, plot_g4, power_min, power_max, plot_epsilon, v_end, wave_type), index=False)

    # Looking at changing read voltage impact, but apparently this isn't important
    """while e_read <= 100:
        idvg_meta_plot = []
        vg_meta_plot = []
        legend_idvg_plot = []

        idvg, vg, vt = trad_Id_Vg_calc(vt_naught)
        idvg_meta_plot.append(idvg)
        vg_meta_plot.append(vg)
        legend_idvg_plot.append("Vd:{}V, Vt:{}V".format(v_d, round(vt_naught, 3)))

        polarization = runge_kutta(epsilon, e_pend, e_plot, t_plot, time_range, time_period, time_step, g2, g4, g6, False,
                               wave_type, time_pulse_start, time_pulse_length, number_of_periods, time_pulse_scale)
        plotting(idvg_meta_plot, vg_meta_plot, "Current (A)", "Gate Voltage (V)",
                 "{} Id-Vg {}V Vt0  g2_{}E13 g4_{}E11 g6_{}E8 ep_{}E13 Eamp_{}kvcm-1 E-read_{}kvcm-1 wave_{}".format(
                     device_type, round(vt_naught, 3), math.floor(g2 / math.pow(10, 13)),
                     math.floor(g4 / math.pow(10, 11)), math.floor(g6 / math.pow(10, 8)),
                     math.floor(epsilon / math.pow(10, 13)), e_prog, e_read, wave_type), "linear", 1, 1, 1, 1, 1, False,
                 legend_idvg_plot, True, True, 0.3, 0)
        e_read += 1
    """
    """
        # Sweep for changing e-field/applied voltage
        v = v_end
        while v <= v_end_max:
            # Voltage definitions
            v_end_kv = v / math.pow(10, 3)
            # E-field variables, in units of kV/cm
            e_pend = v_end_kv / thickness_cm
            e_nend = -v_end_kv / thickness_cm
    
            e_range_period_1 = e_pend - e_start
            e_step_period_1 = e_range_period_1 / (time_length / time_step)
            e_range_period_2 = e_pend - e_nend
            e_step_period_2 = -e_range_period_2 / ((2 * time_length) / time_step)
            e_range_period_3 = e_nend - e_start
            e_step_period_3 = -e_range_period_3 / (time_length / time_step)
    
            t_plot = []
    
            if wave_type == "triangle":
                e_range_1 = my_range(e_start, e_pend, e_step_period_1)
                e_range_2 = my_range(e_pend, e_nend, e_step_period_2)
                e_range_3 = my_range(e_nend, e_start, e_step_period_3)
                e_plot = e_range_1 + e_range_2 + e_range_3 + e_range_1
                time_range = my_range(time_start, time_period, time_step)
            else:
                time_range = my_range(0, time_period, time_step)
    
            if (len(time_range) < len(e_plot)):
                difference = len(e_plot) - len(time_range)
                i = 0
                while i < difference:
                    time_range.append(time_range[-1] + time_step)
                    i += 1
            elif (len(e_plot) < len(time_range)):
                difference = len(time_range) - len(e_plot)
                i = 0
                while i < difference:
                    e_plot.append(0)
                    i += 1
    
            runge_kutta(epsilon, e_pend, e_plot, t_plot, time_range, time_period, time_step, g2, g4, g6, True, wave_type, time_pulse_start, time_pulse_length, number_of_periods)
            
            v += 1
    """

    """
        # Nested loops for looking at all fitting parameters  
        while epsilon <= epsilon_max:
            # print("ep: {}E15".format(math.floor(epsilon / math.pow(10, 15))))
            runge_kutta(epsilon, e_pend, e_plot, t_plot, time_range, time_period, time_step, g2, g4, g6, True, wave_type, time_pulse_start, time_pulse_length, number_of_periods)
            g2 = g2_min
            while g2 >= g2_max:
                # print("g2: {}E5".format(math.floor(g2 / math.pow(10, 5))))
                runge_kutta(epsilon, e_pend, e_plot, t_plot, time_range, time_period, time_step, g2, g4, g6, True, wave_type, time_pulse_start, time_pulse_length, number_of_periods)
                g4 = g4_min
                while g4 >= g4_max:
                    print("g4: {}E7".format(math.floor(g4 / math.pow(10, 7))))
                    g6 = g6_min
                    while g6 <= g6_max:
                        print("g6: {}E9".format(math.floor(g6 / math.pow(10, 9))))
                        runge_kutta(epsilon, e_pend, e_plot, t_plot, time_range, time_period, time_step, g2, g4, g6, True, wave_type, time_pulse_start, time_pulse_length, number_of_periods)
                        g6 += 1 * math.pow(10, 10)
                    g4 += -1 * math.pow(10, 9)
                g2 += -1 * math.pow(10, 7)
            
            r += i % 2
            #print("r:", r)
            i += 1
            if i % 2 == 0:
                epsilon += 5 * math.pow(10, 15 + r-1)
            else:
                epsilon += 4 * math.pow(10, 15 + r)
    """
