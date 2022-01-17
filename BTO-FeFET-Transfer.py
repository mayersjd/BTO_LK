"""
Joshua Mayersky
BTO ferroelectric polarization simulation based on landau khalatnikov equations
Tae Kwon Song, "Landau-Khalatnikov Simulations for Ferroelectric Switching in Ferroelectric Random Access Memory Application", Journal of the Korean Physical Society, Vol. 46, No. 1, January 2005, pp. 59

To use: change fitting parameters epsilon, g_2, g_4, and g_6 as desired. Other parameters to play with are applied votlage, film thickness, and the time scales for pulses and the time step for calculating new values
"""

import math
import os
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
import numpy as np


"""Plot Fonts and Text size"""
plt.rcParams['font.size'] = 16
plt.rcParams['font.serif'] = "Times New Roman"

"""Polarization lists"""
bto_0min = [26.6032443438545 * math.pow(10, -6), -18.4175208452096 * math.pow(10, -6)]   # Polarization for +20V and -20V in C/cm^2
bto_30min = [16.80874827319 * math.pow(10, -6), -12.8903907077738 * math.pow(10, -6)]   # Polarization for +20V and -20V in C/cm^2
bto_60min = [7.14012215670369 * math.pow(10, -6), -7.00656333401504 * math.pow(10, -6)]   # Polarization for +20V and -20V in C/cm^2
bto_120min = [10.8047044250526 * math.pow(10, -6), -1.92776351357426 * math.pow(10, -6)]   # Polarization for +20V and -20V in C/cm^2
bto_list = [bto_0min, bto_30min, bto_60min, bto_120min]

"""Global material constants/parameters"""
# k = 8.62 * math.pow(10, -5)   # Boltzmann constant (eV/K)
k = 1.38 * math.pow(10, -23)  # Boltzmann constant (J/K)
T = 300  # room temperature (K)
q = 1.602 * math.pow(10, -19)  # Charge of electron (C) or the electron volt conversion of J to eV (J)
na = 1 * math.pow(10, 21)  # p-doping concentration of the substrate, used in nmos (1/cm^3)
nd = 1 * math.pow(10, 18)  # n-doping concentration of the substrate, used in pmos (1/cm^3)
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
area = math.pow(0.025, -4)  # area of devices in cm^2
l = 130 * math.pow(10, -9)  # length of the transistor (130nm)
wn = 200 * math.pow(10, -9)  # width of the nmos transistor (200nm)
wp = 500 * math.pow(10, -9)  # width of the pmos transistor (500nm)
w = 0   # width of transistor currently being simulated
dopant = 0  # variable for dopant concentration, either na or nd (1/cm^3)
mobility = 0  # variable for carrier mobility, either mu_n or mu_p (cm^2/V*s)

"""Thickness of FE material"""
thickness_nm = 105.7 * math.pow(10, -9)
thickness_cm = thickness_nm / math.pow(10, -2)

"""Calculated material parameters"""
phi_f = 0  # fermi level in si
phi_si = 0  # fermi work function of n-si
v_fb = (phi_m - phi_si)  # flatband voltage
phi_ms = v_fb  # difference in workfunctions of metal and silicon is the same as the flatband voltage
c_ox = ep_naught * ep_ox * area / thickness_cm  # oxide capacitance (F)
c_ox_p = ep_naught * ep_ox / thickness_cm  # oxide capacitance (F/cm^2)
#print("cox", c_ox)
#print("cox'", c_ox_p)

c_ox_meas = 9.34498537932526 * math.pow(10, -11)    #BTO 27 cap value at 3.4V
#c_ox_meas = 2.92489617133692 * math.pow(10, -10)    #BTO 29 COX calc value

ep_ox_calc = c_ox_meas * thickness_cm / ep_naught
#print("ep_ox:", ep_ox, "ep_ox_calc:", ep_ox_calc)

"""Applied voltage"""
drain_bias = 3.3  # drain bias (V)
v_d = 0

"""Other global variables for referencing"""
device_type = ""
vd_meta_plot = []
vg_meta_plot = []
idvd_meta_plot = []
idvg_meta_plot = []
legend_idvd_plot = []
legend_idvg_plot = []


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
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    colors = ["blue", "green", "red", "black", "magenta", "orange", "cyan"]
    colorcycler = cycle(colors)

    fig, ax = plt.subplots()
    for k in range(0, len(x_axis)):
        ax.plot(x_axis[k], y_axis[k], linestyle=next(linecycler), color=next(colorcycler), linewidth=1, label="{}".format(legend[k]))

    ax.set_xlabel(x_label)
    ax.legend(loc='lower right', fontsize="small")
    #ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=1, borderaxespad=10, frameon=False)
    if yaxislim:
        axes = plt.gca()
        axes.set_ylim(ymin, ymax)
    if scale == "linear":
        ax.set_ylabel("{}".format(y_label))
    else:
        ax.set_ylabel(y_label)
        ax.set_yscale("log")
    plt.tight_layout()

    fn = "{}".format(title)

    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    folder_name = plots_folder

    plt.savefig(folder_name + "/{}.png".format(fn), format="png")
    plt.close()


def trad_Id_Vg_calc(vt, bto, vprog):
    # Section of loops and whatnot to look at traditional Id-Vg curves for nmos and pmos devices
    vg_plot = []
    idvg_plot = []
    if device_type == "nmos":
        for v_g in create_range(0, 5, 0.1):
            if v_g < vt:
                id_vg = 0
            elif v_g - vt > v_d:  # linear, mobility controls nmos or pmos
                id_vg = (w * c_ox_p * mobility / l) * (v_d * (v_g - vt) - math.pow(v_d, 2) / 2)
            elif v_g - vt <= v_d:  # saturation, mobility controls nmos or pmos
                id_vg = (w * c_ox_p * mobility / l) * (math.pow(v_g - vt, 2) / 2)
            ida_vg = id_vg / area
            vg_plot.append(v_g)
            idvg_plot.append(ida_vg)
    else:
        for v_g in create_range(0, -5, -0.1):
            if v_g > vt:
                id_vg = 0
            elif abs(v_g - vt) > abs(v_d):  # linear, mobility controls nmos or pmos
                id_vg = -(w * c_ox_p * mobility / l) * (v_d * (v_g - vt) - math.pow(v_d, 2) / 2)
            elif abs(v_g - vt) <= abs(v_d):  # saturation, mobility controls nmos or pmos
                id_vg = -(w * c_ox_p * mobility / l) * (math.pow(v_g - vt, 2) / 2)
            ida_vg = id_vg / area
            ida_vg = math.fabs(ida_vg)
            vg_plot.append(v_g)
            idvg_plot.append(ida_vg)

    idvg_meta_plot.append(idvg_plot)
    vg_meta_plot.append(vg_plot)
    legend_idvg_plot.append("BTO {}, {}".format(bto, vprog))
    return


def trad_Id_Vd_calc(vt, bto, vprog):
    # Section of loops and whatnot to look at traditional Id-Vd curves for nmos and pmos devices
    vd_plot = []
    idvd_plot = []
    if device_type == "nmos":
        v_g = 3.3
        for vd in create_range(0, 5, 0.1):
            if v_g < vt:
                id_vd = 0
            elif v_g - vt > vd:  # linear, mobility controls nmos or pmos
                id_vd = (w * c_ox_p * mobility / l) * (vd * (v_g - vt) - math.pow(vd, 2) / 2)
            elif v_g - vt <= vd:  # saturation, mobility controls nmos or pmos
                id_vd = (w * c_ox_p * mobility / l) * (math.pow(v_g - vt, 2) / 2)
            ida_vd = id_vd / area
            vd_plot.append(vd)
            idvd_plot.append(ida_vd)
    else:
        v_g = -3.3
        for vd in create_range(0, -5, -0.1):
            if v_g > vt:
                id_vd = 0
            elif abs(v_g - vt) > abs(vd):  # linear, mobility controls nmos or pmos
                id_vd = -(w * c_ox_p * mobility / l) * (vd * (v_g - vt) - math.pow(vd, 2) / 2)
            elif abs(v_g - vt) <= abs(vd):  # saturation, mobility controls nmos or pmos
                id_vd = -(w * c_ox_p * mobility / l) * (math.pow(v_g - vt, 2) / 2)
            ida_vd = id_vd / area
            vd_plot.append(vd)
            idvd_plot.append(ida_vd)

    idvd_meta_plot.append(idvd_plot)
    vd_meta_plot.append(vd_plot)
    legend_idvd_plot.append("BTO {}, {}".format(bto, vprog))
    return



if __name__ == '__main__':
    """nmos or pmos"""
    device_type = "nmos"  # p-doped silicon, with a n-channel (electron is dominant carrier)
    #device_type = "pmos"  # n-doped silicon, with a p-channel (hole is dominant carrier)

    if device_type == "nmos":
        w = wn
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
        w = wp
        phi_f = (k * T / q) * math.log(nd / ni)  # fermi level in n-si
        phi_si = chi_si + eg_si / 2 - phi_f  # fermi work function of n-si
        dopant = nd
        mobility = mu_p
        v_d = -1 * drain_bias
        v_fb = (phi_m - phi_si)  # flatband voltage
        phi_ms = v_fb  # difference in workfunctions of metal and silicon is the same as the flatband voltage
        sqrt_input = 4 * ep_si * ep_naught * q * dopant * abs(phi_f)
        vt_naught = v_fb - (abs(2 * phi_f)) + (math.sqrt(sqrt_input) / c_ox_p)  # initial threshold voltage of the transistor

    # print("phi_f:", phi_f)
    # print("phi_si:", phi_si)
    # print("V_fb:", v_fb)
    # print("Vt_naught:", vt_naught)

    bto = 0
    for bto_iter in bto_list:
        bto += 1
        iter = 0
        for pr in bto_iter:
            if iter % 2 == 0:
                vprog = "$V_{T,Min}$"
            else:
                vprog = "$V_{T,Max}$"
            vt = vt_naught - (pr / c_ox_p)
            #print(bto, vprog, vt)
            trad_Id_Vd_calc(vt, bto, vprog)
            trad_Id_Vg_calc(vt, bto, vprog)
            iter += 1

    plotting(idvd_meta_plot, vd_meta_plot, "$J_D$ ($A/cm^2$)", "$V_D$ (V)", "{} Id-Vd".format(device_type), "linear", legend_idvd_plot, False, 0, 0)
    #plotting(idvg_meta_plot, vg_meta_plot, "Abs $J_D$ ($|A/cm^2|$)", "$V_G$ (V)", "{} Id-Vg".format(device_type), "log", legend_idvg_plot, False, 0, 0)
    plotting(idvg_meta_plot, vg_meta_plot, "$J_D$ ($A/cm^2$)", "$V_G$ (V)", "{} Id-Vg".format(device_type), "log",legend_idvg_plot, False, 0, 0)

    """Saving simulated data to a csv file"""
    #vd_csv = np.array(vd_meta_plot).T.tolist()
    idvd_meta_plot.append(vd_meta_plot[0])
    idvd_csv = np.array(idvd_meta_plot).T.tolist()
    legend_idvd_plot.append("V_D")
    idvd_columns = np.array(legend_idvd_plot).T.tolist()

    #vg_csv = np.array(vg_meta_plot).T.tolist()
    idvg_meta_plot.append(vg_meta_plot[0])
    idvg_csv = np.array(idvg_meta_plot).T.tolist()
    legend_idvg_plot.append("V_G")
    idvg_columns = np.array(legend_idvg_plot).T.tolist()

    df_1 = pd.DataFrame(idvd_csv, columns=[idvd_columns])
    df_1.to_csv("{} Id-Vd.csv".format(device_type), index=False)
    df_2 = pd.DataFrame(idvg_csv, columns=[idvg_columns])
    df_2.to_csv("{} Id-Vg.csv".format(device_type), index=False)