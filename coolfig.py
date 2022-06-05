#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ipywidgets as wd
import matplotlib.pyplot as plt
import cool as cc

# %matplotlib inline  # uncomment for inline figure
# %matplotlib qt      # uncomment for figure in separate window
plt.rcParams["figure.figsize"] = (10, 7.7)
font = {'size': 16}
plt.rc('font', **font)

Kθ, Kw = 1e10, 0        # Gain factors of the P-controllers
β = 0.16                # By-pass factor of the cooling coil

m, mo = 3.1, 1.         # kg/s, mass flow rate, supply and outdoor air
θo, φo = 32., 0.5       # °C, -, outdoor air temperature and relative humidity
θ5sp, φ5sp = 26., 0.5   # °C, -, indoor air set points

mi = 1.35               # kg/s, mass flow rate of infiltration air
UA = 675.               # W/K, overall heat transfet coeffcient
QsBL, QlBL = 34000., 4000.    # W, sensible & latent auxiliar heat

parameters = m, mo, β, Kθ, Kw
inputs = θo, φo, θ5sp, φ5sp, mi, UA, QsBL, QlBL


def figkw():
    cool0 = cc.MxCcRhTzBl(parameters, inputs)
    Kw = 1e10
    cool0.actual[4] = Kw
    wd.interact(cool0.CAV_wd, θo=(26, 34), φo=(0.4, 1),
                θ5sp=(20, 28), φ5sp=(0.30, 1, 0.01),
                mi=(0.1, 3, 0.1), UA=(500, 800, 10),
                QsBL=(0, 60_000, 500), QlBL=(0, 20_000, 500))
    print("Control of indoor air temperature & humidity (θ5, φ5)")
    print("in CAV systems with reheating")


def fig():
    cool1 = cc.MxCcRhTzBl(parameters, inputs)
    wd.interact(cool1.CAV_wd, θo=(26, 34), φo=(0.4, 1),
                θ5sp=(20, 28), φI5sp=(0.4, 1, 0.01),
                mi=(0.5, 3, 0.1), UA=(500, 8000, 10),
                QsBL=(0, 60_000, 500), QlBL=(0, 20_000, 500))
    print("Control of indoor air temperature (θ5)")
    print("in CAV systems without reheating")
