import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from valladopy.astro.twobody.frame_conversions import rv2coe, coe2rv
from valladopy.astro.celestial.moon import position as moon_pos
from valladopy.astro.celestial.sun import position as sun_pos
from valladopy.astro.celestial.sun import in_light
from valladopy.astro.time.data import iau80in
from src.gravity import get_acc_grav
from src.drag import get_acc_drag
from src.thridbody import get_acc_3rd_body
import math
import time


MU = 398600.4415 # in km³/s² from Fundamentals of Astrodynamics and Applications - Davie A. Vallado
MU_SUN = 1.32712440042e11
MU_MOON = 4902.799
R_EARTH = 6378.1363 # From FUNDAMENTALS OF ASTRODYNAMICS - Karel F. Wakker, 2015
J2 = 0.001082626174
J4 = -1.6198976e-06
RAD2DEG = 180.0/math.pi
SEC_IN_DAY = 3600*24
iau80 = iau80in("data")
def deriv(state: ArrayLike, params: dict, date: float) -> np.ndarray:
    """_summary_

    Args:
        state (ArrayLike): _description_
        params (dict): _description_

    Returns:
        np.ndarray: _description_
    """
    satellite = params["satellite_properties"]["physical_properties"]
    perturbations = params["integration"]["perturbations"]

    acc = get_all_acc(state=state, perturbations=perturbations, satellite=satellite, date=date)
    vel = state[3:6]
    deriv = np.zeros(6)
    deriv[0:3] = vel
    deriv[3:6] = acc

    return deriv

def get_all_acc(state: ArrayLike, perturbations: dict, satellite: dict, date: float) -> np.ndarray:
    """_summary_

    Args:
        state (ArrayLike): _description_
        perturbations (dict): _description_

    Returns:
        np.ndarray: _description_
    """
    all_acc = get_acc_grav(state=state, geopotenital_order=perturbations["Geopotential"])
    if(perturbations["drag"]):
        all_acc += get_acc_drag(state=state, sat=satellite, date=date, frame=iau80)
    if(perturbations["Moon"] or perturbations["Sun"]):
        all_acc += get_acc_3rd_body(state=state, date=date, bodies=perturbations)
    if(perturbations["srp"]):
        all_acc += get_acc_srp(state=state, date=date, satellite=satellite)

    return all_acc


def get_acc_srp(state: ArrayLike, date: float, satellite: dict) -> np.ndarray:
    """Estimates the Accelaration due to Gravity at a given moment

    Args:
        state (ArrayLike): State Vector at given time

    Returns:
        np.ndarray: Accelaration vector
    """
    if(not in_light(state[0:3], date)):
        return np.zeros(3)
    # Get Sun position
    pos_earth_3rd, _, _ = sun_pos(date)
    # Get vector from Sun to Satellite
    pos_sat_3rd = pos_earth_3rd - state[0:3]
    srp_effect = -4.57e-9*((satellite["c_r"]*satellite["area"])/satellite["mass"])
    return srp_effect*(pos_sat_3rd*(1/np.linalg.norm(pos_sat_3rd)))


def integrate_step_rk4(state: ArrayLike, step: float, params: dict, date:float, fun_deriv=deriv) -> np.ndarray:

    k1 = fun_deriv(state=state, params=params, date=date)
    state_1 = state + k1*(step/2)

    k2 = fun_deriv(state=state_1, params=params, date=(date+0.5*step/SEC_IN_DAY))
    state_2 = state + k2*(step/2)

    k3 = fun_deriv(state=state_2, params=params, date=(date+0.5*step/SEC_IN_DAY))
    state_3 = state + k3*(step)

    k4 = fun_deriv(state=state_3, params=params, date=(date+step/SEC_IN_DAY))

    return (1.0/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)*step




state_init = np.array([7100.0, 0.0, 0.0, 0.0, 7.5, 0.0])
state_dot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
state_dot[0:3] = state_init[3:6]
state_dot[3:6] = get_acc_grav(state_init)
alt = []
vel = []
t = []
# for i in range(0, 5400000):
#     state_init = state_init + state_dot*0.01
#     state_dot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     state_dot[0:3] = state_init[3:6]
#     state_dot[3:6] = -(MU/(np.linalg.norm(state_init[0:3]))**3)*state_init[0:3]
#     alt.append(np.linalg.norm(state_init[0:3])-R_EARTH)
#     vel.append(np.linalg.norm(state_init[3:6]))
#     t.append((i*0.01)/60.0)

r, v = coe2rv(7100.00, 0.00001, 98.0/RAD2DEG, 0.0, 0.0, 0.0)
date = 2461202.3122
state_init = np.array([7100.0, 0.0, 0.0, 0.0, 5.304, 5.304])
state_init[0:3] = r
state_init[3:6] = v
print(f"state init: {state_init}")
state_dot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
state_dot[0:3] = state_init[3:6]
state_dot[3:6] = get_acc_grav(state_init)
alt_rk4 = []
vel_rk4 = []
t_rk4 = []
sma = []
ecc = []
inc = []
raan = []
argp = []
tran = []

dict = {"Geopotential": 4, "drag": True, "Sun":True, "Moon":True, "srp":True}
sat_dict = {"mass": 1900.0, "area": 1.0, "c_d": 2.2, "c_r": 1.3}
params = {
    "satellite_properties": {
        "state": {
            "x": 7100.0,
            "y": 0.0,
            "z": 0.0,
            "x_dot": 0.0,
            "y_dot": 0.0,
            "z_dot": 7.5
        },
        "physical_properties": {
            "mass": 1900,
            "area": 1,
            "c_d": 2.2,
            "c_r": 1.3
        }
    },
    "integration": {
        "t_0": 0.0,
        "t_f": 1209600.0,
        "step": 1.0,
        "perturbations": {
            "Geopotential": 4,
            "drag": True,
            "Sun": True,
            "Moon": True,
            "srp": True
        }

    }
}


print( rv2coe(state_init[0:3], state_init[3:6]))
for i in range(0, (3600*24*10)):
    try:
        state_init = state_init + integrate_step_rk4(state=state_init, step=1.0, params=params, date=date)
    except Exception as e:
        print(e)
        break

    
    p, a, eccentricity, inclination, right_asc, argument_of_perigee, nu, _, _, _, _, _ = rv2coe(state_init[0:3], state_init[3:6])
    
    if(i%(60)==0):
        alt_rk4.append(np.linalg.norm(state_init[0:3])-R_EARTH)
        vel_rk4.append(np.linalg.norm(state_init[3:6]))
        sma.append(a)
        ecc.append(eccentricity)
        inc.append(inclination*RAD2DEG)
        raan.append(right_asc*RAD2DEG)
        argp.append(argument_of_perigee*RAD2DEG)
        tran.append(nu*RAD2DEG)
        t_rk4.append((i)/(3600*24))
    if(i%(3600*24)==0):    
        print(t_rk4[-1])
        print(sma[-1])
        print(ecc[-1])
        print(inc[-1])
print(vel_rk4[-1])
print(state_init)
# r_new, v_new = pkepler(r, v, 3600*24*30, 0.0, 0.0)
# print(r_new, v_new)
# print(rv2coe(r_new, v_new))
print( rv2coe(state_init[0:3], state_init[3:6]))
print(max(sma))
print(min(sma))
print(t_rk4[-1])
print(vel_rk4[-1])
print(sma[-1])
print(ecc[-1])
print(inc[-1])
print(raan[-1])
print(argp[-1])
print(tran[-1])
fig, ax1 = plt.subplots(6,figsize=(8.27, 11.69))

color = 'tab:red'
ax1[0].set_xlabel('Time [days]')
ax1[0].set_ylabel('Height [km]', color=color)
# ax1.plot(t, alt, color=color, label="Implicit Euler, 0.1 s timestep")
ax1[0].plot(t_rk4, alt_rk4, color=color, label="semi-major axis [km]", linestyle="solid")
ax1[0].tick_params(axis='y', labelcolor=color)
# ax1.legend()
ax1[0].grid()

color = 'tab:blue'
ax1[1].set_xlabel('time [days]')
ax1[1].set_ylabel('Eccentricity [-]', color=color)
# ax1.plot(t, alt, color=color, label="Implicit Euler, 0.1 s timestep")
ax1[1].plot(t_rk4, ecc, label="Right Ascension of Ascending Node")
ax1[1].tick_params(axis='y', labelcolor=color)
# ax1.legend()
ax1[1].grid()

color = 'tab:green'
ax1[2].set_xlabel('time [days]')
ax1[2].set_ylabel('Inclination [°]', color=color)
# ax1.plot(t, alt, color=color, label="Implicit Euler, 0.1 s timestep")
ax1[2].plot(t_rk4, inc, label="Right Ascension\nof Ascending Node")
ax1[2].tick_params(axis='y', labelcolor=color)
# ax1.legend()
ax1[2].grid()

ax1[3].set_xlabel('time [days]')
ax1[3].set_ylabel('Right-Ascension of\nAscending node [°]', color=color)
# ax1.plot(t, alt, color=color, label="Implicit Euler, 0.1 s timestep")
ax1[3].plot(t_rk4, raan, label="Right Ascension of Ascending Node")
ax1[3].tick_params(axis='y', labelcolor=color)
# ax1.legend()
ax1[3].grid()

ax1[4].set_xlabel('time [days]')
ax1[4].set_ylabel('Argument of\nPerigee [°]', color=color)
# ax1.plot(t, alt, color=color, label="Implicit Euler, 0.1 s timestep")
ax1[4].plot(t_rk4, argp, label="Right Ascension of Ascending Node")
ax1[4].tick_params(axis='y', labelcolor=color)
# ax1.legend()
ax1[4].grid()

ax1[5].set_xlabel('time [days]')
ax1[5].set_ylabel('Velocity [km/s]', color=color)
# ax1.plot(t, alt, color=color, label="Implicit Euler, 0.1 s timestep")
ax1[5].plot(t_rk4, vel_rk4, label="Right Ascension of Ascending Node")
ax1[5].tick_params(axis='y', labelcolor=color)
# ax1.legend()
ax1[5].grid()
# ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('[-]', color=color)  # we already handled the x-label with ax1
# # ax2.plot(t, vel, color=color, label="Implicit Euler, 0.1 s timestep")
# ax2.plot(t_rk4, ecc, color=color, label="Eccentricity", linestyle="-")
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.legend()
# ax2.grid()


# ax3 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

# color = 'tab:green'
# ax3.set_ylabel('[°]', color=color)  # we already handled the x-label with ax1
# # ax2.plot(t, vel, color=color, label="Implicit Euler, 0.1 s timestep")
# # ax3.plot(t_rk4, inc, label="Inclination")
# ax3.plot(t_rk4, raan, label="Right Ascension of Ascending Node", linestyle="-.")
# # ax3.plot(t_rk4, tran, label="True Anomaly")

# ax3.tick_params(axis='y', labelcolor=color)
# ax3.legend()
# ax3.grid()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("Envisat.png")
plt.show()