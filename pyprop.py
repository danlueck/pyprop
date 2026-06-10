import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from valladopy.astro.twobody.frame_conversions import rv2coe, coe2rv
from valladopy.astro.celestial.moon import position as moon_pos
from valladopy.astro.celestial.sun import position as sun_pos
import math

MU = 398600.4415 # in km³/s² from Fundamentals of Astrodynamics and Applications - Davie A. Vallado
MU_SUN = 1.32712440042e11
MU_MOON = 4902.799
R_EARTH = 6378.1363 # From FUNDAMENTALS OF ASTRODYNAMICS - Karel F. Wakker, 2015
J2 = 0.0010826357 # From FUNDAMENTALS OF ASTRODYNAMICS - Karel F. Wakker, 2015
RAD2DEG = 180.0/math.pi
SEC_IN_DAY = 3600*24

def get_all_acc(state: ArrayLike, perturbations: dict, satellite: dict, date: float) -> np.ndarray:
    """_summary_

    Args:
        state (ArrayLike): _description_
        perturbations (dict): _description_

    Returns:
        np.ndarray: _description_
    """
    all_acc = get_acc_grav(state=state)
    
    if(perturbations["J2"]):
        all_acc += get_acc_J2(state=state)
    if(perturbations["drag"]):
        all_acc += get_acc_drag(state=state, sat=satellite)
    if(perturbations["Moon"] or perturbations["Sun"]):
       all_acc += get_acc_3rd_body(state=state, date=date, bodies=perturbations)

    return all_acc

def get_acc_grav(state: ArrayLike) -> np.ndarray:
    """Estimates the Accelaration due to Gravity at a given moment

    Args:
        state (ArrayLike): State Vector at given time

    Returns:
        np.ndarray: Accelaration vector
    """
    acc_grav = -(MU/(np.linalg.norm(state[0:3]))**3)*state[0:3]

    return acc_grav

def get_acc_drag(state: ArrayLike, sat: dict) -> np.ndarray:
    """Estimates the Accelaration due to Gravity at a given moment

    Args:
        state (ArrayLike): State Vector at given time

    Returns:
        np.ndarray: Accelaration vector
    """
    altitude = np.linalg.norm(state[0:3]) - R_EARTH
    density = get_density(altitude=altitude)
    B_inv = (sat["area"]*sat["c_d"]*1e-6)/sat["mass"]
    acc_drag = -0.5 * B_inv * density * np.linalg.norm(state[3:6])**2*(state[3:6]/(np.linalg.norm(state[3:6])))
    return acc_drag



def get_acc_J2(state: ArrayLike) -> np.ndarray:
    """Estimates the Accelaration due to Gravity at a given moment

    Args:
        state (ArrayLike): State Vector at given time

    Returns:
        np.ndarray: Accelaration vector
    """
    acc_J2 = np.array([0.0, 0.0, 0.0])
    pre_term = -(3.0/2.0) * MU*J2 * ((R_EARTH**2)/(np.linalg.norm(state[0:3])**5))
    acc_J2[0] = pre_term * state[0] * (1.0 - 5.0*((state[2]**2)/(np.linalg.norm(state[0:3])**2)))
    acc_J2[1] = pre_term * state[1] * (1.0 - 5.0*((state[2]**2)/(np.linalg.norm(state[0:3])**2)))
    acc_J2[2] = pre_term * state[2] * (3.0 - 5.0*((state[2]**2)/(np.linalg.norm(state[0:3])**2)))
    return acc_J2

def get_acc_3rd_body(state: ArrayLike, date: float, bodies: dict):
    acc_3rd = np.zeros(3)
    if(bodies["Sun"]):
        acc_3rd += get_acc_n_body(state=state, date=date, pos_fun=sun_pos, mu_body=MU_SUN)
    if(bodies["Moon"]):
        acc_3rd += get_acc_n_body(state=state, date=date, pos_fun=moon_pos, mu_body=MU_MOON)
    return acc_3rd

def get_acc_n_body(state: ArrayLike, date: float, pos_fun, mu_body: float) -> np.ndarray:
    """Estimates the Accelaration due to Gravity at a given moment

    Args:
        state (ArrayLike): State Vector at given time

    Returns:
        np.ndarray: Accelaration vector
    """
    pos_earth_3rd, _, _ = pos_fun(date)
    pos_sat_3rd = pos_earth_3rd - state[0:3]

    return mu_body*(pos_sat_3rd*(1.0/np.linalg.norm(pos_sat_3rd)**3)-pos_earth_3rd*(1.0/np.linalg.norm(pos_earth_3rd)**3))

def get_density(altitude:float) -> float:
    expAltitudes = np.array([25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,100.0, 
                             110.0,120.0,130.0,140.0,150.0,180.0,200.0,250.0,300.0, 
                             350.0,400.0,450.0,500.0,600.0,700.0,800.0,900.0,1000.0])
    exph0 = np.array([0.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,100.0, 
                     110.0,120.0,130.0,140.0,150.0,180.0,200.0,250.0,300.0, 
                     350.0,400.0,450.0,500.0,600.0,700.0,800.0,900.0])
    expDensity = np.array([1.225,3.899e-2,1.774e-2,3.972e-3,1.057e-3,3.206e-4,8.770e-5, 
                     1.905e-5, 3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, 
                     2.070e-9, 5.464e-10,2.789e-10,7.248e-11,2.418e-11,9.518e-12,3.725e-12,
                     1.585e-12,6.967e-13,1.454e-13,3.614e-14,1.170e-14,5.245e-15])
    expHeight = np.array([7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 
                        5.877, 7.263, 9.473,12.636,16.149,22.523,29.740,37.105,45.546,
                        53.628,53.298,58.515,60.828,63.822,71.835,88.667,124.64,181.05])
    if(altitude<0.0):
        raise Exception("Object Impacted EARTH")
    if(altitude>1000.0): return 0.0

    for i in range(0, len(expAltitudes)):
        if(altitude<expAltitudes[i]):
            idx = i
            return 1e9* expDensity[idx]*math.exp(-(altitude - exph0[idx])/expHeight[idx])
        
    return 0.0
   

def integrate_step_rk(state: ArrayLike, step: float, perturbations: dict, satellite: dict, date:float) -> np.ndarray:

    k1_acc = get_all_acc(state=state, perturbations=perturbations, satellite=satellite, date=date)
    k1_vel = state[3:6]
    step_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    step_1[0:3] = k1_vel
    step_1[3:6] = k1_acc
    state_1 = state + step_1*(step/2)

    k2_acc = get_all_acc(state=state_1, perturbations=perturbations, satellite=satellite, date=(date+0.5*step/SEC_IN_DAY))
    k2_vel = state_1[3:6]
    step_2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    step_2[0:3] = k2_vel
    step_2[3:6] = k2_acc
    state_2 = state + step_2*(step/2)

    k3_acc = get_all_acc(state=state_2, perturbations=perturbations, satellite=satellite, date=(date+0.5*step/SEC_IN_DAY))
    k3_vel = state_2[3:6]
    step_3 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    step_3[0:3] = k3_vel
    step_3[3:6] = k3_acc
    state_3 = state + step_3*(step)

    k4_acc = get_all_acc(state=state_3, perturbations=perturbations, satellite=satellite, date=(date+step/SEC_IN_DAY))
    k4_vel = state_3[3:6]
    step_4 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    step_4[0:3] = k4_vel
    step_4[3:6] = k4_acc

    return (1.0/6.0)*(step_1 + 2.0*step_2 + 2.0*step_3 + step_4)




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

r, v = coe2rv(42000.5, 0.0001, 0.01/RAD2DEG, 0.0, 0.0)
date = 2461202.3122
state_init = np.array([7100.0, 0.0, 0.0, 0.0, 5.304, 5.304])
state_init[0:3] = r
state_init[3:6] = v
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

dict = {"J2": True, "drag": True, "Sun":True, "Moon":True}
sat_dict = {"mass": 100, "area": 1, "c_d": 2.2}
print( rv2coe(state_init[0:3], state_init[3:6]))
for i in range(0, (3600*24*30)):
    try:
        state_init = state_init + integrate_step_rk(state=state_init, step=10.0, perturbations=dict, satellite=sat_dict, date=date)*10.0
    except Exception as e:
        print(e)
        break

    alt_rk4.append(np.linalg.norm(state_init[0:3])-R_EARTH)
    vel_rk4.append(np.linalg.norm(state_init[3:6]))
    p, a, eccentricity, inclination, right_asc, argument_of_perigee, nu, _, _, _, _, _ = rv2coe(state_init[0:3], state_init[3:6])
    sma.append(a)
    ecc.append(eccentricity)
    inc.append(inclination*RAD2DEG)
    raan.append(right_asc*RAD2DEG)
    argp.append(argument_of_perigee*RAD2DEG)
    tran.append(nu*RAD2DEG)
    t_rk4.append((i)/(24*360.0))
    if(i%(3600*24)==0):
        print(t_rk4[-1])
        print(sma[-1])
        print(inc[-1])



print( rv2coe(state_init[0:3], state_init[3:6]))
print(max(sma))
print(min(sma))
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time [days]')
ax1.set_ylabel('Height [km]', color=color)
# ax1.plot(t, alt, color=color, label="Implicit Euler, 0.1 s timestep")
ax1.plot(t_rk4, alt_rk4, color=color, label="semi-major axis [km]")
ax1.tick_params(axis='y', labelcolor=color)
# ax1.legend()
ax1.grid()

# ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('[-]', color=color)  # we already handled the x-label with ax1
# # ax2.plot(t, vel, color=color, label="Implicit Euler, 0.1 s timestep")
# ax2.plot(t_rk4, ecc, color=color, label="Eccentricity")
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.legend()
# ax2.grid()


ax3 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax3.set_ylabel('[°]', color=color)  # we already handled the x-label with ax1
# ax2.plot(t, vel, color=color, label="Implicit Euler, 0.1 s timestep")
# ax3.plot(t_rk4, inc, label="Inclination")
ax3.plot(t_rk4, inc, label="Inclination")
# ax3.plot(t_rk4, tran, label="True Anomaly")

ax3.tick_params(axis='y', labelcolor=color)
ax3.legend()
ax3.grid()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("GEO.png")
plt.show()