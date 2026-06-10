import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

MU = 398600.4415 # in km³/s²
R_EARTH = 6378.1363

def get_all_acc(state: ArrayLike, perturbations: dict) -> np.ndarray:
    """_summary_

    Args:
        state (ArrayLike): _description_
        perturbations (dict): _description_

    Returns:
        np.ndarray: _description_
    """
    acc_grav = get_acc_grav(state=state)
    return acc_grav

def get_acc_grav(state: ArrayLike) -> np.ndarray:
    """Estimates the Accelaration due to Gravity at a given moment

    Args:
        state (ArrayLike): State Vector at given time

    Returns:
        np.ndarray: Accelaration vector
    """
    acc_grav = -(MU/(np.linalg.norm(state[0:3]))**3)*state[0:3]

    return acc_grav

def integrate_step_rk(state: ArrayLike, step: float, perturbations: dict) -> np.ndarray:
    
    k1_acc = get_all_acc(state=state, perturbations=perturbations)
    k1_vel = state[3:6]
    step_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    step_1[0:3] = k1_vel
    step_1[3:6] = k1_acc
    state_1 = state + step_1*(step/2)

    k2_acc = get_all_acc(state=state_1, perturbations=perturbations)
    k2_vel = state_1[3:6]
    step_2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    step_2[0:3] = k2_vel
    step_2[3:6] = k2_acc
    state_2 = state + step_2*(step/2)

    k3_acc = get_all_acc(state=state_2, perturbations=perturbations)
    k3_vel = state_2[3:6]
    step_3 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    step_3[0:3] = k3_vel
    step_3[3:6] = k3_acc
    state_3 = state + step_3*(step)

    k4_acc = get_all_acc(state=state_3, perturbations=perturbations)
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


state_init = np.array([7100.0, 0.0, 0.0, 0.0, 7.5, 0.0])
state_dot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
state_dot[0:3] = state_init[3:6]
state_dot[3:6] = get_acc_grav(state_init)
alt_rk4 = []
vel_rk4 = []
t_rk4 = []


dict = {"none": 0}
for i in range(0, 1209600):
    state_init = state_init + integrate_step_rk(state=state_init, step=1.0, perturbations=dict)

    alt_rk4.append(np.linalg.norm(state_init[0:3])-R_EARTH)
    vel_rk4.append(np.linalg.norm(state_init[3:6]))
    t_rk4.append((i)/(24*3600.0))

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time [days]')
ax1.set_ylabel('Altitude over earth [km]', color=color)
# ax1.plot(t, alt, color=color, label="Implicit Euler, 0.1 s timestep")
ax1.plot(t_rk4, alt_rk4, color="green", label="RK4, 1 s timestep")
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend()
ax1.grid()

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Velocity [km/s]', color=color)  # we already handled the x-label with ax1
# ax2.plot(t, vel, color=color, label="Implicit Euler, 0.1 s timestep")
ax2.plot(t_rk4, vel_rk4, color="yellow", label="RK4, 1 s timestep")
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend()
ax2.grid()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("simple_prop_rk4.png")