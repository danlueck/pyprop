
import numpy as np

from src.gravity import get_acc_grav
from src.drag import get_acc_drag
from src.thridbody import get_acc_3rd_body
from src.srp import get_acc_srp
from datetime import datetime, timedelta

def deriv(state: np.ndarray, params: dict, date: datetime) -> np.ndarray:
    """Find the derivative of the state at a given epoch

    Args:
        state (ArrayLike): Cartesian state vector in ECI frame
        params (dict): Dictionary of input paramets
        date (datetime): Date for which to evaluate derivative
    Returns:
        np.ndarray: Derivative of state at date
    """
    satellite = params["satellite_properties"]["physical_properties"]
    perturbations = params["integration"]["perturbations"]

    acc = get_all_acc(state=state, perturbations=perturbations, satellite=satellite, date=date)
    vel = state[3:6]
    deriv = np.zeros(6)
    deriv[0:3] = vel
    deriv[3:6] = acc

    return deriv

def get_all_acc(state: np.ndarray, perturbations: dict, satellite: dict, date: datetime) -> np.ndarray:
    """Estimate all Accelerations on 

    Args:
        state (np.ndarray): Cartesian state vector in ECI frame
        perturbations (dict): Dictionary of considered perturbations
        satellite (dict): Dictionary of satellite properties
        date (datetime): Date to evaluate Acceleration
    Returns:
        np.ndarray: Vector of Accelerations
    """

    all_acc = get_acc_grav(state=state, geopotenital_order=perturbations["Geopotential"])
    if(perturbations["drag"]):
        all_acc += get_acc_drag(state=state, sat=satellite)
    if(perturbations["Moon"] or perturbations["Sun"]):
        all_acc += get_acc_3rd_body(state=state, date=date, bodies=perturbations)
    if(perturbations["srp"]):
        all_acc += get_acc_srp(state=state, date=date, satellite=satellite)

    return all_acc

def integrate_step_explicit_euler(state: np.ndarray, step: float, params: dict, date:datetime, fun_deriv=deriv) -> np.ndarray:
    """Integrate one step using 4th Order Runge Kutta

    Args:
        state (np.ndarray): Cartesian state vector in ECI frame
        step (float): Step size in seconds
        params (dict): Dictionary of input paramets
        date (datetime): Start date of current propagation step
        fun_deriv (function, optional): Funtion to find derivative of state. Defaults to deriv.

    Returns:
        np.ndarray: Integrated change for one step
    """
    return fun_deriv(state=state, params=params, date=date)*step

def integrate_step_rk4(state: np.ndarray, step: float, params: dict, date:datetime, fun_deriv=deriv) -> np.ndarray:
    """Integrate one step using 4th Order Runge Kutta

    Args:
        state (np.ndarray): Cartesian state vector in ECI frame
        step (float): Step size in seconds
        params (dict): Dictionary of input paramets
        date (datetime): Start date of current propagation step
        fun_deriv (function, optional): Funtion to find derivative of state. Defaults to deriv.

    Returns:
        np.ndarray: Integrated change for one step
    """

    k1 = fun_deriv(state=state, params=params, date=date)
    state_1 = state + k1*(step/2)

    k2 = fun_deriv(state=state_1, params=params, date=(date+timedelta(0,0.5*step)))
    state_2 = state + k2*(step/2)

    k3 = fun_deriv(state=state_2, params=params, date=(date+timedelta(0,0.5*step)))
    state_3 = state + k3*(step)

    k4 = fun_deriv(state=state_3, params=params, date=(date+timedelta(0,step)))

    return (1.0/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)*step



def integrate_step_rkf(state: np.ndarray, step: float, params: dict, date:datetime, fun_deriv=deriv) -> tuple[np.ndarray, float]:
    """Integrate one step using an RKF45 variable step size integrator.

    Args:
        state (np.ndarray): Cartesian state vector in ECI frame
        step (float): Step size in seconds
        params (dict): Dictionary of input paramets
        date (datetime): Start date of current propagation step
        fun_deriv (function, optional): Funtion to find derivative of state. Defaults to deriv.

    Returns:
        tuple[np.ndarray, float]: Integrated state change for one step and the new step size
    """
    te = 1000
    new_step = step
    while(te>1e-9):
        k1 = fun_deriv(state=state, params=params, date=date)*new_step
        
        state_1 = state + k1*0.25

        k2 = fun_deriv(state=state_1, params=params, date=(date+timedelta(0,0.25*new_step)))*new_step
        state_2 = state + (3/32)*k1 + (9/32)*k2
    
        k3 = fun_deriv(state=state_2, params=params, date=(date+timedelta(0,(3/8)*new_step)))*new_step
        state_3 = state + (1932/2197)*k1 - (7200/2197)*k2 +(7296/2197)*k3

        k4 = fun_deriv(state=state_3, params=params, date=(date+timedelta(0,(12/13)*new_step)))*new_step
        state_4 = state + (439/216)*k1 - 8.0*k2 +(3680/513)*k3 - (845/4104)*k4

        k5 = fun_deriv(state=state_4, params=params, date=(date+timedelta(0,new_step)))*new_step
        state_5 = state - (8/27)*k1 + 2.0*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5

        k6 = fun_deriv(state=state_5, params=params, date=(date+timedelta(0,(1/2)*new_step)))*new_step
        te = sum(abs((1/360) * k1 - (128/4275)*k3 - (2197/75240)*k4 + (1/50)*k5 + (2/55)*k6))

        new_step = 0.9 * new_step * (1e-9/te)**(1/5)
        if (new_step < 0.01):
            new_step = 0.01
            te = 0.0
            
    k1 = fun_deriv(state=state, params=params, date=date)*new_step
    
    state_1 = state + k1*0.25

    k2 = fun_deriv(state=state_1, params=params, date=(date+timedelta(0,0.25*new_step)))*new_step
    state_2 = state + (3/32)*k1 + (9/32)*k2

    k3 = fun_deriv(state=state_2, params=params, date=(date+timedelta(0,(3/8)*new_step)))*new_step
    state_3 = state + (1932/2197)*k1 - (7200/2197)*k2 +(7296/2197)*k3

    k4 = fun_deriv(state=state_3, params=params, date=(date+timedelta(0,(12/13)*new_step)))*new_step
    state_4 = state + (439/216)*k1 - 8.0*k2 +(3680/513)*k3 - (845/4104)*k4

    k5 = fun_deriv(state=state_4, params=params, date=(date+timedelta(0,new_step)))*new_step
    state_5 = state - (8/27)*k1 + 2.0*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5

    k6 = fun_deriv(state=state_5, params=params, date=(date+timedelta(0,(1/2)*new_step)))*new_step
    
    state_change = ((25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - (1/5)*k5)
    
    return state_change, new_step