import pandas as pd
from valladopy.mathtime.julian_date import invjday, jday
from valladopy.astro.twobody.frame_conversions import rv2coe, coe2rv
from numpy.typing import ArrayLike
from datetime import datetime
import json
import numpy as np
from src.constants import RAD2DEG
import matplotlib.pyplot as plt
from src.constants import R_EARTH


def initialize_dataframes(params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sets up the dataframe for saving intermediate outputs

    Args:
        params (dict): Dictionary of settings

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:  Dataframe of Keplerian states,
                                            Dataframe of Cartesian states
    """
    df_cart, df_kep = None, None
    if params["outputs"]["Keplerian"]:
        df_kep = pd.DataFrame(columns=['epoch JD','epoch', 'Semi-Major axis [km]','Eccentricity [-]','Inclination [°]','Rigth Ascension of ascending node [°]','Argumnet of Perigee [°]', 'True Anomaly [°]'])

    if params["outputs"]["Cartesian"]:
        df_cart = pd.DataFrame(columns=['epoch JD', 'epoch', 'X [km]','Y [km]','Z [km]','X_DOT [km/s]','Y_DOT [km/s]', 'Z_DOT [km/s]'])

    return df_kep, df_cart

def append_dataframe(params: dict, state:np.ndarray, date: datetime, df_kep: pd.DataFrame, df_cart: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append the dataframe of orbital states by one epoch

    Args:
        params (dict): Dictionary of settings
        state (np.ndarray): Cartesian state in ECI frame
        date (datetime): Epoch to be appended
        df_kep (pd.DataFrame): Dataframe of Keplerian states
        df_cart (pd.DataFrame): Dataframe of Cartesian states

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:  Dataframe of Keplerian states,
                                            Dataframe of Cartesian states
    """
    if params["outputs"]["Keplerian"]:
        julian_day, julian_fraction = jday(date.year, date.month, date.day, date.hour, date.minute, date.second)
        julian_date = julian_day + julian_fraction 
        _, a, eccentricity, inclination, right_asc, argument_of_perigee, nu, m, _, _, _, _ = rv2coe(state[0:3], state[3:6])
        data = pd.DataFrame({"epoch JD": [julian_date],
            "epoch": [date],
            "Semi-Major axis [km]": [a],
            "Eccentricity [-]": [eccentricity],
            "Inclination [°]": [inclination*RAD2DEG],
            "Rigth Ascension of ascending node [°]": [right_asc*RAD2DEG],
            "Argumnet of Perigee [°]": [argument_of_perigee*RAD2DEG],
            "True Anomaly [°]": [nu*RAD2DEG]})
        df_kep = pd.concat([df_kep, data])

    if params["outputs"]["Cartesian"]:
        julian_day, julian_fraction = jday(date.year, date.month, date.day, date.hour, date.minute, date.second)
        julian_date = julian_day + julian_fraction 
        data = pd.DataFrame({"epoch JD": [julian_date],
            "epoch": [date],
            "X [km]": [state[0]],
            "Y [km]": [state[1]],
            "Z [km]": [state[2]],
            "X_DOT [km/s]": [state[3]],
            "Y_DOT [km/s]": [state[4]],
            "Z_DOT [km/s]": [state[5]]})
        df_cart = pd.concat([df_cart, data])

    return df_kep, df_cart

def write_outputs(output, df_kep, df_cart):
    df_cart.to_csv(path_or_buf=output, index=False)
    df_kep.to_csv(path_or_buf=output, index=False)

def process_input(input: str) -> tuple[dict, np.ndarray, datetime, datetime]:
    """Parse the input Json in a given location

    Args:
        input (str): Locaction of Input json
        
    Returns:
        tuple[  params (dict): Dictionary of parameters,
                state (np.ndarray): Initial State vector, 
                date_0 (datetime): Initial date, 
                date_f (datetime): Propagation end date]
    """
    
    with open(input) as f:
        params = json.load(f)
    if params["satellite_properties"]["input_mode"] == 0:
        state = np.array([params["satellite_properties"]["state"]["x"],
                          params["satellite_properties"]["state"]["y"],
                          params["satellite_properties"]["state"]["z"],
                          params["satellite_properties"]["state"]["x_dot"],
                          params["satellite_properties"]["state"]["y_dot"],
                          params["satellite_properties"]["state"]["z_dot"]])
    else:
        p = params["satellite_properties"]["kep_state"]["sma"]*(1.0- params["satellite_properties"]["kep_state"]["ecc"]**2)
        r, v = coe2rv(p,
                        params["satellite_properties"]["kep_state"]["ecc"],
                        params["satellite_properties"]["kep_state"]["inc"]/RAD2DEG,
                        params["satellite_properties"]["kep_state"]["raan"]/RAD2DEG,
                        params["satellite_properties"]["kep_state"]["argp"]/RAD2DEG,
                        params["satellite_properties"]["kep_state"]["nu"]/RAD2DEG)
        state = np.zeros(6)
        state[0:3] = r
        state[3:6] = v

    date_0 = datetime(  params["integration"]["start_date"]["year"],
                    params["integration"]["start_date"]["month"],
                    params["integration"]["start_date"]["day"],
                    params["integration"]["start_date"]["hour"],
                    params["integration"]["start_date"]["minute"],
                    int(params["integration"]["start_date"]["second"])
                    )
    date_f = datetime(  params["integration"]["end_date"]["year"],
                    params["integration"]["end_date"]["month"],
                    params["integration"]["end_date"]["day"],
                    params["integration"]["end_date"]["hour"],
                    params["integration"]["end_date"]["minute"],
                    int(params["integration"]["end_date"]["second"]))
   
    return params, state, date_0, date_f

def write_outputs(kep_df, cart_df, output):
    cart_df.to_csv(path_or_buf=output+"/cartesian.csv", index=False)
    kep_df.to_csv(path_or_buf=output+"/keplerian.csv", index=False)   

def plot_altitude(cart_df, output):
    alt = list()
    vel = list()
    for index, row in cart_df.iterrows():
        alt.append(np.linalg.norm(np.array([row["X [km]"], row["Y [km]"], row["Z [km]"]])) - R_EARTH)
        vel.append(np.linalg.norm(np.array([row["X_DOT [km/s]"], row["Y_DOT [km/s]"], row["Z_DOT [km/s]"]])))
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
    colors = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    ]

    linestyles = [
        "-",                 # solid
        "--",                # dashed
        "-.",                # dash-dot
        ":",                 # dotted
        (0, (5, 1)),         # long dash
        (0, (3, 1, 1, 1)),  # dash-dot-dot
    ]
    elements = [
        ("alt", "Altitude [km]"),
        ("vel", "Velocity [km/s]")
    ]       
    axes[0].plot(cart_df["epoch"], alt, linewidth=1.5, color=colors[0], linestyle=linestyles[0])
    axes[0].set_ylabel("Altitude [km]")
    axes[0].grid(True, alpha=0.3)   

    axes[1].plot(cart_df["epoch"], vel, linewidth=1.5, color=colors[1], linestyle=linestyles[1])
    axes[1].set_ylabel("Velocity [km/s]")
    axes[1].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Epoch")

    fig.suptitle("Altitude and Velocity vs Epoch")
    fig.tight_layout()
    plt.show()
    plt.savefig(output + "/plots/AltVel.png", dpi=1600)
    plt.show()

def plot_results(kep_df, output):
    fig, axes = plt.subplots(6, 1, figsize=(8.27, 11.69), sharex=True)
    colors = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    ]

    linestyles = [
        "-",                 # solid
        "--",                # dashed
        "-.",                # dash-dot
        ":",                 # dotted
        (0, (5, 1)),         # long dash
        (0, (3, 1, 1, 1)),  # dash-dot-dot
    ]
    elements = [
        ("Semi-Major axis [km]", "a [km]"),
        ("Eccentricity [-]", "e [-]"),
        ("Inclination [°]", "i [°]"),
        ("Rigth Ascension of ascending node [°]", "RAAN [°]"),
        ("Argumnet of Perigee [°]", "ω [°]"),
        ("True Anomaly [°]", "ν [°]")
    ]       
    i = 0
    for ax, (col, ylabel) in zip(axes, elements):
        ax.plot(kep_df["epoch"], kep_df[col], linewidth=1.5, color=colors[i], linestyle=linestyles[i])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        i += 1

    axes[-1].set_xlabel("Epoch")

    fig.suptitle("Orbital Elements vs Epoch")
    fig.tight_layout()
    plt.show()
    plt.savefig(output + "/plots/Keplerian.png", dpi=1600)
    plt.show()
