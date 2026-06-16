import numpy as np
def apply_dv(state: np.ndarray, manv: np.ndarray) -> np.ndarray:
    """Applay a delta v in NTW frame directly to a state vector

    Args:
        state (np.ndarray): State Vector
        manv (np.ndarray): Manoeuvre delta v in NTW frame

    Returns:
        np.ndarray: Delta v in ECI frame
    """
    r = state[0:3]
    v = state[3:6]

    T = v/np.linalg.norm(v)
    W = np.cross(r, v)/np.linalg.norm(np.cross(r, v))
    N = np.cross(T,W)
    rot_eci2ntw = np.vstack([N,T,W])
    
    return np.dot(np.linalg.inv(rot_eci2ntw), manv)