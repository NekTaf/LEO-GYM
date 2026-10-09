"""
Generate collision at TCA (time of closest approach). 
"""

# Standard library
import copy
from typing import Any, Dict, List, Optional, Tuple
import logging

# Third-party
import numpy as np

# Local
from leo_gym.satellite.satellite_base import SatelliteConfig, Satellite
from leo_gym.utils.utils import generate_random_perpendicular_normalized_vector
from leo_gym.orbit.dynamics.conversions import Delta_roe_to_rv

logger = logging.getLogger(__name__)

def collision_generator(
    rv0:np.ndarray,
    relative_t_tca:int,
    dt:int,
    params_dyn:Any,
    days:float,
    Dlon_range:tuple,
    Decc_range:tuple,
    beta_sampling_values:tuple, 
)->List:
    
    """
    Generates starting coordinates for debris collision 
    
    :param rv0: starting position and velocity of debris object [m, m, m, m/s, m/s, m/s]
    :type rv0: np.ndarray
    :param relative_t_tca: discrete relative time to collision from starting simulation ephemeris time [sec/dt]
    :type relative_t_tca: int
    :param dt: sampling time for simulation [sec]
    :type dt: int
    :param params_dyn: parameters for debris object using satellite_base class, all perturbations are included
    :type params_dyn: Any
    :param days: number of days for simulation (in this case anything above relative_t_tca is fine)
    :type days: float
    :param Dlon_range: ROE longitude deviation applied at conjuction point [m]
    :type Dlon_range: tuple
    :param Decc_range: ROE eccentricity deviation applied at conjuction point [m]
    :type Decc_range: tuple
    :param beta_sampling_values: a, b beta distribution values for sampling longitude and eccentricity displacment
    :type beta_sampling_values: tuple

    :return: (7,n) pvm of obstacle
    :rtype: List
    """
    
    u = np.random.beta(beta_sampling_values[0], beta_sampling_values[1]) 
    Dlon = Dlon_range[0] + u * (Dlon_range[1] - Dlon_range[0])
    
    u = np.random.beta(beta_sampling_values[0], beta_sampling_values[1]) 
    Decc = Decc_range[0] + u * (Decc_range[1] - Decc_range[0])  
    
    Droe = np.array([0,Dlon,Decc,Decc,0,0])
    logger.debug("ROE deviation applied at conjunction point: %s", Droe)
    
    rv0 = Delta_roe_to_rv(
        Droe=Droe,
        rv_ref=rv0
    )

    satellite_config = SatelliteConfig(
        rv0=rv0, 
        params_dyn=params_dyn, 
        dt=dt,
        days=days)

    object_secondary = Satellite(satellite_config)
    
    for _ in range(relative_t_tca):
        object_secondary.sat_propagate(np.zeros(3))   
        
    p_vector = object_secondary.rvm_eci_states[-1][:3]
        
    # generate random velocity vector but keep position same
    old_v = object_secondary.rvm_eci_states[-1][3:6]
    v_norm = np.linalg.norm(old_v)

    while True:
        v_vector = generate_random_perpendicular_normalized_vector(p_vector) * v_norm
        h = np.cross(p_vector, v_vector)
        
        # Reject orbits that are too similar to the original one
        # and reject nearly equatorial orbits for numerical stability
        if (abs(np.dot(v_vector, old_v)) < 0.95 * v_norm**2 and
            abs(h[2]) < 0.9998 * np.linalg.norm(h)):
            break
    object_secondary.rvm_eci_states[-1] = np.concatenate((p_vector,v_vector,object_secondary.rvm_eci_states[-1][6:7]),axis=0)
    
    # do back propagation to get starting relative position and velocity vectors for debris
    object_secondary.dt = -abs(object_secondary.dt)
    cached_debris_rvm = []
    cached_debris_rvm.append(object_secondary.rvm_eci_states[-1].copy())
    
    for _ in range(relative_t_tca):
        object_secondary.sat_propagate(np.zeros(3))  
        cached_debris_rvm.append(object_secondary.rvm_eci_states[-1].copy())

    assert np.isfinite(object_secondary.rvm_eci_states[-1]).all(), (
        f"debris rvm contains NaN:\n"
        f"{object_secondary.rvm_eci_states[-1]}"
    )
        
    return cached_debris_rvm[::-1]
    
def get_collision_metrics(
    delta_r_b: np.ndarray,
    cov_b: np.ndarray,
    l: float,
    eps: float = 1e-8
) -> Tuple[float,float,float,float,float]:
    """
    Compute collision probability metrics in the B-plane.

    :param delta_r_b: (3,) [xi,eta,zeta] -b-plane meters
    :type delta_r_b: np.ndarray
    :param cov_b: (3,3) Their covariance matrix
    :type cov_b: np.ndarray
    :param l: The collision sphere radius in meters
    :type l: float
    :param eps: Small value to avoid division by zero
    :type eps: float

    :return: Tuple of

        - Pc: Collision probability
        - Pc_max: Max collision probability
        - mahalanobis_dist: Mahalanobis distance
        - xi, zeta: 2D projected position on B-plane
    :rtype: Tuple[float, float, float, float, float]
    """
    # Project to (xi, zeta)
    delta_3d = np.asarray(delta_r_b).reshape(3,)
    delta_2d = np.array([delta_3d[0], delta_3d[2]]).reshape(2, 1)

    cov_2d = cov_b[np.ix_([0, 2], [0, 2])]
    inv_cov_2d = np.linalg.inv(cov_2d)

    d2 = (delta_2d.T @ inv_cov_2d @ delta_2d).item()
    d = np.sqrt(max(d2, 0.0))
    d_safe = max(d, eps)

    det_cov = float(np.linalg.det(cov_2d))
    det_sqrt = np.sqrt(max(det_cov, eps))

    Pc = (l**2 / (2 * det_sqrt)) * np.exp(-0.5 * d_safe**2)
    Pc_max = (l**2 / (d_safe**2 * det_sqrt * np.e))

    Pc = float(np.clip(Pc, 0.0, 1.0))
    Pc_max = float(np.clip(Pc_max, 0.0, 1.0))

    return Pc, Pc_max, d, delta_2d[0, 0], delta_2d[1, 0]
