import inspect
import numpy as np
from functools import reduce
from typing import Callable, Any, Dict, List, Tuple

def compose(*functions:Callable) -> Callable:
    """Left to right function composition."""
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


def call_with_filtered(func:Callable, arg_dict:Dict[str, Any]) -> Any:
    """Calls a function with a dictionary of arguments, filtering out any that are not valid for the function."""
    sig = inspect.signature(func)
    valid_args = set(sig.parameters)
    filtered_dict = {k: v for k, v in arg_dict.items() if k in valid_args}
    return func(**filtered_dict)

def inv_sql(x:np.ndarray, magnitude:float = 1.0, scale:float = 1.0, tol:float = 1e-3) -> np.ndarray:
    """Simple inverse square law function."""
    d = np.where(x/scale < tol, tol, x/scale)
    return magnitude/d**2

def exponential(x:np.ndarray, magnitude:float = 1.0, scale:float = 1.0) -> np.ndarray:
    """Simple exponential decay function."""
    return magnitude*np.exp(-x/scale)


# Movement functions from pathing.py
from itertools import repeat

def random_walk(start:np.ndarray,
                mu: np.ndarray, 
                sigma: np.ndarray, 
                dt:float, 
                steps:int,
                trials:int) -> np.ndarray:
    """Generate random walk paths based on multivariate normal distributions."""
    path = np.zeros((len(mu),trials, steps+1, len(mu[0])))
    path[:,:,0] = start[:,None,:]
    path[:,:,1:] = dt*np.array([*map(np.random.multivariate_normal, mu, sigma, repeat((trials, steps)))])
    return path.cumsum(axis=2)
    

def elliptical(center:np.ndarray,
            periods: np.ndarray,
            a:np.ndarray,
            b:np.ndarray,
            phi:np.ndarray,
            steps:int) -> np.ndarray:
    """Generate elliptical paths with specified parameters."""
    t = np.linspace(0, 2*np.pi*periods, steps+1)
    X = lambda t: a*np.cos(t)*np.cos(phi) - b*np.sin(t)*np.sin(phi)
    Y = lambda t: a*np.cos(t)*np.sin(phi) + b*np.sin(t)*np.cos(phi)
    return np.array([X(t),Y(t)]).T + center[:,None,:]


def linear(start:np.ndarray,
           velocity:np.ndarray,
           angle:float,
           dt:float,
           steps:int) -> np.ndarray:
     """Generate linear paths with constant velocity and direction."""
     v = (velocity*np.array([np.cos(angle), np.sin(angle)])*dt).T
     return v[:,None,:]*np.arange(steps+1)[None,:,None] + start[:,None,:]

def stationary(start:np.ndarray,
                 steps:int) -> np.ndarray:
    """Generate stationary paths."""
    return np.repeat(start[:,None,:], steps+1, axis=1)



## computation
def compute_distance(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Computes the distance between x and z"""
    return np.linalg.norm(x - z, axis=-1)

def compute_all_distances(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Computes the distance for all k x l sensor/source combinations."""
    x = x[None,...] if x.ndim == 2 else x
    z = z[None,...] if z.ndim == 2 else z
    return np.array([compute_distance(x, z) for z in z])

def compute_gradient(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Computes the gradient"""
    return np.abs(np.gradient(r, t, axis=1))

def approximate_line_integral(f: Callable[[np.ndarray], np.ndarray], 
                              r: np.ndarray,
                              t: np.ndarray) -> np.ndarray:
    """Approximates the line integral using Simpson's rule for multiple paths."""
    return np.trapezoid(f(r) * compute_gradient(r, t), t, axis=1)
