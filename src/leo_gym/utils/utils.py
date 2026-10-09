# Standard library
import csv
import gzip
import os
import random
import shutil
import socket
from datetime import datetime
import inspect
from pathlib import Path
from typing import Any

# Third-party
import numpy as np
import torch
import yaml


def create_dir(
    parent_name: str
)->str:
    """
    Creates a directory named with the parent name, computer's hostname, and the current datetime.
    
    :param parent_name: The parent name part of the directory to be created.
    :type parent_name: str
    
    :return: The name of the created directory.
    :rtype: str
    """
    computer_name = socket.gethostname()
    directory_save = f"{parent_name}_{computer_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if not os.path.exists(directory_save):
        os.makedirs(directory_save)
        
    return directory_save


def hypers2txt(
    hyper_dict:dict,
    file_path: str
)->None:
    """
    Writes the items of a dictionary to a text file, each key-value pair on a new line.
    
    :param hyper_dict: The dictionary to write to the file.
    :type hyper_dict: dict
    :param file_path: The path to the file where the dictionary should be written.
    :type file_path: str
    """
    with open(file_path, "w") as file:
        for key, value in hyper_dict.items():
            file.write(f"{key}: {value}\n")
            
    return
            
            
def save_params_to_file(
    params: object,
    file_name: str
)->None:
    """
    Saves parameters from an object's attributes to a file.
    
    :param params: The object containing parameters as attributes.
    :type params: object
    :param file_name: The name of the file to save the parameters to.
    :type file_name: str
    """
    with open(file_name, 'w') as file:
        for key, value in params.__dict__.items():
            file.write(f'{key} = {value}\n')
            
    return

def yaml2dict(
    filename:str
)->dict:
    """
    Loads a YAML file and returns its contents as a dictionary.
    
    :param filename: The path to the YAML file.
    :type filename: str
    
    :return: The YAML file contents as a dictionary.
    :rtype: dict
    """
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
        
    return data



def save_yaml_copy(
    original_file:str,
    copy_file:str
)->None:
    """
    Copies the contents of a YAML file to another file.
    
    :param original_file: The path to the original YAML file.
    :type original_file: str
    :param copy_file: The path to the file where the copy should be saved.
    :type copy_file: str
    """
    with open(original_file, 'r') as file:
        content = file.read()
    with open(copy_file, 'a') as file:
        file.write(content)

    return


def write2txt(
    text_file:str, 
    data:str
)->None:
    """
    Appends data to a text file.
    
    :param text_file: The path to the text file.
    :type text_file: str
    :param data: The data to append to the file.
    :type data: str
    """
    with open(text_file, 'a') as file:
        file.write(data)

    return


def dict2txt(
    text_file:str, 
    dict: str
)->None:
    """
    Appends the contents of a dictionary to a text file, formatting each key-value pair.
    
    :param text_file: The path to the text file.
    :type text_file: str
    :param dict: The dictionary whose contents are to be written to the file.
    :type dict: dict
    """
    with open(text_file, 'a') as file:
        for var_name, var_value in dict.items():
            file.write(f"{var_name}:\n{var_value}\n\n")
            
    return


def clear_csv(
    name: str
)->None:
    """
    Clears the contents of a CSV file.
    
    :param name: The path to the CSV file.
    :type name: str
    """
    with open(name, 'w', newline='') as csvfile:
        pass
    
    return



def append_states_to_csv(
    file_path:str, 
    states:list
)->None:
    """
    Appends a list of states to a CSV file.
    
    :param file_path: The path to the CSV file.
    :type file_path: str
    :param states: The list of states to append.
    :type states: list
    """
    with open(file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(states)

    return

# def init_env(env_cls):
    
#     """
#     Initializes an environment class for simulations, setting it to its initial state.
    
#     Args:
#         env_cls (class): The class of the environment to be initialized.
    
#     Returns:
#         object: An instance of the initialized environment.
#     """
    
#     env = env_cls()
#     env_cls.episode_length_index(0)
#     env.initialise_data()
#     env.reset()
#     return env



def create_unique_folder(
    base_foldername: str
)->str:
    """
    Creates a unique folder in the current directory. If a folder with the same name already exists,
    a counter is appended to the folder name (e.g., results0, results1, etc.).
    
    :param base_foldername: The base name of the folder without the counter.
    :type base_foldername: str
    
    :return: The path of the created folder.
    :rtype: str
    """
    counter = 0
    while True:
        foldername = f"{base_foldername}{counter}"
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            return foldername
        counter += 1




def read_and_copy_python_file(
    init_file:str,
    copied_file:str
)->None:
    
    with open(init_file, 'r') as file:
        content = file.read()
    with open(copied_file, 'w') as file:
        file.write(content)
        
    return



def compress_file(
    source_file_path:str, 
    destination_file_path:str
)->None:
    """
    Compresses a file using gzip and saves it to a new location.

    :param source_file_path: The path of the file to compress.
    :type source_file_path: str
    :param destination_file_path: The path where the compressed file will be saved.
    :type destination_file_path: str
    """
    with open(source_file_path, 'rb') as f_in:
        with gzip.open(destination_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    return

    
def seed_all(
    seed: int
)-> None:

    """
    Seed for current environment instance, called once per process during async_env training

    https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    https://stackoverflow.com/questions/47331235/how-should-openai-environments-gyms-use-env-seed0

    :param seed: Random seed
    :type seed: int
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return




def clear_folder(
    folder_path: str
) -> None:
    """
    Remove all files and subdirectories inside ``folder_path``,
    but leave ``folder_path`` itself intact.

    :param folder_path: The path of the folder to clear.
    :type folder_path: str
    """
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        return

    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        # If it's a file or a symlink, remove it
        if os.path.isfile(entry_path) or os.path.islink(entry_path):
            os.unlink(entry_path)
        # If it's a directory, remove it and all its contents
        elif os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
    return


def list_2_txt(
    list:list, 
    filename:str
)->None:
    
    with open(filename, 'w') as f:
        for item in list:
            f.write(str(item) + '\n')
    
    return 


def random_vector_know_norm(
    k:int,
    norm:float
)->np.ndarray:
    
    """
    Find a vector size k, with a know final norm

    Generates the vector value sfrom a multinomial dirichlet distribution
    
    https://en.wikipedia.org/wiki/Dirichlet_distribution

    :param k: desired vector size
    :type k: int
    :param norm: desired norm
    :type norm: float

    :return: output array (k,)
    :rtype: np.ndarray
    """
    
    flag_1 = False
    
    if norm<1:
        norm*=1e3
        flag_1 = True
        
    norm = norm**2
    
    weights = np.random.dirichlet(np.ones(k))  

    v = np.random.multinomial(norm, weights)
    v = np.sqrt(v) * np.random.choice([-1,1], size=k)
    
    if flag_1:
        v=v/1e3
    
    return v   



def generate_random_perpendicular_normalized_vector(
    n:np.ndarray
)->np.ndarray:
    
    """
    Generate a random unit vector perpendicular to the input vector.

    Reference: 

    .. code-block:: bibtex

        @MISC {2347293,
            TITLE = {How to find a random unit vector orthogonal to a random unit vector in 3D?},
            AUTHOR = {Ronald Blaak (https://math.stackexchange.com/users/458842/ronald-blaak)},
            HOWPUBLISHED = {Mathematics Stack Exchange},
            NOTE = {URL:https://math.stackexchange.com/q/2347293 (version: 2017-07-05)},
            EPRINT = {https://math.stackexchange.com/q/2347293},
            URL = {https://math.stackexchange.com/q/2347293}
        }

    :param n: input vector (3,)
    :type n: np.ndarray

    :return: perpendicular vector to input vector n (3,)
    :rtype: np.ndarray
    """
    
    n = n/np.linalg.norm(n)
    
    u = np.array(([n[1],-n[0],0]))
    
    v = np.cross(n,u)
    
    theta = np.random.uniform(0,2*np.pi)
    w = np.cos(theta)*u/np.linalg.norm(u) +np.sin(theta)*v/np.linalg.norm(v)
    
    return w


def gen_rv0(
    sma:float, 
    GM_Earth:float=3.986004415000000e+14
)->np.ndarray:
    """
    Calculate from the desired semimajor axis length the starting positon 
    and velocity vectors from a multinomial dirichlet distribution

    :param sma: semimajor axis value in meters.
    :type sma: float
    :param GM_Earth: Standard gravitational parameter of Earth
    :type GM_Earth: float

    :return: position and velocity vector in ECI
    :rtype: NDArray[np.float64]
    """

    nu = np.sqrt(GM_Earth / sma**3)           # mean motion
    v_norm = nu * sma * (1 + 1e-6)  

    # uniform random position on sphere of radius a
    r = random_vector_know_norm(k=3, norm=sma)
    # a vector perpendicular to p_vector, scaled to v_norm
    v = generate_random_perpendicular_normalized_vector(r) * v_norm
    
    assert_no_nan(r,"position")
    assert_no_nan(v,"velocity")
    
    return np.concatenate((r, v), axis=0)

def assert_no_nan(arr:Any, name:str):
    frame = inspect.currentframe().f_back
    assert not np.isnan(arr).any(), (
        f"{Path(frame.f_code.co_filename).resolve()}:{frame.f_lineno}: "
        f"{name} contains NaN:\n{arr}"
    )
