LEO-GYM
=======

**A reinforcement learning library for satellite control in Low Earth Orbit.**

LEO-GYM provides `Gymnasium <https://gymnasium.farama.org/>`_ environments,
high-fidelity orbital dynamics and RL algorithms for training policies that
control satellites operating in LEO, from station keeping to collision avoidance.

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 4 4 0 0

   .. grid-item-card:: :octicon:`globe;1.5em;sd-mr-1` Orbital dynamics
      :link: leo_gym.orbit
      :link-type: doc

      Numerical propagation with 20x20 gravity, Sun and Moon third-body
      effects, solar radiation pressure and atmospheric drag.

   .. grid-item-card:: :octicon:`rocket;1.5em;sd-mr-1` Gym environments
      :link: leo_gym.gyms
      :link-type: doc

      Station-keeping environments based on Relative Orbital Elements (ROE)
      for Along-Track and Across-Track maneuvers.

   .. grid-item-card:: :octicon:`cpu;1.5em;sd-mr-1` RL algorithms
      :link: leo_gym.rl_algorithms
      :link-type: doc

      Hybrid-action PPO (H-PPO) and a Semi-MDP extension of Stable-Baselines3
      PPO for variable-duration maneuvers.

   .. grid-item-card:: :octicon:`alert;1.5em;sd-mr-1` Collision avoidance
      :link: leo_gym.orbit.dynamics
      :link-type: doc

      Debris conjunction generation and B-plane collision probability metrics.

   .. grid-item-card:: :octicon:`graph;1.5em;sd-mr-1` Classical controllers
      :link: leo_gym.controllers
      :link-type: doc

      Linear MPC baseline built on the Clohessy-Wiltshire equations.

   .. grid-item-card:: :octicon:`tools;1.5em;sd-mr-1` Utilities
      :link: leo_gym.utils
      :link-type: doc

      MLflow logging, plotting styles and experiment helpers.


Installation
------------

It is recommended to first create a virtual environment, then clone and install
the library (use ``-e`` for editable mode):

.. code-block:: bash

   git clone https://github.com/NekTaf/leo_gym.git
   cd leo_gym
   pip install -e .

Alternatively, install directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/NekTaf/leo_gym.git


Tutorials
---------

Example notebooks are available in the
`notebooks <https://github.com/NekTaf/leo_gym/tree/main/notebooks>`_ folder:

- **Orbit perturbations visualization**: an introduction to the LEO-GYM structure,
  propagating ideal and perturbed orbits and plotting the LEO perturbations.
- **Across-Track (ACT) corrections**: training a policy to compute finite burn
  maneuvers, first with step-based discounting and then reformulated as a
  Semi-MDP to account for the variable maneuver duration.
- **LQR / MPC example**: LQR and linear MPC controllers tracking an ECI
  reference trajectory.


Citation
--------

If you use LEO-GYM in your research, please cite it as:

.. code-block:: bibtex

   @article{tafanidis2025leo,
     title={LEO-GYM: A Reinforcement Learning Library for Satellite Control in LEO},
     author={Tafanidis, Nektarios Aristeidis and Banerjee, Avijit and Nikolakopoulos, George},
     journal={IFAC-PapersOnLine},
     volume={59},
     number={31},
     pages={127--132},
     year={2025},
     publisher={Elsevier}
   }


Acknowledgements
----------------

This work has been partially funded by the European Space Agency (ESA) open
Invitations to Tender (ITT) and innovation research grant in OPTACOM project,
in collaboration with OHB Sweden under Grant Contract no: OPC-OSE-CC-0536.


.. toctree::
   :hidden:
   :caption: API Reference

   leo_gym.controllers
   leo_gym.gyms
   leo_gym.orbit
   leo_gym.rl_algorithms
   leo_gym.satellite
   leo_gym.utils

.. toctree::
   :hidden:
   :caption: Project

   GitHub <https://github.com/NekTaf/leo_gym>
   Index <genindex>
