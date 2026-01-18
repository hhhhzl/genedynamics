# D3IL Simulation Framework
This project contains the D3IL simulation environments code. 

## File System

```
SimulationFramework
└── environments.d3il.d3il_sim # basic simulation code, including robots, robot controller, camera, etc.
    └── controllers
    └── core
    └── gyms
    └── sims
    └── utils
    ...
└── envs # define all robot learning tasks here
    └── gym_pushing_env               # Pushing
    └── gym_stacking_env            # Stacking
    └── gym_avoiding_env      # Avoiding
    └── gym_aligning_env              # Aligning
    └── gym_sorting_env                 # Sorting
    └── gym_table_env                   # Arranging

└── models # xml files for robot and objects
    └── mj
    ...
...
```

## Register gym environment
In order to import the environment in other projects, you need to register each gym environment manually. 

For example, to register the Pushing environment:

```
# activate your conda environment
cd envs/gym_pushing_env/
pip install -e .
```

## Make your own environment
Based the 6 tasks we defined, you can easily make any task you want.

- Build a task folder following the gym_pushing_env
  ```
  └── gym_pushing_env/
    └── gym_pushing/
        └── envs/
            └── objects/            # objects implementation
                └── bp_objects.py   
            aligning.py           # Pushing environment definition
        __init__.py                 # gym register
    setup.py                        # setup for gym package
  ```
- Create the objects xml files in the `models/mj/common-objects/`
- Register your gym environment

## Key components for environments
Take Pushing task for example,

- Objects used in the environment should be defined under the folder `objects/` and the function
`get_obj_list()` should be implemented which returns the list of the required objects. If there are
custom objects that you defined in `models/mj/common-objects/`, you need to create a class which extends
`SimObject` and `MjXmlLoadable` (check the class `PushObject` in the Aligning task for an example).
- Each environment contains a context manager, which defines the observation space for objects and 
the related reset function. In `aligning.py`, the BlockContextManager defines the space of two 
blocks. Therefore, you should create such a manager to manage your objects for new tasks.
- The environment class should extend `GymEnvWrapper`. In `__init__()`, do the following steps:
    - Create a simulation factory `self.sim_factory` of the class `MjFactory`.
    - Create a scene object with `self.sim_factory.create_scene()`.
    - Create the robot object of the class `MjRobot`.
    - Call `super().__init__()`.
    - Define `self.action_space` and `self.observation_space`.
    - Create the dictionaries `self.log_dict` with `ObjectLogger` objects and `self.cam_dict` with
        `CamLogger` objects and add them to the scene with `self.scene.add_logger()`.
- The environment should implement the following functions:
  - `get_observation()` which computes an observation for your environment
  - `step()` which runs one timestep of the environment's dynamics based on the given action
  - `get_reward()` which calculates a reward (only for RL, not needed for IL)
  - `_check_early_termination()` which checks if a condition is satisfied for an early termination
  - `_reset_env()` which resets the environment to an initial state
- Given a policy, the `check_mode` function defines which mode it is currently executing. If you define 
a multi-modal task, you need to know how many modes/solutions there.