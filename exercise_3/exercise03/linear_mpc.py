import numpy as np
from crazyflow.constants import GRAVITY, MASS
from crazyflow.gymnasium_envs.crazyflow import CrazyflowBaseEnv

from exercise03.ocp_setup import create_ocp_linear

try:
    from base_controller import BaseController
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # required for importing base_controller in development repository
    from base_controller import BaseController
    


class LinearModelPredictiveController(BaseController):
    """Model Predictive Controller (MPC) for linear dynamic systems."""

    def __init__(self, env: CrazyflowBaseEnv, options: dict):
        """Initialize the Model Predictive Controller.

        Args:
            env: The env for controller deployment.
            options: Configuration options for the MPC.
        """
        self.options = options
        self.symbolic_model = self.get_symbolic(env)

        ## parameters ##
        self.N = options["solver"]["n_pred"]
        self.dt = options["solver"]["Ts"]

        self.u_op = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0])
        self.u_pred = np.tile(self.u_op, (options["solver"]["n_pred"], 1))
        self.x_pred = np.zeros(
            (options["solver"]["n_pred"] + 1, self.symbolic_model.x_sym.size()[0])
        )

        ## create acados ocp solver
        self.ocp, self.ocp_solver = create_ocp_linear(self.symbolic_model, options)

    def step_control(self, x: np.ndarray, y_ref: np.ndarray, y_ref_e: np.ndarray) -> np.ndarray:
        """Compute the control input.

        Args:
            x (np.ndarray): The current state of the system, with shape (nx,).
            y_ref (np.ndarray): The reference trajectory for the states, with
                                shape (horizon, n_y)
            y_ref_e (np.ndarray): The reference trajectory for the terminal
                                  state, with shape (n_y_e,)

        Returns:
            u: The control action of current step, with shape (1, nu).
        """
        ########################################################################
        # Task 6
        # TODO:
        # 1. Set initial state constraints
        # 2. Set the reference trajectory
        # 3. Warm start
        # 4. Solve the OCP
        # 5. Extract the control
        # 6. Store predictions
        # Hints:
        # 1. Please use the methods of AcadosOcpSolver
        # 2. If the solver fails to converge, consider returning a safe
        #    fallback control (such as a zero vector or the previous control)
        #    to ensure system stability and safety.
        # 3. Don't forget to reshape and convert the control input to the
        #    appropriate shape and dtype for your environment.
        ########################################################################
        



























































        ########################################################################
        #                           END OF YOUR CODE
        ########################################################################
        return u
