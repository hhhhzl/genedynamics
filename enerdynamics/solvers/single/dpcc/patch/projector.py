import numpy as np
import torch
from scipy.optimize import Bounds, minimize


class Projector:

    def __init__(
        self,
        horizon,
        transition_dim,
        action_dim=0,
        goal_dim=0,
        constraint_list=None,
        normalizer=None,
        variant="states",
        dt=0.1,
        cost_dims=None,
        skip_initial_state=True,
        diffusion_timestep_threshold=0.5,
        gradient=False,
        gradient_weights=None,
        device="cuda",
        solver="proxsuite",
        parallelize=False,
    ):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.dt = torch.tensor(dt, device=device)
        self.skip_initial_state = skip_initial_state
        self.diffusion_timestep_threshold = diffusion_timestep_threshold
        self.gradient = gradient
        self.gradient_weights = gradient_weights
        self.device = device
        self.solver = solver
        self.parallelize = parallelize

        constraint_list = constraint_list or []

        if normalizer is None:
            self.normalizer = None
        elif variant == "states":
            self.normalizer = ProjectionNormalizer(
                observation_normalizer=normalizer.normalizers["observations"],
                goal_dim=goal_dim,
            )
        elif variant == "states_actions":
            self.normalizer = ProjectionNormalizer(
                observation_normalizer=normalizer.normalizers["observations"],
                action_normalizer=normalizer.normalizers["actions"],
                goal_dim=goal_dim,
            )
        else:
            raise KeyError('Invalid variant. Choose either "states" or "states_actions".')

        if cost_dims is not None:
            costs = torch.ones(transition_dim, device=self.device)
            for idx in cost_dims:
                costs[idx] = 1
            self.Q = torch.diag(torch.tile(costs, (self.horizon,)))
        else:
            self.Q = torch.eye(transition_dim * horizon, device=self.device)

        self.A = torch.empty((0, self.transition_dim * self.horizon), device=self.device)
        self.b = torch.empty(0, device=self.device)
        self.C = torch.empty((0, self.transition_dim * self.horizon), device=self.device)
        self.d = torch.empty(0, device=self.device)

        self.safety_constraints = SafetyConstraints(
            horizon=horizon,
            transition_dim=transition_dim,
            normalizer=self.normalizer,
            skip_initial_state=self.skip_initial_state,
            action_dim=action_dim,
            device=self.device,
        )
        self.dynamic_constraints = DynamicConstraints(
            horizon=horizon,
            transition_dim=transition_dim,
            normalizer=self.normalizer,
            skip_initial_state=self.skip_initial_state,
            dt=self.dt,
            device=self.device,
        )
        self.obstacle_constraints = ObstacleConstraints(
            horizon=horizon,
            transition_dim=transition_dim,
            normalizer=self.normalizer,
            skip_initial_state=self.skip_initial_state,
            dt=self.dt,
            device=self.device,
        )

        for constraint_spec in constraint_list:
            if isinstance(constraint_spec, (list, tuple)) and str(constraint_spec[0]).startswith("deriv"):
                self.dynamic_constraints.constraint_list.append(constraint_spec)
            elif constraint_spec[0] in {"lb", "ub", "eq", "ineq"}:
                self.safety_constraints.constraint_list.append(constraint_spec)
            elif constraint_spec[0] in {"sphere_inside", "sphere_outside"}:
                self.obstacle_constraints.constraint_list.append(constraint_spec)

        self.safety_constraints.build_matrices()
        self.dynamic_constraints.build_matrices()
        self.obstacle_constraints.build_matrices()
        self.append_linear_constraint(self.safety_constraints)
        self.append_linear_constraint(self.dynamic_constraints)
        self.add_numpy_constraints()

    def project(self, trajectory, constraints=None):
        dims = trajectory.shape
        batch_size = trajectory.shape[0]
        trajectory_reshaped = trajectory.reshape(trajectory.shape[0], -1)

        r = -trajectory_reshaped @ self.Q
        r_np = r.cpu().numpy()
        q = self.Q_np.astype("double")
        trajectory_np = trajectory_reshaped.cpu().numpy()

        A = self.A_np.astype("double")
        b = self.b_np.astype("double")
        C = self.C_np.astype("double")
        d = self.d_np.astype("double")

        if self.skip_initial_state:
            s_0 = trajectory_reshaped[0, : self.transition_dim]
            if self.solver in {"proxsuite", "gurobi"}:
                s_0 = s_0.cpu().numpy()
            counter = 0
            for constraint in self.dynamic_constraints.constraint_list:
                if isinstance(constraint, (list, tuple)) and str(constraint[0]).startswith("deriv"):
                    vals = constraint[1]
                    # ("deriv", [x_idx, dx_idx]) or ("deriv_lin", (x_idx, [dx_idxs], [coeffs]))
                    x_idx = int(vals[0]) if hasattr(vals, "__len__") else int(vals)
                    b[counter * self.horizon] = s_0[x_idx]
                    counter += 1

        r_np_double = r_np.astype("double")
        trajectory_np_double = trajectory_np.astype("double")
        constraints = ()
        for constraint_idx in range(len(self.obstacle_constraints.P_list)):
            P = self.obstacle_constraints.P_list[constraint_idx]
            q_vec = self.obstacle_constraints.q_list[constraint_idx]
            v = self.obstacle_constraints.v_list[constraint_idx]
            for t in range(1, self.horizon):
                start_idx = t * self.transition_dim
                end_idx = (t + 1) * self.transition_dim
                constraints += (
                    {
                        "type": "ineq",
                        "fun": lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q_vec=q_vec, v=v: -x[
                            start_idx:end_idx
                        ]
                        @ P
                        @ x[start_idx:end_idx]
                        - q_vec @ x[start_idx:end_idx]
                        + v,
                        "jac": lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q_vec=q_vec: np.concatenate(
                            [
                                np.zeros(start_idx),
                                -2 * P @ x[start_idx:end_idx] - q_vec,
                                np.zeros(len(x) - end_idx),
                            ]
                        ),
                    },
                )

        if C.size > 0:
            constraints += ({"type": "ineq", "fun": lambda x: -C @ x + d, "jac": lambda x: -C},)
        if A.size > 0:
            constraints += ({"type": "eq", "fun": lambda x: A @ x - b, "jac": lambda x: A},)

        projection_costs = np.ones(batch_size, dtype=np.float32)
        sol_np = np.zeros((batch_size, self.horizon * self.transition_dim), dtype=np.float32)
        for i in range(batch_size):
            cost_fun = lambda x: 0.5 * x @ q @ x + r_np_double[i] @ x
            jac_cost_fun = lambda x: q @ x + r_np_double[i]
            res = minimize(
                fun=cost_fun,
                x0=trajectory_np_double[i],
                constraints=constraints,
                method="SLSQP",
                jac=jac_cost_fun,
                bounds=Bounds(-5 * np.ones_like(trajectory_np_double[i]), 5 * np.ones_like(trajectory_np_double[i])),
                tol=1e-6,
                options={"maxiter": 1000, "disp": False},
            )

            sol_np[i] = res.x
            projection_costs[i] = (
                0.5 * sol_np[i] @ q @ sol_np[i]
                + r_np[i] @ sol_np[i]
                + 0.5 * trajectory_np[i] @ q @ trajectory_np[i]
            )

        sol = torch.tensor(sol_np, device=self.device).reshape(dims)
        return sol, projection_costs

    def compute_gradient(self, trajectory, constraints=None):
        trajectory_reshaped = trajectory.reshape(trajectory.shape[0], -1)
        trajectory_np = trajectory_reshaped.cpu().numpy()

        A, b, C, d = self.A, self.b, self.C, self.d

        if self.skip_initial_state:
            s_0 = trajectory_reshaped[0, : self.transition_dim]
            counter = 0
            for constraint in self.dynamic_constraints.constraint_list:
                if constraint[0] == "deriv":
                    x_idx = int(constraint[1][0])
                    b[counter * self.horizon] = s_0[x_idx]
                    counter += 1

        grad1 = torch.zeros_like(trajectory_reshaped)
        grad2 = torch.zeros_like(trajectory_reshaped)
        for i in range(trajectory.shape[0]):
            grad1[i] = -A.T @ (A @ trajectory_reshaped[i] - b)
            grad2[i] = -C.T @ torch.max(
                torch.zeros_like(C @ trajectory_reshaped[i] - d), C @ trajectory_reshaped[i] - d
            )
        grad1 = grad1.reshape(trajectory.shape)
        grad2 = grad2.reshape(trajectory.shape)

        grad3 = np.zeros_like(trajectory_np)
        for constraint_idx in range(len(self.obstacle_constraints.P_list)):
            P = self.obstacle_constraints.P_list[constraint_idx]
            q = self.obstacle_constraints.q_list[constraint_idx]
            v = self.obstacle_constraints.v_list[constraint_idx]
            for t in range(1, self.horizon):
                start_idx = t * self.transition_dim
                end_idx = (t + 1) * self.transition_dim
                for i in range(trajectory.shape[0]):
                    if trajectory_np[i, start_idx:end_idx] @ P @ trajectory_np[i, start_idx:end_idx] + q @ trajectory_np[i, start_idx:end_idx] <= v:
                        continue
                    grad3[i, start_idx:end_idx] -= 2 * P @ trajectory_np[i, start_idx:end_idx] + q
        grad3 = torch.tensor(grad3, device=self.device).reshape(trajectory.shape)

        if self.gradient_weights is not None:
            grad1 = grad1 * self.gradient_weights[0]
            grad2 = grad2 * self.gradient_weights[1]
            grad3 = grad3 * self.gradient_weights[2]

        return grad1 + grad2 + grad3

    def append_linear_constraint(self, constraint):
        self.C = torch.cat([self.C, constraint.C], dim=0)
        self.d = torch.cat([self.d, constraint.d], dim=0)
        self.A = torch.cat([self.A, constraint.A], dim=0)
        self.b = torch.cat([self.b, constraint.b], dim=0)
        if constraint.__class__.__name__ == "SafetyConstraints":
            self.A_safe, self.b_safe, self.C_safe, self.d_safe = (
                constraint.A,
                constraint.b,
                constraint.C,
                constraint.d,
            )
        elif constraint.__class__.__name__ == "DynamicConstraints":
            self.A_dyn, self.b_dyn, self.C_dyn, self.d_dyn = (
                constraint.A,
                constraint.b,
                constraint.C,
                constraint.d,
            )

    def add_numpy_constraints(self):
        self.A_np = self.A.cpu().numpy()
        self.b_np = self.b.cpu().numpy()
        self.C_np = self.C.cpu().numpy()
        self.d_np = self.d.cpu().numpy()
        self.Q_np = self.Q.cpu().numpy()


class Constraints:

    def __init__(self, horizon, transition_dim, normalizer=None, device="cuda"):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.normalizer = normalizer
        self.device = device

        self.A = torch.empty((0, self.transition_dim * self.horizon), device=device)
        self.b = torch.empty(0, device=device)
        self.C = torch.empty((0, self.transition_dim * self.horizon), device=device)
        self.d = torch.empty(0, device=device)

    def build_matrices(self):
        pass


class SafetyConstraints(Constraints):

    def __init__(self, skip_initial_state=True, action_dim=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.action_dim = action_dim
        self.constraint_list = []

    def build_matrices(self, constraint_list=None):
        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        for constraint in constraint_list:
            type = constraint[0]
            bound = constraint[1]
            if type in {"lb", "ub"}:
                for dim in range(len(bound)):
                    if bound[dim] in {-np.inf, np.inf}:
                        continue

                    mat_append = torch.zeros(self.horizon, self.transition_dim * self.horizon, device=self.device)
                    vec_append = torch.zeros(self.horizon, device=self.device)

                    sign = 1 if type == "ub" else -1
                    for t in range(self.horizon):
                        mat_append[t, t * self.transition_dim + dim] = sign
                        vec_append[t] = sign * bound[dim]

                    if self.normalizer is not None:
                        x_min = self.normalizer.mins[dim]
                        x_max = self.normalizer.maxs[dim]
                        mat_append = mat_append * (x_max - x_min) / 2
                        vec_append = vec_append - sign * (x_min + x_max) / 2

                    if self.skip_initial_state and dim >= self.action_dim:
                        mat_append = mat_append[1:]
                        vec_append = vec_append[1:]

                    self.C = torch.cat((self.C, mat_append), dim=0)
                    self.d = torch.cat((self.d, vec_append), dim=0)
                continue

            mat_append = torch.zeros(self.horizon, self.transition_dim * self.horizon, device=self.device)
            vec_append = torch.zeros(self.horizon, device=self.device)

            for i in range(self.horizon):
                if self.normalizer is not None:
                    x_min = self.normalizer.mins
                    x_max = self.normalizer.maxs
                    a = bound[0] * (x_max - x_min) / 2
                    b = bound[1] - bound[0] @ (x_max + x_min) / 2
                else:
                    a = bound[0]
                    b = bound[1]

                mat_append[i, i * self.transition_dim : (i + 1) * self.transition_dim] = torch.tensor(
                    a, device=self.device
                )
                vec_append[i] = torch.tensor(b, device=self.device)

            if self.skip_initial_state:
                mat_append = mat_append[1:]
                vec_append = vec_append[1:]

            if type == "eq":
                self.A = torch.cat((self.A, mat_append), dim=0)
                self.b = torch.cat((self.b, vec_append), dim=0)
            else:
                self.C = torch.cat((self.C, mat_append), dim=0)
                self.d = torch.cat((self.d, vec_append), dim=0)


class DynamicConstraints(Constraints):
    def __init__(self, skip_initial_state=True, dt=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.dt = dt
        self.constraint_list = []

    def build_matrices(self, constraint_list=None):
        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        for constraint in constraint_list:
            type = constraint[0]
            vals = constraint[1]
            if type == "deriv":
                x_idx = int(vals[0])
                dx_idx = int(vals[1])

                mat_append = torch.zeros(self.horizon - 1, self.transition_dim * self.horizon, device=self.device)
                vec_append = torch.zeros(self.horizon - 1, device=self.device)

                if self.normalizer is not None:
                    x_min = float(self.normalizer.mins[x_idx])
                    x_max = float(self.normalizer.maxs[x_idx])
                    dx_min = float(self.normalizer.mins[dx_idx])
                    dx_max = float(self.normalizer.maxs[dx_idx])
                    x_diff = x_max - x_min
                    dx_diff = dx_max - dx_min
                    dx_sum = dx_max + dx_min

                for i in range(self.horizon - 1):
                    if self.normalizer is not None:
                        mat_append[i, i * self.transition_dim + x_idx] = 1 * x_diff
                        mat_append[i, i * self.transition_dim + dx_idx] = self.dt * dx_diff
                        mat_append[i, (i + 1) * self.transition_dim + x_idx] = -1 * x_diff
                        vec_append[i] = -dx_sum * self.dt
                    else:
                        mat_append[i, i * self.transition_dim + x_idx] = 1
                        mat_append[i, i * self.transition_dim + dx_idx] = self.dt
                        mat_append[i, (i + 1) * self.transition_dim + x_idx] = -1
                        vec_append[i] = 0

                if self.skip_initial_state:
                    mat_fix_initial = torch.zeros(1, self.transition_dim * self.horizon, device=self.device)
                    mat_fix_initial[0, x_idx] = 1
                    mat_append = torch.cat((mat_fix_initial, mat_append), dim=0)
                    vec_append = torch.cat((torch.tensor([0], device=self.device), vec_append), dim=0)

                self.A = torch.cat((self.A, mat_append), dim=0)
                self.b = torch.cat((self.b, vec_append), dim=0)
            elif type == "deriv_lin":
                # Linear dynamics: x_{t+1} = x_t + dt * sum_k coeff_k * dx_k_t
                # vals = (x_idx, dx_idxs, coeffs)
                x_idx = int(vals[0])
                dx_idxs = list(vals[1])
                coeffs = np.asarray(vals[2], dtype=np.float32).reshape(-1)
                if len(dx_idxs) != int(coeffs.size):
                    raise ValueError("deriv_lin expects matching dx_idxs and coeffs lengths")

                mat_append = torch.zeros(self.horizon - 1, self.transition_dim * self.horizon, device=self.device)
                vec_append = torch.zeros(self.horizon - 1, device=self.device)

                if self.normalizer is not None:
                    x_min = float(self.normalizer.mins[x_idx])
                    x_max = float(self.normalizer.maxs[x_idx])
                    x_diff = x_max - x_min
                    # For each dx dim
                    dx_min = [float(self.normalizer.mins[int(j)]) for j in dx_idxs]
                    dx_max = [float(self.normalizer.maxs[int(j)]) for j in dx_idxs]
                    dx_diff = [dx_max[k] - dx_min[k] for k in range(len(dx_idxs))]
                    dx_sum = [dx_max[k] + dx_min[k] for k in range(len(dx_idxs))]

                for i in range(self.horizon - 1):
                    if self.normalizer is not None:
                        mat_append[i, i * self.transition_dim + x_idx] = 1 * x_diff
                        mat_append[i, (i + 1) * self.transition_dim + x_idx] = -1 * x_diff
                        const = 0.0
                        for k, dx_idx in enumerate(dx_idxs):
                            mat_append[i, i * self.transition_dim + int(dx_idx)] = (
                                mat_append[i, i * self.transition_dim + int(dx_idx)]
                                + float(self.dt) * float(coeffs[k]) * float(dx_diff[k])
                            )
                            const += float(coeffs[k]) * float(dx_sum[k])
                        vec_append[i] = -const * float(self.dt)
                    else:
                        mat_append[i, i * self.transition_dim + x_idx] = 1
                        mat_append[i, (i + 1) * self.transition_dim + x_idx] = -1
                        for k, dx_idx in enumerate(dx_idxs):
                            mat_append[i, i * self.transition_dim + int(dx_idx)] = (
                                mat_append[i, i * self.transition_dim + int(dx_idx)]
                                + float(self.dt) * float(coeffs[k])
                            )
                        vec_append[i] = 0

                if self.skip_initial_state:
                    mat_fix_initial = torch.zeros(1, self.transition_dim * self.horizon, device=self.device)
                    mat_fix_initial[0, x_idx] = 1
                    mat_append = torch.cat((mat_fix_initial, mat_append), dim=0)
                    vec_append = torch.cat((torch.tensor([0], device=self.device), vec_append), dim=0)

                self.A = torch.cat((self.A, mat_append), dim=0)
                self.b = torch.cat((self.b, vec_append), dim=0)


class ObstacleConstraints(Constraints):
    def __init__(self, skip_initial_state=True, dt=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.dt = dt
        self.constraint_list = []

    def build_matrices(self, constraint_list=None):
        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        self.P_list = []
        self.q_list = []
        self.v_list = []
        for constraint in constraint_list:
            type = constraint[0]
            dims = constraint[1]
            center = constraint[2]
            radius = constraint[3]

            P = np.zeros((self.transition_dim, self.transition_dim))
            q = np.zeros(self.transition_dim)
            v = radius**2

            dim_counter = 0
            for dim in dims:
                if self.normalizer is not None:
                    delta_s = self.normalizer.maxs[dim] - self.normalizer.mins[dim]
                    s_min = self.normalizer.mins[dim]
                    P[dim, dim] = delta_s**2 / 4
                    q[dim] = delta_s**2 / 2 + delta_s * (s_min - center[dim_counter])
                    v -= (
                        delta_s**2 / 4
                        + delta_s * (s_min - center[dim_counter])
                        + (s_min - center[dim_counter]) ** 2
                    )
                else:
                    P[dim, dim] = 1
                    q[dim] = -2 * center[dim_counter]
                    v -= center[dim_counter] ** 2
                dim_counter += 1

            if type == "sphere_outside":
                P = -P
                q = -q
                v = -v

            self.P_list.append(P)
            self.q_list.append(q)
            self.v_list.append(v)


class ProjectionNormalizer:
    def __init__(self, observation_normalizer=None, action_normalizer=None, goal_dim=0):
        self.observation_normalizer = observation_normalizer
        self.action_normalizer = action_normalizer
        self.goal_dim = goal_dim
        self.get_limits()

    def get_limits(self):
        if self.observation_normalizer is not None and self.action_normalizer is not None:
            x_max_obs = (
                self.observation_normalizer.maxs[:-self.goal_dim]
                if self.goal_dim > 0
                else self.observation_normalizer.maxs
            )
            x_min_obs = (
                self.observation_normalizer.mins[:-self.goal_dim]
                if self.goal_dim > 0
                else self.observation_normalizer.mins
            )
            x_max = np.concatenate([self.action_normalizer.maxs, x_max_obs])
            x_min = np.concatenate([self.action_normalizer.mins, x_min_obs])
        elif self.observation_normalizer is not None:
            x_max = (
                self.observation_normalizer.maxs[:-self.goal_dim]
                if self.goal_dim > 0
                else self.observation_normalizer.maxs
            )
            x_min = (
                self.observation_normalizer.mins[:-self.goal_dim]
                if self.goal_dim > 0
                else self.observation_normalizer.mins
            )
        elif self.action_normalizer is not None:
            x_max = self.action_normalizer.maxs
            x_min = self.action_normalizer.mins
        else:
            raise ValueError("ProjectionNormalizer requires observation or action normalizers.")

        self.maxs = x_max
        self.mins = x_min

    def normalize(self, x):
        return (x - self.mins) / (self.maxs - self.mins) * 2 - 1

    def unnormalize(self, x_normalized):
        return (x_normalized + 1) * (self.maxs - self.mins) / 2 + self.mins
