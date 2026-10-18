"""CPU reference for task-only candidate MPC with observable state restart."""
import time
import numpy as np
import mujoco
from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
from genedynamics.tasks.transfer.contact import ContactState


class NumpyContactPlanner:
    def __init__(self, config, mechanism, candidates, horizon_chunks):
        if config.mobile:
            raise NotImplementedError("fixed-arm MPC qualification only")
        self.config = config
        self.runner = ContactRunner(config=config, mechanism=mechanism)
        self.candidates = tuple(candidates)
        self.horizon = horizon_chunks
        self.decisions = 0
        self.candidate_rollouts = 0
        self.valid_control_steps = 0
        self.wall_seconds = 0.

    def restart(self, observed):
        """Use encoder q/qdot and known controller carry; no hidden model copy."""
        if len(observed.sample_id) != 1:
            raise NotImplementedError("CPU reference currently takes one observed world per call")
        model = self.runner._impl.model
        measured = observed.physics[0]
        if measured.qpos.shape != (model.nq,) or measured.qvel.shape != (model.nv,):
            raise ValueError("observed body topology does not match calibrated planner")
        data = mujoco.MjData(model)
        data.qpos[:] = measured.qpos
        data.qvel[:] = measured.qvel
        mujoco.mj_forward(model, data)
        return ContactState((data,), *(np.array(x, copy=True) for x in observed[1:]))

    def select(self, observed, carry, policy):
        start = time.monotonic()
        scores = []
        steps = []
        for candidate in self.candidates:
            state = self.restart(observed)
            memory = type(carry)(*(np.array(x, copy=True) for x in carry))
            local_costs = []
            valid_steps = 0
            for _ in range(self.horizon):
                command, memory, reference = policy(self.runner, state, memory)
                command = np.asarray(candidate.apply(command, phase=memory.phase))
                result = self.runner.collect_chunk(state, command)
                state = result.state
                mask = result.trace.valid[0]
                valid_steps += int(mask.sum())
                if mask.any():
                    position = result.trace.tcp_position[0, mask]
                    tracking = np.minimum(np.linalg.norm((position-reference[0, :3])/.02, axis=-1), 3.)/3.
                    force = np.minimum(np.abs(result.trace.normal_force[0, mask]-reference[0, 4])/10., 1.)
                    loaded = float(reference[0, 4] > .1)
                    local_costs.extend((.7*tracking+.3*loaded*force).tolist())
                if state.status[0] != 0:
                    break
            failure = float(state.status[0] not in (0, 1))
            # Same task tracking and force objective for every candidate. This
            # baseline does not consume the source response signature or truth.
            score = 10.*failure + (float(np.mean(local_costs)) if local_costs else 1.) - .05*float(memory.phase[0])
            scores.append(score)
            steps.append(valid_steps)
        self.decisions += 1
        self.candidate_rollouts += len(self.candidates)
        self.valid_control_steps += sum(steps)
        self.wall_seconds += time.monotonic()-start
        return int(np.argmin(scores)), {"scores": scores, "valid_control_steps": sum(steps)}

    def statistics(self):
        return {"decisions": self.decisions, "candidate_rollouts": self.candidate_rollouts,
                "valid_model_control_steps": self.valid_control_steps,
                "planner_wall_seconds": self.wall_seconds,
                "horizon_chunks": self.horizon, "candidate_count": len(self.candidates),
                "scope": "short-horizon finite candidate MPC; online model work is additional to offline search; no global optimality claim"}
