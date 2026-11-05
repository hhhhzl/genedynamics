"""Environment-owned construction of contact simulation providers."""

from genedynamics.tasks.transfer.contact import ContactConfig
from genedynamics.tasks.transfer.runtime import MechanismSpec, mechanism_specs


class ContactRunner:
    """Select a numerical contact provider without exposing physics to policies."""

    def __init__(
        self,
        *,
        config=None,
        backend="numpy",
        device="cpu",
        mechanism=None,
        microbatch_size=None,
    ):
        self.config = config or ContactConfig()
        mechanism = MechanismSpec() if mechanism is None else mechanism
        mechanism_specs(mechanism)
        compliant = getattr(self.config, "fixture_profile", None) == "passive_x_v1"
        if compliant:
            from genedynamics.tasks.transfer.compliant_socket import CompliantSocketMechanismSpec
            if backend != "numpy" or device != "cpu":
                raise NotImplementedError("passive socket v1 is explicitly CPU-only")
            if not isinstance(mechanism, CompliantSocketMechanismSpec):
                raise TypeError("passive socket requires its declared construction mechanism")
        nonlegacy = (
            self.config.socket_profile != "legacy"
            or self.config.guide_profile != "legacy"
            or self.config.wrench_semantics != "legacy_aggregate"
        )
        if backend != "numpy" and nonlegacy:
            supported_jax = (
                backend == "jax"
                and self.config.socket_profile == "matched_keyway"
                and self.config.guide_profile == "legacy"
                and self.config.wrench_semantics == "tool_at_tcp_world"
                and self.config.robot in {"panda", "xarm7"}
                and not self.config.mobile
            )
            if not supported_jax:
                raise NotImplementedError(
                    "new JAX contact semantics are limited to fixed "
                    "Panda/xArm7 matched keyway/TCP wrench"
                )
        if microbatch_size is not None and (
            isinstance(microbatch_size, bool)
            or not isinstance(microbatch_size, int)
            or microbatch_size < 1
        ):
            raise ValueError("invalid contact microbatch size")
        if self.config.robot == "go2_z1":
            if not isinstance(mechanism, MechanismSpec):
                raise NotImplementedError(
                    "floating OSQP runner takes one mechanism per topology instance"
                )
            if backend != "numpy":
                raise NotImplementedError(
                    "floating-base QP adapter is CPU only; JAX/GPU provider not yet qualified"
                )
            from .backends.floating_contact import FloatingContact

            provider = FloatingContact
        elif backend == "numpy":
            if compliant:
                from .backends.contact import MujocoCompliantSocket
                provider = MujocoCompliantSocket
            elif isinstance(mechanism, MechanismSpec):
                from .backends.contact import MujocoContact

                provider = MujocoContact
            else:
                from .backends.contact_batch import MujocoContactBatch

                provider = MujocoContactBatch
        elif backend == "jax":
            from .backends.jax.contact import JaxContact

            provider = JaxContact
        else:
            raise ValueError("contact backend must be numpy or jax")
        if backend == "jax":
            self._impl = provider(
                self.config,
                device,
                mechanism,
                microbatch_size=microbatch_size,
            )
        else:
            self._impl = provider(self.config, device, mechanism)

    def reset_batch(self, request, *, previous=None, mask=None, initial="clear"):
        return self._impl.reset_batch(
            request, previous=previous, mask=mask, initial=initial
        )

    def collect_chunk(self, state, commands):
        return self._impl.collect_chunk(state, commands)

    evaluate_chunk = collect_chunk

    def synchronize(self, value):
        return self._impl.synchronize(value)

    def describe(self):
        return self._impl.describe()
