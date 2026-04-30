"""JAX-native serial-arm kinematics.

Pure JAX implementation of forward kinematics, geometric Jacobian, damped
pseudo-inverse, and joint-space nullspace. URDF parsing happens once on host
(via ``urchin``) and is baked into a static ``_KinematicsChain`` description;
all numerical operations are ``jax.numpy`` and ``jaxlie`` so they jit, vmap,
and grad cleanly.

Conforms to :class:`KinematicsProtocol`.

Optional dependencies: ``jaxlie``, ``urchin`` (install via the ``manipulator``
extra).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# Joint type codes (used in the static chain; trade str for int so the chain
# is hashable and works as a static argument under jit).
JT_FIXED = 0
JT_REVOLUTE = 1
JT_PRISMATIC = 2


@dataclass(frozen=True)
class _KinematicsChain:
    """Static description of a base→ee chain.

    Stored as a tuple-of-tuples plus jnp arrays so the whole object is hashable
    (Python primitives) and the array fields can be passed through jit without
    re-tracing. ``n_dof`` is the number of *actuated* joints in the chain.

    Layout: there are ``n_links`` joints from base to ee in traversal order.
    For each link slot ``i`` we store ``joint_type[i]``, the parent→joint
    fixed transform ``origin[i]``, and the joint axis ``axis[i]``. A boolean
    ``is_actuated[i]`` marks which slots draw from ``q``; ``dof_index[i]``
    maps actuated slots to the column of ``q``.
    """

    # Static (Python primitives) — kept out of array leaves so jit treats them
    # as compile-time constants.
    joint_types: Tuple[int, ...]      # length n_links
    is_actuated: Tuple[bool, ...]     # length n_links
    dof_index: Tuple[int, ...]        # length n_links; -1 for fixed
    n_dof: int
    n_links: int
    base_link: str
    ee_link: str
    joint_names: Tuple[str, ...]      # actuated joints, length n_dof

    # Array fields (jit-traced).
    origin: jnp.ndarray               # (n_links, 4, 4) parent→joint fixed transform
    axis: jnp.ndarray                 # (n_links, 3) joint axis in joint frame
    joint_lower: jnp.ndarray          # (n_dof,)
    joint_upper: jnp.ndarray          # (n_dof,)
    vel_lower: jnp.ndarray            # (n_dof,)
    vel_upper: jnp.ndarray            # (n_dof,)


def _resolve_chain(robot, base_link: str, ee_link: str) -> List:
    """Walk parent→child once to find the joint chain from ``base`` to ``ee``.

    Raises ``ValueError`` if no path exists or links are unknown.
    """
    parent_of: dict = {}
    joint_to_parent: dict = {}
    for j in robot.joints:
        parent_of[j.child] = j.parent
        joint_to_parent[j.child] = j

    if ee_link not in parent_of and ee_link != base_link:
        raise ValueError(f"ee_link {ee_link!r} not found in URDF")

    chain_joints = []
    cur = ee_link
    while cur != base_link:
        if cur not in parent_of:
            raise ValueError(
                f"no path from {base_link!r} to {ee_link!r} (stopped at {cur!r})"
            )
        chain_joints.append(joint_to_parent[cur])
        cur = parent_of[cur]
    chain_joints.reverse()
    return chain_joints


def _build_chain(
    urdf_path: str,
    base_link: Optional[str],
    ee_link: str,
) -> _KinematicsChain:
    import urchin  # lazy

    robot = urchin.URDF.load(str(urdf_path), lazy_load_meshes=True)

    if base_link is None:
        # Pick the unique link that is not the child of any joint.
        children = {j.child for j in robot.joints}
        roots = [l.name for l in robot.links if l.name not in children]
        if len(roots) != 1:
            raise ValueError(
                f"cannot infer base_link automatically; candidates: {roots}"
            )
        base_link = roots[0]

    chain = _resolve_chain(robot, base_link, ee_link)

    types = []
    actuated = []
    dof_idx = []
    origins = []
    axes = []
    lower, upper, vlower, vupper = [], [], [], []
    actuated_names = []
    next_dof = 0
    for j in chain:
        origins.append(np.asarray(j.origin, dtype=np.float32))
        if j.joint_type == "revolute" or j.joint_type == "continuous":
            types.append(JT_REVOLUTE)
            actuated.append(True)
            dof_idx.append(next_dof)
            axes.append(np.asarray(j.axis, dtype=np.float32))
            actuated_names.append(j.name)
            lim = j.limit
            lower.append(float(lim.lower) if lim and lim.lower is not None else -np.pi)
            upper.append(float(lim.upper) if lim and lim.upper is not None else np.pi)
            vlim = float(lim.velocity) if lim and lim.velocity is not None else 1.0
            vlower.append(-vlim); vupper.append(vlim)
            next_dof += 1
        elif j.joint_type == "prismatic":
            types.append(JT_PRISMATIC)
            actuated.append(True)
            dof_idx.append(next_dof)
            axes.append(np.asarray(j.axis, dtype=np.float32))
            actuated_names.append(j.name)
            lim = j.limit
            lower.append(float(lim.lower) if lim and lim.lower is not None else -1.0)
            upper.append(float(lim.upper) if lim and lim.upper is not None else 1.0)
            vlim = float(lim.velocity) if lim and lim.velocity is not None else 1.0
            vlower.append(-vlim); vupper.append(vlim)
            next_dof += 1
        else:  # fixed (and unsupported types — treated as rigid)
            types.append(JT_FIXED)
            actuated.append(False)
            dof_idx.append(-1)
            axes.append(np.zeros(3, dtype=np.float32))

    return _KinematicsChain(
        joint_types=tuple(types),
        is_actuated=tuple(actuated),
        dof_index=tuple(dof_idx),
        n_dof=next_dof,
        n_links=len(chain),
        base_link=base_link,
        ee_link=ee_link,
        joint_names=tuple(actuated_names),
        origin=jnp.asarray(np.stack(origins, axis=0)),
        axis=jnp.asarray(np.stack(axes, axis=0)),
        joint_lower=jnp.asarray(np.array(lower, dtype=np.float32)),
        joint_upper=jnp.asarray(np.array(upper, dtype=np.float32)),
        vel_lower=jnp.asarray(np.array(vlower, dtype=np.float32)),
        vel_upper=jnp.asarray(np.array(vupper, dtype=np.float32)),
    )


def _hat(v: jnp.ndarray) -> jnp.ndarray:
    """3-vector to 3×3 skew-symmetric matrix."""
    zero = jnp.zeros((), dtype=v.dtype)
    return jnp.stack([
        jnp.stack([zero,   -v[2],  v[1]]),
        jnp.stack([v[2],    zero, -v[0]]),
        jnp.stack([-v[1],  v[0],   zero]),
    ])


def _so3_exp(axis: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Rodrigues: rotation matrix from axis (unit) and angle."""
    K = _hat(axis)
    s = jnp.sin(theta); c = jnp.cos(theta)
    return jnp.eye(3, dtype=axis.dtype) + s * K + (1.0 - c) * (K @ K)


def _se3_revolute(axis: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """4×4 transform for a revolute joint: rotation about ``axis`` by ``theta``."""
    R = _so3_exp(axis, theta)
    T = jnp.eye(4, dtype=axis.dtype)
    T = T.at[:3, :3].set(R)
    return T


def _se3_prismatic(axis: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """4×4 transform for a prismatic joint: translation along ``axis`` by ``d``."""
    T = jnp.eye(4, dtype=axis.dtype)
    T = T.at[:3, 3].set(axis * d)
    return T


def _se3_fixed(axis: jnp.ndarray) -> jnp.ndarray:
    """Identity transform (axis ignored — kept to share the lax.switch shape)."""
    return jnp.eye(4, dtype=axis.dtype)


def _joint_transform(joint_type: int, axis: jnp.ndarray, q_i: jnp.ndarray) -> jnp.ndarray:
    """Per-link joint transform; ``joint_type`` is a static int."""
    if joint_type == JT_REVOLUTE:
        return _se3_revolute(axis, q_i)
    if joint_type == JT_PRISMATIC:
        return _se3_prismatic(axis, q_i)
    return _se3_fixed(axis)


class JAXKinematics:
    """JAX-native forward kinematics + geometric Jacobian.

    Build with :meth:`from_urdf`. All public methods are jit/vmap/grad-safe.
    Conforms to :class:`KinematicsProtocol`.
    """

    def __init__(self, chain: _KinematicsChain) -> None:
        self._chain = chain

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path,
        ee_link: str,
        base_link: Optional[str] = None,
    ) -> "JAXKinematics":
        return cls(_build_chain(str(urdf_path), base_link, ee_link))

    @property
    def n_dof(self) -> int:
        return self._chain.n_dof

    @property
    def n_joints(self) -> int:
        return self._chain.n_links

    @property
    def joint_names(self) -> Tuple[str, ...]:
        return self._chain.joint_names

    @property
    def joint_limits(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._chain.joint_lower, self._chain.joint_upper

    @property
    def velocity_limits(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._chain.vel_lower, self._chain.vel_upper

    # ----- core kinematics ----------------------------------------------------

    def _accumulate_transforms(self, q: jnp.ndarray) -> jnp.ndarray:
        """Return ``T_base_to_link[i]`` for i = 0..n_links (n_links+1 entries).

        Index 0 is the base frame (identity); index k is the frame *after*
        applying joint k. The last entry is the end-effector pose.
        """
        ch = self._chain
        T = jnp.eye(4, dtype=q.dtype)
        Ts = [T]
        for i in range(ch.n_links):
            origin = ch.origin[i].astype(q.dtype)
            axis = ch.axis[i].astype(q.dtype)
            jt = ch.joint_types[i]
            if ch.is_actuated[i]:
                q_i = q[ch.dof_index[i]]
            else:
                q_i = jnp.zeros((), dtype=q.dtype)
            T_local = origin @ _joint_transform(jt, axis, q_i)
            T = T @ T_local
            Ts.append(T)
        return jnp.stack(Ts, axis=0)

    def fk(self, q: jnp.ndarray) -> jnp.ndarray:
        """Forward kinematics. Returns ``(4, 4)`` ee transform in base frame."""
        return self._accumulate_transforms(q)[-1]

    def jacobian(self, q: jnp.ndarray) -> jnp.ndarray:
        """Geometric Jacobian, ``(6, n_dof)``.

        Columns ordered by dof index. Rows 0–2 give linear-velocity
        contributions, rows 3–5 angular-velocity contributions, both in the
        base frame.
        """
        ch = self._chain
        Ts = self._accumulate_transforms(q)         # (n_links+1, 4, 4)
        p_ee = Ts[-1, :3, 3]
        J_cols = [None] * ch.n_dof
        for i in range(ch.n_links):
            if not ch.is_actuated[i]:
                continue
            T_parent = Ts[i]                        # frame BEFORE this joint
            origin_i = ch.origin[i].astype(q.dtype)
            axis_local = ch.axis[i].astype(q.dtype)
            # Joint axis expressed in base frame: parent rot · origin rot · axis
            R_joint = T_parent[:3, :3] @ origin_i[:3, :3]
            axis_world = R_joint @ axis_local
            # Joint origin in base frame.
            p_joint = (T_parent @ origin_i)[:3, 3]
            jt = ch.joint_types[i]
            if jt == JT_REVOLUTE:
                col = jnp.concatenate(
                    [jnp.cross(axis_world, p_ee - p_joint), axis_world]
                )
            else:  # prismatic
                col = jnp.concatenate([axis_world, jnp.zeros(3, dtype=q.dtype)])
            J_cols[ch.dof_index[i]] = col
        return jnp.stack(J_cols, axis=1)

    def jacobian_pinv(self, q: jnp.ndarray, damping: float = 0.0) -> jnp.ndarray:
        """Damped pseudo-inverse, shape ``(n_dof, 6)``.

        Branches statically on ``n_dof`` vs the twist dim (6):
          * ``n_dof > 6`` (wide / redundant): ``J^T (J J^T + λI)^{-1}``.
          * ``n_dof <= 6`` (tall / square / underactuated): ``(J^T J + λI)^{-1} J^T``.

        Both reduce to the Moore–Penrose pseudo-inverse as ``λ = damping**2``
        → 0, when ``J`` has full row/column rank respectively.
        """
        J = self.jacobian(q)
        lam2 = jnp.asarray(damping, dtype=q.dtype) ** 2
        n = self.n_dof
        if n > 6:
            JJt = J @ J.T
            reg = JJt + lam2 * jnp.eye(6, dtype=q.dtype)
            return J.T @ jnp.linalg.solve(reg, jnp.eye(6, dtype=q.dtype))
        else:
            JtJ = J.T @ J
            reg = JtJ + lam2 * jnp.eye(n, dtype=q.dtype)
            return jnp.linalg.solve(reg, J.T)

    def nullspace(self, q: jnp.ndarray) -> jnp.ndarray:
        """``N(q) = I - J†(q) J(q)``, shape ``(n_dof, n_dof)``."""
        J = self.jacobian(q)
        Jp = self.jacobian_pinv(q, damping=0.0)
        return jnp.eye(self.n_dof, dtype=q.dtype) - Jp @ J


__all__ = [
    "JAXKinematics",
    "JT_FIXED",
    "JT_REVOLUTE",
    "JT_PRISMATIC",
]
