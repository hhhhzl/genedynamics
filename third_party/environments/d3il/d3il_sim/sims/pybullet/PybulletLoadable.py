import abc


class PybulletLoadable(abc.ABC):
    def pb_load(self, pb_sim) -> int:
        raise NotImplementedError
