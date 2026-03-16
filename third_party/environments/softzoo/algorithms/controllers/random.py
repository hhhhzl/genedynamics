from .base import Base


class Random(Base):
    def __init__(self, env):
        super(Random, self).__init__()
        self.env = env

    def forward(self, s, inp):
        return self.env.action_space.sample()

    def update(self, grad, retain_graph=False):
        pass
