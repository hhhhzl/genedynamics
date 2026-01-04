"""
Obstacle generator plugin implementations.
"""

from .box2d import Box2DObstacleGeneratorPlugin
from .box3d import Box3DObstacleGeneratorPlugin

__all__ = ['Box2DObstacleGeneratorPlugin', 'Box3DObstacleGeneratorPlugin']

