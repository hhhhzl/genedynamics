"""Solver-style response calibration for collision-free 3D arm motion."""
class ObstacleResponse:
    def __init__(self,config,skill,mechanism,*,backend="numpy",device="cpu"):
        if backend!="numpy" or device!="cpu":raise NotImplementedError("obstacle twin requires qualified numpy/cpu provider")
        from .backends.numpy.obstacle_response import NumpyObstacleResponse
        self._impl=NumpyObstacleResponse(config,skill,mechanism)
    def predict_probe(self,record):return self._impl.predict_probe(record)


def fit_obstacle_context(config,skill,records):
    from .backends.numpy.obstacle_response import fit_obstacle_context as fit
    return fit(config,skill,records)


def validate_obstacle_response(config,skill,mechanism,records):
    from .backends.numpy.obstacle_response import validate_obstacle_response as validate
    return validate(config,skill,mechanism,records)
