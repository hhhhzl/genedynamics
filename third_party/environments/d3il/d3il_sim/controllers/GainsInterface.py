import gin


class PDGains:
    def __init__(self, pgain, dgain):
        self.pgain = pgain
        self.dgain = dgain


@gin.configurable
class JointPDGains(PDGains):
    def __init__(self, pgain, dgain):
        super(JointPDGains, self).__init__(pgain, dgain)


@gin.configurable
class DampingGains(PDGains):
    def __init__(self, dgain):
        super(DampingGains, self).__init__(None, dgain)


@gin.configurable
class CartPosControllerConfig:
    def __init__(self, pgain_pos, J_reg, W, pgain_null, rest_posture):
        self.pgain_pos = pgain_pos
        self.J_reg = J_reg  # Jacobian regularization constant
        self.W = W

        # Null-space theta configuration
        self.rest_posture = rest_posture
        self.pgain_null = pgain_null


@gin.configurable
class CartPosQuatControllerConfig:
    def __init__(
        self,
        pgain_pos,
        pgain_quat,
        J_reg,
        W,
        pgain_null,
        rest_posture,
        ddgain,
        joint_filter_coefficient,
        min_svd_values,
        max_svd_values,
        num_iter,
        learningRate,
    ):

        self.pgain_pos = pgain_pos
        self.pgain_quat = pgain_quat
        self.J_reg = J_reg  # Jacobian regularization constant
        self.W = W

        # Null-space theta configuration
        self.rest_posture = rest_posture
        self.pgain_null = pgain_null

        self.ddgain = ddgain
        self.joint_filter_coefficient = joint_filter_coefficient
        self.min_svd_values = min_svd_values
        self.max_svd_values = max_svd_values

        self.num_iter = num_iter
        self.learningRate = learningRate


@gin.configurable
class CartPosQuatJacTransposeControllerConfig:
    def __init__(
        self,
        pgain_pos,
        dgain,
        pgain_quat,
        J_reg,
        W,
        pgain_null,
        dgain_null,
        rest_posture,
    ):

        self.pgain_pos = pgain_pos
        self.dgain = dgain
        self.pgain_quat = pgain_quat
        self.dgain_null = dgain_null
        self.J_reg = J_reg  # Jacobian regularization constant
        self.W = W

        # Null-space theta configuration
        self.rest_posture = rest_posture
        self.pgain_null = pgain_null
