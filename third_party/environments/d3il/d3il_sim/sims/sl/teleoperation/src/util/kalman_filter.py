import numpy as np


class KalmanFilter:
    def __init__(self, X, dt=0.001, x_cov=1.0e-7, dx_cov=0, obs_noise_cov=1.2e-03):
        # Time interval
        self.sz = X.shape[0]

        # State vector
        self.X = np.append(X, np.zeros((self.sz,)))

        # Motion Model
        self.F = np.diag(
            np.ones(
                2 * self.sz,
            )
        )
        self.F[: self.sz, self.sz :] = np.diag(np.full((self.sz,), dt))

        # Motion Noise Covariance
        self.Q = np.diag(
            np.concatenate([np.full((self.sz,), x_cov), np.full((self.sz,), dx_cov)])
        )

        # Correlation Matrix
        self.P = self.Q

        # Observation Model
        self.H = np.zeros((7, 14))
        np.fill_diagonal(self.H, 1)

        # Observation Noise Covariance (load - grav)
        self.R = np.diag(np.full((self.sz,), obs_noise_cov))

        self.S = np.zeros((self.sz, self.sz))
        self.K = self.X

    def get_filtered(self, Z):
        self.X = self.F.dot(self.X)
        self.P = self.F.dot(self.P).dot(self.F.transpose()) + self.Q

        self.S = self.H.dot(self.P).dot(self.H.transpose()) + self.R
        self.K = self.P.dot(self.H.transpose()).dot(np.linalg.inv(self.S))
        self.X = self.X + self.K.dot(Z - self.H.dot(self.X))
        self.P = self.P - self.K.dot(self.S).dot(self.K.transpose())
        return self.X[: self.sz]
