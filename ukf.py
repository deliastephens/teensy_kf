import numpy as np

def quat_to_euler(q):
    """
    Converts the quaternion to the Euler angles
    """
    q0, q1, q2, q3 = q
    roll = np.arctan2(2*(q0*q1+q2*q3), 1-2*(q1**2+q2**2))
    pitch = np.arcsin(2*(q0*q2-q3*q1))
    yaw = np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2 + q3**2))
    
    return np.array([roll, pitch, yaw])
    
def vec_norm(v):
    """
    Vector normalization function. Returns v/norm(v)
    """
    return v / np.linalg.norm(v)

class UKF():
    def __init__(self, imu, dt, Q0, R, W, x0, g, r_m, n, mm, α, β,κ):
        """
        imu: IMU Object to access data in real time
        dt: samling time
        Q0: Initial covariance matrix
        R: ??
        W: ??
        x0: Initial state guess, [q0 q1 q2 q3 omega_x omega_y omega_z].T
        g: gravity vector
        r_m: magnetic inflication vector
        mm: number of measurements
        n: number of states
        """
        self.imu = imu
        self.dt = dt
        self.Q = Q0
        self.R = R
        self.W = W
        self.x0 = x0
        self.xhat = x0
        
        self.g = g
        self.r_m = r_m
        self.n = n
        self.mm = mm
        
        self.α = α
        self.κ = κ
        self.β = β
    
    def get_state(self):
        """
        Returns the quaternion state
        """
        return self.xhat
    
    def get_rpy(self):
        """
        Returns an np array of roll, pitch, yaw
        """
        return quat_to_euler(self.xhat[:4])
    
    def Omega(self, p, q, r):
        return np.array([
            [0, -p, -q, -r],
            [p, 0, r, -q],
            [q, -r, 0, p],
            [r, q, -p, 0]
        ])
    
    def f(self, sp):
        p_sp = np.zeros(np.shape(sp))
        for i in range(np.shape(sp)[1]):
            wx, wy, wz = sp[4:,i]
            sp[:4,i] = vec_norm(sp[:4,i]) # Normalize sigma point
            p_sp[:4,i] = (np.identity(4) + self.dt/2*self.Omega(wx, wy, wz))@sp[:4,i] # Propagate sigma points
            p_sp[:4,i] = vec_norm(p_sp[:4,i]) # Renormalize
        return p_sp

    def calc_C(self, sp):
        # Rotating vectors through quaternion
        # for singular sigma point
        # Generates the rotation matrix
        qw, qx, qy, qz = sp[:4] 

        return np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2)]
        ])

    def h(self, Xp):
        h = np.zeros((6, np.shape(Xp)[1]))
        for i in range(np.shape(Xp)[1]):
            Xp[:4,i] = vec_norm(Xp[:4,i]) # normalize Xp to enforce unit quaternion constraint
            C = self.calc_C(Xp[:,i])
            ahat = C.T@self.g
            mhat = C.T@self.r_m

            h[:,i] = np.vstack((ahat, mhat)).reshape(6,) # measurement model
        return h
    
    def get_measurements(self):
        # Get measurements
        Ax, Ay, Az = self.imu.get_acc()
        p, q, r = self.imu.get_gyr()
        mx, my, mz = self.imu.get_mag()

        self.z = np.array([[Ax, Ay, Az, mx, my, mz]]).T.reshape(6,)
        
    def get_weights(self, α, κ, β):
        # unscented transform weights
        n = self.n
        self.Wm = np.zeros(2 * n + 1)
        self.Wc = np.zeros(2 * n + 1)

        self.λ = α**2 * (n + κ) - n
        λ = self.λ
        self.Wm[:] = 0.5/(n + λ)
        self.Wc[:] = 0.5/(n + λ)
        self.Wm[0] = λ/(λ + n)
        self.Wc[0] = λ/(λ + n) + (1 - α**2 + β)
        
    def generate_sigma_points(self):
        # Generate sigma points
        self.Q = 1/2*(self.Q + self.Q.T) # Enforce PSD on Q to avoid numerical errors
        sqrtQ = np.linalg.cholesky((self.λ + self.n) * self.Q).T
        X = np.column_stack([self.xhat, (self.xhat + sqrtQ.T).T, (self.xhat - sqrtQ.T).T])
        
        return X
    
    def propagate(self):
        X = self.generate_sigma_points()
        # Pass sigma points through motion model
        Xp = self.f(X)

        # Recover Gaussian statistics - mean and covariance
        self.xhat = np.sum(self.Wm * Xp, axis=1)
        self.xhat[:4] = vec_norm(self.xhat[:4])
        self.Q = ((Xp.T - self.xhat).T * self.Wc).dot(Xp.T - self.xhat) + self.W
    
    def update(self):
        # Generate new sigma points
        Xp = self.generate_sigma_points()
        
        # Pass sigma points through measurement model
        Z = self.h(Xp)

        # Recover Gaussian statistics - mean and covariance
        zhat = np.sum(self.Wm * Z, axis=1)

        S = ((Z.T - zhat).T * self.Wc).dot(Z.T - zhat) + self.R

        # Recover cross covariance
        Σ_xz = ((Xp.T - self.xhat).T * self.Wc).dot(Z.T - zhat)
        
        # Perform measurement update
        K = Σ_xz @ np.linalg.inv(S)
        self.xhat += K @ (self.z - zhat)
        self.xhat[:4] = vec_norm(self.xhat[:4])
        self.Q += - K @ S @ K.T
        
        
        
    def run(self):
        self.get_measurements()
        
        self.get_weights(self.α, self.κ, self.β)

        self.propagate()
        
        self.update()

    