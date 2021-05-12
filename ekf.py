class EKF():
    def __init__(self, imu, dt, Qc, sa2, sm2, g, r):
        self.g = g
        self.r_m = r
        self.imu = imu
        self.dt = dt
        self.Qc = Qc
        
        # Initializes P, Q, and R matrices
        self.P = np.identity(4)
        self.Q = np.identity(4)*Qc
        R = np.hstack((np.identity(3)*sa2, np.zeros((3,3))))
        self.R = np.vstack((R, np.hstack((np.zeros((3,3)),(np.identity(3)*sm2)))))
        
        # Initial state guess
        self.x = np.array([[1], [0], [0], [0]])
    
    def get_state(self):
        """
        Returns the quaternion state
        """
        return self.x
    
    def get_rpy(self):
        """
        Returns an np array of roll, pitch, yaw
        """
        return quat_to_euler(self.x)
    
    def calc_F(self):
        # Unpacking (to not have lots of "selfs")
        p = self.p
        q = self.q
        r = self.r
        dt = self.dt
        
        return np.array([
            [1, -dt/2*p, -dt/2*q, -dt/2*r],
            [dt/2*p, 1, dt/2*r, -dt/2*q],
            [dt/2*q, -dt/2*r, 1, dt/2*p],
            [dt/2*r, dt/2*q, -dt/2*p, 1],
        ])
    
    def calc_W(self):
        qw, qx, qy, qz = self.x[:,0] # inconsistent to stay with notation from ahrs readthedocs

        res = np.array([
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw]
        ])

        return self.dt/2.*res
    
    def calc_C(self):
        # Rotating vectors through quaternion
        # Generates the rotation matrix
        qw, qx, qy, qz = self.x[:,0] 

        return np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2)]
        ])
    
    def calc_H(self):
        # measurement matrix jacobian, can refactor later
        qw, qx, qy, qz = self.x[:,0] 
        v = np.vstack((self.g, self.r_m))

        H = np.array([[-qy*v[2] + qz*v[1],  qy*v[1] + qz*v[2], -qw*v[2] + qx*v[1] - 2.0*qy*v[0],  qw*v[1] + qx*v[2] - 2.0*qz*v[0]],
                  [ qx*v[2] - qz*v[0],  qw*v[2] - 2.0*qx*v[1] + qy*v[0],  qx*v[0] + qz*v[2], -qw*v[0] + qy*v[2] - 2.0*qz*v[1]],
                  [-qx*v[1] + qy*v[0], -qw*v[1] - 2.0*qx*v[2] + qz*v[0],  qw*v[0] - 2.0*qy*v[2] + qz*v[1],  qx*v[0] + qy*v[1]]])
        H_2 = np.array([[-qy*v[5] + qz*v[4],                qy*v[4] + qz*v[5], -qw*v[5] + qx*v[4] - 2.0*qy*v[3],  qw*v[4] + qx*v[5] - 2.0*qz*v[3]],
                        [ qx*v[5] - qz*v[3],  qw*v[5] - 2.0*qx*v[4] + qy*v[3],                qx*v[3] + qz*v[5], -qw*v[3] + qy*v[5] - 2.0*qz*v[4]],
                        [-qx*v[4] + qy*v[3], -qw*v[4] - 2.0*qx*v[5] + qz*v[3],  qw*v[3] - 2.0*qy*v[5] + qz*v[4],  qx*v[3] + qy*v[4]]])
        H = np.vstack((H, H_2))
        return 2.0*H
        
    def get_normalized_measurements(self):
        """
        Pulls measurements from IMU. 
        Normalizes acceleration and magnetometer readings for measurement model.
        """
        
        self.Ax, self.Ay, self.Az = self.imu.get_acc()
        self.p, self.q, self.r = imu.get_gyr()
        self.mx, self.my, self.mz = imu.get_mag()
        
        # Normalized Measurements
        self.a = np.array([[self.Ax, self.Ay, self.Az]]).T/np.linalg.norm(np.array([[self.Ax, self.Ay, self.Az]]))
        self.m = np.array([[self.mx, self.my, self.mz]]).T/np.linalg.norm(np.array([[self.mx, self.my, self.mz]]))
    
    def predict(self):
        Om = Omega(self.p, self.q, self.r)
        self.x_hat = f(self.x, Om, self.dt) # estimate based on first-order KF
        
        F = self.calc_F() 
        W = self.calc_W()
        
        Q = (self.Qc**2)*W@W.T
        
        # Calculate Predicted Error State Covariance
        self.Phat = F@self.P@F.T + Q
        
    def update(self):
        C = self.calc_C()
        
        z = np.array([[self.Ax, self.Ay, self.Az, self.mx, self.my, self.mz]]).T
        ahat = C.T@g
        mhat = C.T@r_m

        h = np.vstack((ahat, mhat))# measurement model
        H = self.calc_H()[:,:,0] # hacky, should fix

        v = z-h
        S = H@self.Phat@H.T+self.R
        K = self.Phat@H.T@np.linalg.inv(S)

        self.x = self.x_hat + K@v
        self.P = (np.identity(4)-K@H)@self.Phat

        # renormalize
        self.x = self.x / (np.linalg.norm(self.x))