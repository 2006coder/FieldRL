import rtde_control
import rtde_receive
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation
import numpy as np
rtde_c = rtde_control.RTDEControlInterface("10.0.0.78")
rtde_r = rtde_receive.RTDEReceiveInterface("10.0.0.78")

print(Rotation.from_rotvec(rtde_r.getActualTCPPose()[3:]).as_matrix() )

print((180.0/np.pi)*np.array(rtde_r.getActualQ()))