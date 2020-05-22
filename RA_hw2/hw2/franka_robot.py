import math
import numpy as np
import xml.etree.ElementTree as ET
from pyquaternion import Quaternion

def parse_urdf(urdf_file_path):
    '''
    TODO(Q3.1.1)

    Implement a urdf file parser and extract the origins and axes for each joint into numpy arrays.
    Arguments: string
    Returns: A dictionary with numpy arrays that contain the origins and axes of each joint
    '''

    num_joints = 8 # Change this to the number of times you encounter the word "axis" in the document
    origin = np.zeros((num_joints+1, 6))
    axis = np.zeros((num_joints+1, 3))

    # Use xml.etree.ElementTree to parse the urdf files
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()
    for i, ori in enumerate (root.iter('origin')):
        if i == 0 : continue
        rpy = []
        for k in ori.get('rpy').split(): #split strings
                if k[0] == '$':
                    if k[2] == 'p': rpy.append(math.pi / 2)
                    else : rpy.append(- math.pi/2)
                else:
                    rpy.append(float(k))
        origin[i-1,3:6] = rpy

    for i, ori in enumerate (root.iter('origin')):
        if i == 0: continue
        xyz = []
        for k in ori.get('xyz').split():
                xyz.append(float(k))
        origin[i-1,0:3] = xyz

    for i, ax in enumerate (root.iter('axis')):
        xyz = []
        for k in ax.get('xyz').split():
                xyz.append(float(k))
        axis[i,0:3] = xyz


    # Since the end-effector transformation is not included in the franka urdf, I will manually provide
    # the transformation here for you from the flange frame.
    origin[-1,2] = 0.1034


    return {'origin': origin, 'axis': axis}

class FrankaRobot():

    def __init__(self, urdf_file_path, dh_params, num_dof):

        self.robot_params = parse_urdf(urdf_file_path)
        self.dh_params = dh_params
        self.num_dof = num_dof

    def forward_kinematics_urdf(self, joints):
        '''
        TODO(Q3.1.2)
        
        Calculate the position of each joint using the robot_params
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the 4x4 transformation matrices from the base to the position of each joint.
        '''
        #joints is size 7, therefore append two 0 at the end
        Joints = joints[:]
        Joints.append(0)
        Joints.append(0)
        # origin_location = np.zeros(3)
        ori = self.robot_params['origin']

        forward_kinematics = np.zeros((self.robot_params['axis'].shape[0],4,4))
        for i in range(len(Joints)):
            #rotational matrix for row pitch yaw
            R_x = np.zeros((4,4))
            R_x[0,:] = [1,0,0,0]
            R_x[1,:] = [0,math.cos(ori[i][3]),-math.sin(ori[i][3]),0]
            R_x[2,:] = [0,math.sin(ori[i][3]),math.cos(ori[i][3]),0]
            R_x[3,:] = [0,0,0,1]
            R_y = np.zeros((4,4))
            R_y[0,:] = [math.cos(ori[i][4]),0,math.sin(ori[i][4]),0]
            R_y[1,:] = [0,1,0,0]
            R_y[2,:] = [-math.sin(ori[i][4]),0,math.cos(ori[i][4]),0]
            R_y[3,:] = [0,0,0,1]
            R_z = np.zeros((4,4))
            R_z[0,:] = [math.cos(ori[i][5]+Joints[i]),-math.sin(ori[i][5]+Joints[i]),0,0]
            R_z[1,:] = [math.sin(ori[i][5]+Joints[i]),math.cos(ori[i][5]+Joints[i]),0,0]
            R_z[2,:] = [0,0,1,0]
            R_z[3,:] = [0,0,0,1]

            rotation = np.matmul(R_x,np.matmul(R_y,R_z))

            Trans = np.zeros((4,4))
            Trans[0,:] = [1,0,0,ori[i][0]]
            Trans[1,:] = [0,1,0,ori[i][1]]
            Trans[2,:] = [0,0,1,ori[i][2]]
            Trans[3,:] = [0,0,0,1]

            tmp = np.matmul(Trans,rotation)
            if i == 0:
                forward_kinematics[i,:,:] = tmp
            else:
                forward_kinematics[i,:,:] = np.matmul(forward_kinematics[i-1,:,:],tmp)

        return forward_kinematics

    def forward_kinematics_dh(self, joints):
        '''
        TODO(Q3.2.1)
        
        Calculate the position of each joint using the dh_params
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the 4x4 transformation matrices from the base to the position of each joint.
        '''
        Joints = joints[:]
        forward_kinematics = np.zeros((self.dh_params.shape[0],4,4))
        n = self.dh_params.shape[0]

        for i in range(n):

            params = self.dh_params[i]
            a = params[0]
            d = params[1]
            alpha = params[2]
            if i < 7 :
                theta = Joints[i]
            else :
                theta = 0
            #rotational matrix for row pitch yaw
            R_x = np.zeros((4,4))
            R_x[0,:] = [1,0,0,0]
            R_x[1,:] = [0,math.cos(alpha),-math.sin(alpha),0]
            R_x[2,:] = [0,math.sin(alpha),math.cos(alpha),0]
            R_x[3,:] = [0,0,0,1]

            T_x = np.zeros((4,4))
            T_x[0,:] = [1,0,0,a]
            T_x[1,:] = [0,1,0,0]
            T_x[2,:] = [0,0,1,0]
            T_x[3,:] = [0,0,0,1]

            T_z = np.zeros((4,4))
            T_z[0,:] = [1,0,0,0]
            T_z[1,:] = [0,1,0,0]
            T_z[2,:] = [0,0,1,d]
            T_z[3,:] = [0,0,0,1]

            R_z = np.zeros((4,4))
            R_z[0,:] = [math.cos(theta),-math.sin(theta),0,0]
            R_z[1,:] = [math.sin(theta),math.cos(theta),0,0]
            R_z[2,:] = [0,0,1,0]
            R_z[3,:] = [0,0,0,1]

            tmp = np.matmul(R_x,np.matmul(T_x,np.matmul(T_z,R_z)))
            if i == 0:
                forward_kinematics[i,:,:] = tmp
            else:
                forward_kinematics[i,:,:] = np.matmul(forward_kinematics[i-1,:,:],tmp)
        return forward_kinematics

    def ee(self, joints):
        '''
        TODO(Q3.2.2)
        
        Use one of your forward kinematics implementations to return the position of the end-effector.
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the [x, y, z, roll, pitch, yaw] location of the end-effector.
        '''

        forward_kinematics = self.forward_kinematics_dh(joints)[-1]
        x = forward_kinematics[0][3]
        y = forward_kinematics[1][3]
        z = forward_kinematics[2][3]

        R = forward_kinematics[:3,:3]

        def isclose(x, y, rtol=1.e-5, atol=1.e-8):
            return abs(x - y) <= atol + rtol * abs(y)

        if isclose(R[2,0],-1.0):
            pitch = math.pi/2.0
            roll = math.atan2(R[0,1],R[0,2]) if R[0,2] >= 0 else (math.pi/2)
        elif isclose(R[2,0],1.0):
            pitch = -math.pi/2.0
            roll = math.atan2(-R[0,1],-R[0,2]) if -R[0,2] >= 0 else (math.pi/2)
        else:
            pitch = -math.asin(R[2,0])
            cos_theta = math.cos(pitch)
            roll = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta) if R[2,2]/cos_theta >= 0 else (math.pi/2)
            yaw = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta) if R[0,0]/cos_theta >= 0 else (math.pi/2)

        return np.array([x,y,z,roll,pitch,yaw])

    def jacobian(self, joints):
        '''
        TODO(Q4.1.1)
        
        Calculate the end-effector jacobian analytically using your forward kinematics
        Arguments: array of joint positions (rad)
        Returns: A numpy array that contains the 6 x num_dof end-effector jacobian.
        '''

        end_effector = self.forward_kinematics_dh(joints)[-1,0:3,3]
        forward_kinematics = self.forward_kinematics_dh(joints) [0:7]
        jacobian = np.zeros((6,self.num_dof))
        jacobian[3:6,:] = forward_kinematics[:,0:3,2].T
        jacobian[0:3,:] = np.cross(forward_kinematics[:,0:3,2],end_effector-forward_kinematics[:,0:3,3]).T
        return jacobian

    def inverse_kinematics(self, desired_ee_pos, current_joints):
        '''
        TODO(Q5.1.1)
        
        Implement inverse kinematics using one of the methods mentioned in class. 
        Arguments: desired_ee_pos which is a np array of [x, y, z, r, p, y] which represent the desired end-effector position of the robot
                   current_joints which represents the current location of the robot
        Returns: A numpy array that contains the joints required in order to achieve the desired end-effector position.
        '''
        if len(current_joints) == 9:
            current_joints = current_joints [:-2]

        alpha = 0.1
        error_x = np.linalg.norm(desired_ee_pos-self.ee(current_joints))
        # print(error_x)
        while error_x > 1e-8: # change to 1e-4 for running panda in ROS
            error_x = np.linalg.norm(desired_ee_pos-self.ee(current_joints))
            delta_x = desired_ee_pos-self.ee(current_joints)
            J_t = self.jacobian(current_joints).T
            delta_q = alpha * np.matmul(J_t ,delta_x)
            current_joints = current_joints + delta_q

        joints = current_joints
        return joints

    def check_box_collision(self, joints, box):
        '''
        TODO(Q6.1.1)
        
        Implement collision checking with a box.
        Arguments: joints represents the current location of the robot
                   box contains the position of the center of the box [x, y, z, r, p, y] and the length, width, and height [l, w, h]
        Returns: A boolean where True means the box is in collision with the arm and false means that there are no collisions.
        '''

        # in_collision = False

        BoxesConfig = [
               {"Link":1, "Refer":1, "Translation":np.array([-0.04,0,-0.283]),          "Rotation":np.array([1,0,0,0]),            "Size": np.array([0.23,0.2,0.1])},
               {"Link":2, "Refer":1, "Translation":np.array([-0.009,0,-0.183]),         "Rotation":np.array([1,0,0,0]),            "Size": np.array([0.13,0.12,0.1])},
               {"Link":3, "Refer":1, "Translation":np.array([0,-0.032,-0.082]),        "Rotation":np.array([0.9514,0.3079,0,0]),  "Size": np.array([0.12,0.1,0.2])},
               {"Link":4, "Refer":1, "Translation":np.array([-0.008,0,0]),              "Rotation":np.array([1,0,0,0]),            "Size": np.array([0.15,0.27,0.11])},
               {"Link":5, "Refer":1, "Translation":np.array([0,0.042,0.067]),        "Rotation":np.array([0.9514,0.3079,0,0]),  "Size": np.array([0.12,0.1,0.2])},
               {"Link":6, "Refer":3, "Translation":np.array([0.00687,0,-0.139]),        "Rotation":np.array([1,0,0,0]),            "Size": np.array([0.13,0.12,0.25])},
               {"Link":7, "Refer":4, "Translation":np.array([-0.008,0.004,0]),          "Rotation":np.array([0.7071,-0.7071,0,0]), "Size": np.array([0.13,0.23,0.15])},
               {"Link":8, "Refer":5, "Translation":np.array([0.00422,0.05367,-0.121]),  "Rotation":np.array([0.9962,-0.08715,0,0]),"Size": np.array([0.12,0.12,0.4])},
               {"Link":9, "Refer":5, "Translation":np.array([0.00422,0.00367,-0.263]), "Rotation":np.array([1,0,0,0]),            "Size": np.array([0.12,0.12,0.25])},
               {"Link":10,"Refer":5, "Translation":np.array([0.00328,0.0176,-0.0055]), "Rotation":np.array([1,0,0,0]),            "Size": np.array([0.13,0.23,0.12])},
               {"Link":11,"Refer":7, "Translation":np.array([-0.00136,0.0092,0.0083]), "Rotation":np.array([0,1,0,0]),            "Size": np.array([0.12,0.12,0.2])},
               {"Link":12,"Refer":7, "Translation":np.array([-0.00136,0.0092,0.1407]),  "Rotation":np.array([0.9239,0,0,-0.3827]), "Size": np.array([0.08,0.22,0.17])}]

        frame_transform_matrix= self.forward_kinematics_dh(joints) # frame transformation matrix respect to base at given joints
        frame2box_matrix = np.zeros((12,4,4))

        for i in range( frame2box_matrix.shape[0]):
            orientation = BoxesConfig[i]["Rotation"]
            quaternion = Quaternion(w=orientation[0], x=orientation[1], y=orientation[2], z=orientation[3])
            r = quaternion.rotation_matrix
            frame2box_matrix[i,:3,:3] = r
            frame2box_matrix[i,:3,3] = BoxesConfig[i]["Translation"]
            frame2box_matrix[i,3,3] = 1

        base_frame2box = np.zeros((12,4,4))
        base_frame2box[0,:,:] = np.matmul(frame_transform_matrix[0,:,:],frame2box_matrix[0])
        base_frame2box[1,:,:] = np.matmul(frame_transform_matrix[0,:,:],frame2box_matrix[1])
        base_frame2box[2,:,:] = np.matmul(frame_transform_matrix[0,:,:],frame2box_matrix[2])
        base_frame2box[3,:,:] = np.matmul(frame_transform_matrix[0,:,:],frame2box_matrix[3])
        base_frame2box[4,:,:] = np.matmul(frame_transform_matrix[0,:,:],frame2box_matrix[4])
        base_frame2box[5,:,:] = np.matmul(frame_transform_matrix[2,:,:],frame2box_matrix[5])
        base_frame2box[6,:,:] = np.matmul(frame_transform_matrix[3,:,:],frame2box_matrix[6])
        base_frame2box[7,:,:] = np.matmul(frame_transform_matrix[4,:,:],frame2box_matrix[7])
        base_frame2box[8,:,:] = np.matmul(frame_transform_matrix[4,:,:],frame2box_matrix[8])
        base_frame2box[9,:,:] = np.matmul(frame_transform_matrix[4,:,:],frame2box_matrix[9])
        base_frame2box[10,:,:] = np.matmul(frame_transform_matrix[6,:,:],frame2box_matrix[10])
        base_frame2box[11,:,:] = np.matmul(frame_transform_matrix[6,:,:],frame2box_matrix[11])


        # define moving box xyz and directions
        roll = box[3]
        pitch = box[4]
        yall = box[5]
        R_x = np.zeros((4,4))
        R_x[0,:] = [1,0,0,0]
        R_x[1,:] = [0,math.cos(roll),-math.sin(roll),0]
        R_x[2,:] = [0,math.sin(roll),math.cos(roll),0]
        R_x[3,:] = [0,0,0,1]

        R_y = np.zeros((4,4))
        R_y[0,:] = [math.cos(pitch),0,math.sin(pitch),0]
        R_y[1,:] = [0,1,0,0]
        R_y[2,:] = [-math.sin(pitch),0,math.cos(pitch),0]
        R_y[3,:] = [0,0,0,1]

        R_z = np.zeros((4,4))
        R_z[0,:] = [math.cos(yall),-math.sin(yall),0,0]
        R_z[1,:] = [math.sin(yall),math.cos(yall),0,0]
        R_z[2,:] = [0,0,1,0]
        R_z[3,:] = [0,0,0,1]

        box_matrix = np.matmul(R_z,np.matmul(R_y,R_x))
        collision_check = np.ones(12)

        for idx in range(12):
            A = base_frame2box[idx]
            L = self.L_function(A[:3,:3],box_matrix[:3,:3])
            Pa = A[0:3,3]
            Ax = A[0:3,0]
            Ay = A[0:3,1]
            Az = A[0:3,2]
            Wa = 0.5 * BoxesConfig[idx]["Size"][1]
            Ha = 0.5 * BoxesConfig[idx]["Size"][2]
            Da = 0.5 * BoxesConfig[idx]["Size"][0]

            Pb = box[0:3]
            Bx = box_matrix[0:3,0]
            By = box_matrix[0:3,1]
            Bz = box_matrix[0:3,2]
            Wb = 0.5 * box[7]
            Hb = 0.5 * box[8]
            Db = 0.5 * box[6]

            T = Pb-Pa
            for i in range(15):
                if (np.abs(np.dot(T,L[i])) > (np.abs(np.dot(Wa*Ax,L[i])) + np.abs(np.dot(Ha*Ay,L[i])) + np.abs(np.dot(Da*Az,L[i])) + np.abs(np.dot(Wb*Bx,L[i])) + np.abs(np.dot(Hb*By,L[i])) + np.abs(np.dot(Db*Bz,L[i])))):
                    collision_check[idx] = 0
                    break

        if collision_check.sum() > 0 : return True

        return False

    def L_function (self,A,B):
        #A and B are matrices
        L= np.zeros((15,3))
        L[0,:] = A[:,0]
        L[1,:] = A[:,1]
        L[2,:] = A[:,2]
        L[3,:] = B[:,0]
        L[4,:] = B[:,1]
        L[5,:] = B[:,2]
        L[6,:] = np.cross(A[:,0],B[:,0])
        L[7,:] = np.cross(A[:,0],B[:,1])
        L[8,:] = np.cross(A[:,0],B[:,2])
        L[9,:] = np.cross(A[:,1],B[:,0])
        L[10,:] = np.cross(A[:,1],B[:,1])
        L[11,:] = np.cross(A[:,1],B[:,2])
        L[12,:] = np.cross(A[:,2],B[:,0])
        L[13,:] = np.cross(A[:,2],B[:,1])
        L[14,:] = np.cross(A[:,2],B[:,2])

        return L

