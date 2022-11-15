
from typing import List
import tensorflow as tf
import numpy as np


class Forward_kinematics:
    def __init__(self,a_n:List,alpha_n:List,d_n:List,n_dofs:int) -> None:
        """
        gets kinematic params as list
        """
        self.alpha_n = tf.constant(alpha_n)
        self.d_n = tf.constant(d_n)
        self.a_n = tf.constant(a_n)
        self.n_dofs = n_dofs
        # tensor for transforamation to match urdf from kai
        self.alltensors = self.get_rot_tensor()
        self.translation_matrix= tf.convert_to_tensor([[0.,0.,0.346],[0.,0.,0.],[0.,0.,0.],[0.045, -0.550, 0.000]])


    def compute_forward_dynamics(self,theta_n):
        forward_tensor = tf.TensorArray(tf.float32,size=self.n_dofs)
        t=0
        for dof in range(self.n_dofs):

            forward_tensor =forward_tensor.write(dof,self.alltensors[t]@ np.array([[tf.math.cos(theta_n[dof]), -tf.math.sin(theta_n[dof])*tf.math.cos(self.alpha_n[dof]),
                tf.math.sin(theta_n[dof])*tf.math.sin(self.alpha_n[dof]),self.translation_matrix[dof][0]],
                [tf.math.sin(theta_n[dof]), tf.math.cos(theta_n[dof])*tf.math.cos(self.alpha_n[dof]), 
                -tf.math.cos(theta_n[dof])*tf.math.sin(self.alpha_n[dof]), self.translation_matrix[dof][1]],
                [0,tf.math.sin(self.alpha_n[dof]),tf.math.cos(self.alpha_n[dof]),self.translation_matrix[dof][2]],
                [0,0,0,1]])@ self.alltensors[t+1])
            t+=2

        return self.apply_mult(data=forward_tensor.stack())

    def forward(self,joint_trajectory):
        return self.compute_forward_dynamics(theta_n=joint_trajectory)
    
    def apply_mult(self,data):
        i =0
        while i< data.shape[0]:
            if i ==0 : 
                res = tf.linalg.matmul(data[0],data[1])
                i+=2
                continue
            res = tf.linalg.matmul(res,data[i])
            i+=1 
        return res

    def get_rot_tensor(self):
        """applies the changes in forward kinematiks tensor
        so that it matches the one from urdf from kai simulation in ROS
        """
        fixed_rot1 =tf.expand_dims( tf.convert_to_tensor([[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=tf.float32),axis=0)
        fixed_rot_after1 = tf.expand_dims(tf.convert_to_tensor([[0., 0., 1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]], dtype=tf.float32),axis=0)
        fixed_rot2 = tf.expand_dims(tf.convert_to_tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=tf.float32),axis=0)
        fixed_rot_after2 = tf.expand_dims(tf.convert_to_tensor([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]], dtype=tf.float32),axis=0)
        fixed_rot3 = tf.expand_dims(tf.convert_to_tensor([[1., 0., 0., 0.], [0, 0., -1., 0.], [0.,1., 0., 0.], [0., 0., 0., 1.]], dtype=tf.float32),axis=0)
        fixed_rot_after3 = tf.expand_dims(tf.convert_to_tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=tf.float32),axis=0)
        fixed_rot4 = tf.expand_dims(tf.convert_to_tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=tf.float32),axis=0)
        fixed_rot_after4 = tf.expand_dims(tf.convert_to_tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=tf.float32),axis=0)
        return tf.concat([fixed_rot1,fixed_rot_after1,fixed_rot2,fixed_rot_after2,fixed_rot3,fixed_rot_after3,fixed_rot4,fixed_rot_after4],axis=0)







        



        