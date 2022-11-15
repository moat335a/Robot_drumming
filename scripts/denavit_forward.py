from robot.Forward import Forward_kinematics
from typing import List
import tensorflow as tf
import numpy as np

class Denavit_forward(Forward_kinematics):
    def __init__(self, a_n: List, alpha_n: List, d_n: List, n_dofs: int) -> None:
        super().__init__(a_n, alpha_n, d_n, n_dofs)
    
    def compute_forward_dynamics(self,theta_n):
        t=0
        for dof in range(len(self.a_n)):
            
            
            if dof==len(theta_n):
                angel = 5
            else:
                angel = theta_n[dof]
            if t>0:
                forward_tensor =forward_tensor@ np.array([[np.cos(angel), -np.sin(angel)*np.cos(self.alpha_n[dof]),
                    np.sin(angel)*np.sin(self.alpha_n[dof]),self.a_n[dof]*np.cos(angel)],
                    [np.sin(angel), np.cos(angel)*np.cos(self.alpha_n[dof]), 
                    -np.cos(angel)*np.sin(self.alpha_n[dof]), self.a_n[dof]*np.sin(angel)],
                    [0,np.sin(self.alpha_n[dof]),np.cos(self.alpha_n[dof]),self.d_n[dof]],
                    [0,0,0,1]])
            else:
                forward_tensor = np.array([[np.cos(angel), -np.sin(angel)*np.cos(self.alpha_n[dof]),
                    np.sin(angel)*np.sin(self.alpha_n[dof]),self.a_n[dof]*np.cos(angel)],
                    [np.sin(angel), np.cos(angel)*np.cos(self.alpha_n[dof]), 
                    -np.cos(angel)*np.sin(self.alpha_n[dof]), self.a_n[dof]*np.sin(angel)],
                    [0,np.sin(self.alpha_n[dof]),np.cos(self.alpha_n[dof]),self.d_n[dof]],
                    [0,0,0,1]])

            t+=1

        return forward_tensor

