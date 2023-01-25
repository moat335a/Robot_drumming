import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List
from time import time

class ProMP:
    def __init__(self,n_discretizations:int=50,centers:List = [i/15 for i in range(15)],
                covars:List=[0.004567567567567567 for _ in range(15)]) -> None:
        self.Cs = tf.constant(centers)
        self.num_rbfs = len(centers)
        self.n_discretizations = n_discretizations
        self.covar = tf.constant(covars)
        self.Phi_matrix = self.build_phi_matrix()

    def get_phi(self,phase):
         a = -np.square(phase-self.Cs)/(2*self.covar)
         return np.exp(a) / np.sum(np.exp(a))

    def get_Phi_for_timesteps(self, time_steps,maxi):
        phases = time_steps / maxi
        #phases = time_steps/
        phis=np.zeros((len(phases), self.Cs.shape[0]))
        t=0
        for phase in phases:
            phi = self.get_phi(phase=phase)
            phis[t]=phi.tolist()
            t+=1
        return phis

    def build_phi_matrix(self,t_steps=None):
        if t_steps is None:
            t_steps= self.n_discretizations

        phis = tf.TensorArray(dtype=tf.float32, size=t_steps)
        t=0
        for phase in tf.range(0,1,1/t_steps):
            phis = phis.write(t, self.get_phi(phase=phase))
            t+=1
        return phis.stack()


    def fit_ridge_regression(self,y,k=1e-12,**kwargs):
        weights = np.zeros((y.shape[0],self.Cs.shape[0]*4))
        for i,Y in enumerate(y):
            weights[i] =self.fit_ridge_regression_y(Y,k,**kwargs).flatten()
        self.get_mean_cov(weights=weights)
        return weights
        
    def get_mean_cov(self,weights):
        self.means = np.mean(weights,0)
        self.covariance = np.cov(weights,rowvar=False)


    def fit_ridge_regression_y(self,Y,k=1e-12,process_data=True):
        if process_data:
            Y = self.process_y(Y)
        weights = np.zeros((4,self.Cs.shape[0]))
        for i in range(4):
            phiphi = np.matmul(self.Phi_matrix.T,self.Phi_matrix) + k*np.identity(self.Cs.shape[0])
            weights[i] =np.matmul(np.matmul(np.linalg.inv(phiphi),self.Phi_matrix.T),Y[i].T)
        return weights

    def process_y(self,Y):
        y = np.zeros((4,self.n_discretizations))
        for i in range(4):
            strike = Y[:,i]
            y[i]=np.interp(np.arange(self.n_discretizations), np.linspace(0, self.n_discretizations, len(strike)),strike)
        return y

    def get_trajectories(self, weights):
        sampled_trajectory = np.zeros((4,self.n_discretizations))
        t=0
        for i in range(4):
            sampled_trajectory[i] = self.Phi_matrix @ weights[t:t+self.num_rbfs]
            t += self.num_rbfs

    @staticmethod
    def get_trajectories_for_phi(weights, phi_matrix):
        num_rbfs = tf.shape(phi_matrix)[1]
        sampled_trajectory = tf.TensorArray(size=4, dtype=tf.float32)
        t=0
        for i in range(4):
            sampled_trajectory = sampled_trajectory.write(i, (phi_matrix @ tf.expand_dims(weights[t:t+num_rbfs], 1))[:,0])
            t += num_rbfs
        return sampled_trajectory.concat()
  
    def plot(trajectory):
            fig,axs=plt.subplots(2,2,sharex=True)
            axs[0,0].plot(trajectory[0])
            axs[0,0].set_title("joint 0")
            axs[0,1].plot(trajectory[1])
            axs[0,1].set_title("joint 1")
            axs[1,0].plot(trajectory[2])
            axs[1,0].set_title("joint 2")
            axs[1,1].plot(trajectory[3])
            axs[1,1].set_title("joint 3")
            plt.show()

    def test(self,t_steps=None,save=False,plot=False,num_samples=1,weights=None):
        if num_samples ==0:
            return None
        if weights is None:
            sampled_weights=np.random.default_rng().multivariate_normal(self.means,self.covariance)
        else:
            sampled_weights = weights
        if t_steps is None:
            phis = self.Phi_matrix
            t_steps = self.n_discretizations
            
        else:
            phis = self.build_phi_matrix(t_steps=t_steps)
        sampled_trajectory = np.zeros((4,t_steps))
        t=0
        for i in range(4):
            sampled_trajectory[i] = phis@sampled_weights[t:t+self.num_rbfs]
            t += self.num_rbfs
        if plot:
            self.plot(sampled_trajectory)
        if save:
            np.save(f"/home/atashak/taxi/robot/trajectories/sample_trajectory{time()}.npy",sampled_trajectory)
            self.test(t_steps=t_steps,save=save,plot=plot,num_samples=num_samples-1)
        return sampled_trajectory


 
