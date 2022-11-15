import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List
from time import time

class Reward:
    def __init__(self,n_time_steps:int=50,centers:List = [i/15 for i in range(15)],
                covars:List=[0.004567567567567567 for _ in range(15)]) -> None:
        self.Cs = tf.constant(centers)
        self.n_time_steps = n_time_steps
        self.covar = tf.constant(covars)
        self.Phi_matrix = self.build_phi_matrix()

    def get_phi(self,phase):
         a = -np.square(phase-self.Cs)/(2*self.covar)
         return np.exp(a) / np.sum(np.exp(a))

    def build_phi_matrix(self,t_steps=None):
        if t_steps is None:
            t_steps= self.n_time_steps

        phis=np.zeros((t_steps,self.Cs.shape[0]))
        t=0
        for phase in np.arange(0,1,1/t_steps):
            phi = self.get_phi(phase=phase)
            phis[t]=phi.tolist()
            t+=1
        return phis
                                                     
    #@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32)))
    def calculate_y(self,weights):
        """calculates y for 4 dofs
        weights : weights of proMP with shape (n_dofs,12)
        """
        joint_trajectories = tf.TensorArray(tf.float64,size=self.n_time_steps)
        t= tf.Variable(initial_value=0)
        for i,phase in enumerate( tf.range(0,1,1/self.n_time_steps)):
            joint_trajectory = tf.matmul(tf.reshape( self.get_phi(phase),[1,-1]), weights)
            joint_trajectories = joint_trajectories.write(i,tf.reshape(joint_trajectory,[-1]).numpy())
            t.assign(t+1)
        
        return joint_trajectories.stack()

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
            phiy = np.matmul(self.Phi_matrix.T,Y[i])
            weights[i] =np.matmul(np.linalg.inv(phiphi),phiy)
        return weights

    def process_y(self,Y):
        y = np.zeros((4,self.n_time_steps))
        for i in range(4):
            strike = Y[:,i]
            y[i]=np.interp(np.arange(self.n_time_steps), np.linspace(0, self.n_time_steps, len(strike)),strike)
        return y


    def publich_joint_trajector(self,weights,t_step:float):
        """function for computing joint trajectory to publich for ROS
        
        t_step :  current time step
        """
        Phi = self.get_phi(t_step).numpy()
        joint_trajectory = tf.matmul(tf.reshape(Phi ,[1,-1]), weights)
        return joint_trajectory

    def test(self,t_steps=None,save=False,plot=False,num_samples=1,weights=None):
        if num_samples ==0:
            return None
        if weights is None:
            sampled_weights=np.random.default_rng().multivariate_normal(self.means,self.covariance)
        else:
            sampled_weights = weights
        if t_steps is None:
            phis = self.Phi_matrix
            t_steps = self.n_time_steps
            
        else:
            phis = self.build_phi_matrix(t_steps=t_steps)
        sampled_trajectory = np.zeros((4,t_steps))
        t=0
        for i in range(4):
            sampled_trajectory[i] = phis@sampled_weights[t:t+phis.shape[1]]
            t +=phis.shape[1]
        if plot:
            fig,axs=plt.subplots(2,2,sharex=True)
            axs[0,0].plot(sampled_trajectory[0])
            axs[0,0].set_title("joint 0")
            axs[0,1].plot(sampled_trajectory[1])
            axs[0,1].set_title("joint 1")
            axs[1,0].plot(sampled_trajectory[2])
            axs[1,0].set_title("joint 2")
            axs[1,1].plot(sampled_trajectory[3])
            axs[1,1].set_title("joint 3")
            plt.show()
        if save:
            np.save(f"/home/atashak/taxi/robot/trajectories/sample_trajectory{time()}.npy",sampled_trajectory)
            self.test(t_steps=t_steps,save=save,plot=plot,num_samples=num_samples-1)
        return sampled_trajectory


        

# if __name__ == "__main__":
#     n = 18
#     T = 300
#     wegihts = np.random.random((n,4))
#     rew = Reward(n_time_steps=T,centers = [i/n for i in range(n)],
#                  covars=[0.0085 for _ in range(n)])
#    # rew.calculate_y(weghts)
#     starts = np.load("/home/atashak/taxi/robot/data_process/data_collecion1/starts.npy")
#     times = np.load("/home/atashak/taxi/robot/data_process/data_collecion1/experiment_1/times.npy")
#     ends = np.load("/home/atashak/taxi/robot/data_process/data_collecion1/end.npy")
#     joints = np.load("/home/atashak/taxi/robot/data_process/data_collecion1/experiment_1/joint_positions.npy")
#     joints=joints[~np.isnan(times)]
#     times=times[~np.isnan(times)]
#     index=np.argmin(np.abs(times-starts[0]))
#     stroke=joints[105852:106272] #joints[index+2:index+8,0]
#     ys = np.zeros((4,T))
#     for i in range(4):
#         b=stroke[:,i]
#         ys[i]=np.interp(np.arange(T), np.linspace(0, T, len(b)),b)
#     weights = rew.fit_ridge_regression(ys[3])
#     plt.plot(ys[3])
#     plt.plot(rew.calculate_y(np.expand_dims( weights,1)))
#     # for i in range(rew.Phi_matrix.shape[0]):
#     #     plt.plot(np.squeeze(rew.Phi_matrix[i]))

#     plt.show()


#     print("finished")