import numpy as np
import matplotlib.pyplot as plt
from time import time

class make_hit:
    def __init__(self,weights,t_steps
    ,Cs = np.array([i/15 for i in range(15)]),
                   covars=np.array([0.004567567567567567 for _ in range(15)])) -> None:
        weights = np.load(weights)
        self.means = weights["means"]
        self.covariances = weights["covars"]
        self.t_steps = t_steps
        self.Cs = Cs
        self.covars = covars

    def test(self,t_steps,save=False,plot=False,num_samples=1):
        if num_samples ==0:
            return None
        sampled_weights=np.random.default_rng().multivariate_normal(self.means,self.covariance)
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

    
        