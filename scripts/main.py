from get_reward import Reward
from hits import Get_hits
import numpy as np
from time import time


if __name__ == "__main__":
    # n = 30
    # T = 300
    # run = 8
    # data=Get_hits(f"/home/atashak/taxi/robot/kdata/krun{run}")
    # rew = Reward(n_time_steps=T,centers = [i/n for i in range(n)],
    #                 covars=[0.0006 for _ in range(n)])
    n = 15
    T = 300

    # for run in range(6):
    #     run+=1
    data=Get_hits(f"/home/atashak/taxi/robot/2promps/promp2")
    rew = Reward(n_time_steps=T,centers = [i/n for i in range(n)],
                    covars=[0.004567567567567567 for _ in range(n)])
    rew.fit_ridge_regression(data.hits)
    np.savez(f"/home/atashak/taxi/robot/weights2promps/weights2.npz",means=rew.means,covars=rew.covariance)
#     sampled_weights=np.random.default_rng().multivariate_normal(rew.means,rew.covariance)
#     rew.n_time_steps = 3000
#     phis = rew.build_phi_matrix()
#     sampled_trajectory = np.zeros((4,T*10))
#     t=0
#     for i in range(4):
#         sampled_trajectory[i] = phis@sampled_weights[t:t+30]
#         t +=30
# #   np.save(f"/home/atashak/sample_trajectory{time()}.npy",sampled_trajectory)
    print("finshed")
    # run=7
    # data=Get_hits(f"/home/atashak/taxi/robot/kdata/krun{run}")
    # rew = Reward(n_time_steps=T,centers = [i/n for i in range(n)],
    #                 covars=[0.004567567567567567 for _ in range(n)])
    # rew.fit_ridge_regression(data.hits)
    # rew.test(1000,plot=False,save=True,num_samples=2)

