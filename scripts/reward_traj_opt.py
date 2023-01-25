
from new_promp import ProMP
import tensorflow as tf
import numpy as np
import scipy
from math import pi
from time import time
import matplotlib.pyplot as plt
import scipy.optimize as sopt



class Reward:
    def __init__(self, strike_promps, strike_promp_means,strike_promp_chols,  long_promp,promp_duration=0.6) -> None:
        self.strike_promps = strike_promps # should be a list of list of Promps (one list per drum)
        self.strike_promp_means = strike_promp_means
        #self.strike_promp_covars = strike_promp_covars
        self.strike_promp_chols = strike_promp_chols
        self.long_promp = long_promp
        self.promp_duration = promp_duration

 
    def get_time_indexes(self,t_hits):
        """gets the corresponding time indexes from trajectory
        t_hits: desired hit times"""
        self.times_steps = []
        total =(self.long_promp.n_discretizations*.004 - 0.001)
        for t,t_hit in enumerate(t_hits):
    
            time_steps_this_drum = tf.linspace(t_hit - (self.promp_duration/2) , t_hit + (self.promp_duration/2), 
            len(self.strike_promp_means[t][0])//4) # len(strike_promp_means[0][0])
            t_stpes_this =np.concatenate([time_steps_this_drum,time_steps_this_drum+total/4
            ,time_steps_this_drum+(total)/2,time_steps_this_drum+(3*total)/4])
            self.times_steps.append(np.round(np.array(t_stpes_this)*1000).astype(int))

        #self.times_steps =np.round(np.array(self.times_steps)*1000).astype(int)


    def set_hit_times(self, t_hits, drum_indices):
        """t_hits: desired hit times"""
        total =self.long_promp.n_discretizations*.004
        assert(len(drum_indices) == len(t_hits)) # for each hit_time we specify the index of the drum
        self.Phi_per_segment = []
        for t_hit, drum_index in zip(t_hits, drum_indices):
            phi_segments_this_drum = []
            for strike_promp in self.strike_promps[drum_index]:
                strike_duration = 1#strike_promp.n_discretizations * 0.001 #
                time_steps_this_drum = tf.linspace(t_hit - (self.promp_duration/2) * strike_duration, t_hit + (self.promp_duration/2) * strike_duration, 
                int(strike_promp.n_discretizations/4))
                #
                time_steps_this_drum =np.concatenate([time_steps_this_drum,time_steps_this_drum+total/4,time_steps_this_drum+(total)/2,time_steps_this_drum+(3*total)/4])

                phi_segments_this_drum.append(self.long_promp.get_Phi_for_timesteps(time_steps_this_drum,total))
            self.Phi_per_segment.append(phi_segments_this_drum)



    tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
    def get_cost(self, trajectory,discount=1):
        """tf function to use in tf graph
        trajectory: the trajectory to be optimized"""
        velocities ,acceleration,jerk = self.get_deriv_penalties(trajectory)

        velocities = tf.square(velocities)
        acceleration = tf.square(acceleration)
        jerk = tf.square(jerk)

        strike_rewards = tf.TensorArray(dtype=tf.float32,size=len(self.strike_promps))
        #traj =self.get_intervals(trajectory,[.75,1.85])
        for i in range(len(self.Phi_per_segment)):
            #traj_ = traj[i]
            #strike_rewards = strike_rewards.write(i,self.get_strike_reward(i, traj_))
            strike_rewards = strike_rewards.write(i,self.get_strike_reward(i, trajectory=trajectory))
            

        return - tf.reduce_sum(strike_rewards.stack()) + (tf.reduce_sum(velocities)+tf.reduce_sum(acceleration)+tf.reduce_sum(jerk))/discount

    def get_strike_reward(self, interval_idx, trajectory):
        """gets the reward for the part of trajectory corresponding the 
        interval_idx drum table"""
        log_pi = tf.TensorArray(size = len(self.Phi_per_segment[interval_idx]), dtype=tf.float32)
        for i in range(len(self.Phi_per_segment[interval_idx])): # we call this two times each loop runs 3 times

            

            # 
            # traj = self.component_log_density(interval_idx,i,tf.squeeze(self.strike_promp_means[0][0]))
            traj_segment = tf.experimental.numpy.take(trajectory,self.times_steps[interval_idx])

            log_pi = log_pi.write(i, self.component_log_density(interval_idx,i,traj_segment))
        log_pi = log_pi.stack()
        max_val = tf.reduce_max(log_pi)
        softmax =(max_val + tf.math.log(tf.reduce_sum(tf.exp(log_pi - max_val))))
        return softmax

    def get_deriv_penalties(self,trajectory):
        """get the trajectories velocity, acceleration, jerk penalties"""
        velocities = tf.TensorArray(dtype=tf.float32,size=4)
        acceleration=tf.TensorArray(dtype=tf.float32,size=4)
        jerk=tf.TensorArray(dtype=tf.float32,size=4)
        tr = tf.reshape(trajectory,[4,-1])
        for i in tf.range(4):
            vs = self.my_gradient_tf(tr[i])
            velocities=velocities.write(i,vs)
            a = self.my_gradient_tf(vs)
            acceleration=acceleration.write(i,a)
            jerk=jerk.write(i,self.my_gradient_tf(a))
        return velocities.concat() , acceleration.concat(), jerk.concat()
            

    def component_log_density(self, interval, strike_index, samples: tf.Tensor) -> tf.Tensor:
        """compute the log density under the corresponding ProMPs
        strike_index: index of the drum table"""
        diffs = samples - tf.squeeze(self.strike_promp_means[interval][strike_index])
        diffs = tf.expand_dims(diffs,0)
       
        sqrts = tf.linalg.triangular_solve(self.strike_promp_chols[interval][strike_index], tf.transpose(diffs))
       
        mahalas = - 0.5 * tf.reduce_sum(sqrts * sqrts, axis=0)
        const_parts = - 0.5 * tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(self.strike_promp_chols[interval][strike_index])))) \
                      - 0.5 * self.long_promp.n_discretizations * tf.math.log(2 * pi)
        return mahalas + const_parts

    @staticmethod
    def my_gradient_tf(a):
        """tf function to compute the gradients of a trajectory over the last dimention
        a: trajectory"""
        rght = tf.concat((a[..., 1:], tf.expand_dims(a[..., -1], -1)), -1)
        left = tf.concat((tf.expand_dims(a[...,0], -1), a[..., :-1]), -1)
        ones = tf.ones_like(rght[..., 2:], tf.float32)
        one = tf.expand_dims(ones[...,0], -1)
        divi = tf.concat((one, ones*2, one), -1)
        return (rght-left) / divi


    def plot_trajectory(self,long_trajectory):
        """plots the trajectory and the corresponding ProMPs for different sections of the trajectory """

        plt.ion()
        total =int((self.long_promp.n_discretizations*.004)*1000)
        plt.plot(np.array([j for j in range(total)]),long_trajectory.numpy())
        for i,t_steps in enumerate(self.times_steps):
            t_steps = np.reshape(t_steps,(4,-1))
            
            for j in range(len(self.strike_promp_means[i])):
                means = np.reshape(np.squeeze(self.strike_promp_means[i][j]),(4,-1))
                for t in range(4):
                    plt.plot(t_steps[t],means[t])

                

                

        plt.show()
                    







if __name__ == "__main__":
    def build_long_promp(num_rbfs_long_promp,n_discrets):
        centers = [i/num_rbfs_long_promp for i in range(num_rbfs_long_promp)]
        centers = [i/num_rbfs_long_promp for i in range(num_rbfs_long_promp)]
        covars = [0.06210710710710711 for _ in range(num_rbfs_long_promp)]
        return ProMP(n_discretizations=n_discrets, centers=centers, covars=covars)

    def build_strike_promps(num_discretizations_per_drum):
        strike_promps = []
        for num_discretizations_this_drum in num_discretizations_per_drum:
            this_short_promps = []
            for n_discretizations in num_discretizations_this_drum:
                this_short_promps.append(ProMP(n_discretizations=n_discretizations))
            strike_promps.append(this_short_promps)
        return strike_promps

    def load_means_and_choleskys(strike_promps, ws):   
        """computes the means, covariances of the normal distribution over the trajectories 
        and convert the covariance matrix to choleskys matrix """
        strike_promp_means = []
        strike_promp_covs = []
        strike_covs = []
        for i in range(len(ws)):
            strike_promp_means_this_drum = []
            strike_promp_covs_this_drum = []
            this_strike_covs =[]
            for j in range(len(ws[i])):
                this_promp = strike_promps[i][j]
                mean = tf.convert_to_tensor(np.load(ws[i][j])["means"], dtype=tf.float32)
                covar = tf.convert_to_tensor(np.load(ws[i][j])["covars"], dtype=tf.float32)
                phi_block_diagonal = tf.convert_to_tensor(scipy.linalg.block_diag(this_promp.Phi_matrix, this_promp.Phi_matrix, 
                                                            this_promp.Phi_matrix, this_promp.Phi_matrix))
                phase_means = phi_block_diagonal @ tf.expand_dims(mean, 1)
                phase_covars = tf.linalg.cholesky(phi_block_diagonal @ covar @ tf.transpose(phi_block_diagonal) + 1e-6 * tf.eye(tf.shape(phi_block_diagonal)[0]))
                strike_promp_means_this_drum.append(phase_means)
                strike_promp_covs_this_drum.append(phase_covars)
                this_strike_covs.append(phi_block_diagonal @ covar @ tf.transpose(phi_block_diagonal) )
            strike_promp_means.append(strike_promp_means_this_drum)
            strike_promp_covs.append(strike_promp_covs_this_drum)
            strike_covs.append(this_strike_covs)
        return strike_promp_means, strike_promp_covs ,strike_covs
    
    class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        """decreas the learing rate if the first item in the 
        list passed as last element is the biggest, until the minimum learning rate has been reached"""
        def __init__(self, initial_learning_rate,ceiling = 0.0001):
            self.initial_learning_rate = initial_learning_rate
            self.ceiling = ceiling
           
        
        def __call__(self, step,last_elements=None,):
            changed_lr= self.change_lr(last_elements)
            if changed_lr is None:
                return self.initial_learning_rate
            else:
                st = step/10
                lr = (self.initial_learning_rate - (1/st))
                if lr > self.ceiling:
                    self.initial_learning_rate =lr

                return self.initial_learning_rate

        def change_lr(self,last_elements):
            
            if last_elements is None:
                return
            last_elements = np.abs(last_elements)
            if True in [elem< last_elements[0] for elem in last_elements[1:]]:
                return
            else:
                return True
        
    def grad(x):
        with tf.GradientTape() as tape:
            cost = reward_computer.get_cost(x)
            print(cost)
            grads = tape.gradient(cost,x)
            
        return cost ,grads

    def func(x):

        return [vv.numpy().astype(np.float64)  for vv in grad(tf.Variable(x, dtype=tf.float32))]




    #strike_promps = build_strike_promps([[600]])
    strike_promps = build_strike_promps([[600,600,600],[450, 450, 450]])
    #strike_promps = build_strike_promps([[600],[450]])
    long_promp = build_long_promp(45,1800)
    #long_promp = build_long_promp(45,600)

    # ws = [["/home/atashak/taxi/robot/weights_new15/weights3.npz",],
    #         ["/home/atashak/taxi/robot/weights_new15/weights5.npz",]
    #         ]
    ws = [["/home/atashak/taxi/robot/weights_new15/weights1.npz",
            "/home/atashak/taxi/robot/weights_new15/weights2.npz",
            "/home/atashak/taxi/robot/weights_new15/weights3.npz"],
            ["/home/atashak/taxi/robot/weights_new15/weights4.npz",
            "/home/atashak/taxi/robot/weights_new15/weights5.npz",
            "/home/atashak/taxi/robot/weights_new15/weights6.npz"]
            ]
    # ws = [["/home/atashak/taxi/robot/weights_new15/weights1.npz",],
    # ["/home/atashak/taxi/robot/weights_new15/weights4.npz",] ]
           
    strike_promp_means, strike_promp_chols,strike_covs = load_means_and_choleskys(strike_promps, ws)
    
    reward_computer = Reward(strike_promps, strike_promp_means, strike_promp_chols, long_promp,0.6)
    #reward_computer.set_hit_times(t_hits=[.3], drum_indices=[0])#
    errs = np.array([0.031440544128418055,0.04373688697814933])
    t_hits=np.array([.55,1.45])
    t_hits += errs
    reward_computer.set_hit_times(t_hits=t_hits, drum_indices=[0, 1])
    reward_computer.get_time_indexes(t_hits=t_hits)
    # reward_computer.get_time_indexes(t_hits=[.3])

    #x= tf.Variable(initial_value= np.random.rand(180),dtype=tf.float32)
    #x= tf.Variable(initial_value= np.random.rand(2400),dtype=tf.float32)

    pre_trained = False
    pre_optimize = True
    # lr_schedle = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=.1,decay_steps=5000,decay_rate=0.5)
    # new_lr_schedle = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[2000,5000,8500,11000,25000],values=[0.1,0.05,0.01,0.005,0.002,0.001])
    if pre_trained:
        x = tf.Variable(initial_value=np.load("/home/atashak/taxi/robot/long_trajectory_direct3.npy"))
        lr_iter = np.load("/home/atashak/taxi/robot/long_trajectory_direct3.npz")
        lr = float(lr_iter["lr"])
        i = int(lr_iter["iters"])
        opt = tf.keras.optimizers.Adam(learning_rate=MyLRSchedule(lr))
    elif pre_optimize:
        x= tf.Variable(initial_value= np.random.rand(7200),dtype=tf.float32)
        opt = tf.keras.optimizers.Adam(learning_rate=MyLRSchedule(0.01))
        i=1
        resdd= sopt.minimize(fun=func, x0=x,
                                      jac=True, method='L-BFGS-B')
        x = tf.Variable(resdd.x.astype(np.float32),dtype=np.float32)
    else:
        x= tf.Variable(initial_value= np.random.rand(7200),dtype=tf.float32)
        opt = tf.keras.optimizers.Adam(learning_rate=MyLRSchedule(0.1))
        i=1

    
    hist = []
    lr_border = 5
    # plt_traj = tf.constant(np.load("/home/atashak/taxi/robot/long_trajectory_direct3.npy"))
    # reward_computer.plot_trajectory(plt_traj)

    while True:
        
        with tf.GradientTape() as tape:
            cost = reward_computer.get_cost(x)
            grads = tape.gradient(cost,x)
            opt.apply_gradients(zip([grads],[x]))
        if i%40 ==0 :
            #reward_computer.plot_trajectory(x)
            print(cost)
            #reward_computer.plot_trajectory([.3],[0],x)#reward_computer.plot_trajectory([.75,1.85],[0, 1],x)
       #     print("plotted")
            if len(hist) < lr_border:
                hist.append(cost)
            else:
                opt.learning_rate.__call__(i,hist)
                
                hist = [] # -1*np.max(-1*np.array(hist)) for more aggressive decline in rl
        
        if i%5000 ==0:
            print("saving!!!")
            np.save("long_trajectory_direct_pre_trained.npy",x.numpy())
            np.savez("long_trajectory_direct_pre_trained.npz",lr=np.array(opt.learning_rate.initial_learning_rate),iters = np.array(i))
        i+=1
            
    

    # check if the hit point that is getting optimized is the right place
    print("finished")