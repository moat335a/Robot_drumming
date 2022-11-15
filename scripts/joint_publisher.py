#!/usr/bin/env python
import rospy
# import roslib
# roslib.load_manifest("wam_pub")
from get_reward import Reward
import numpy as np
import tensorflow as tf
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


def joint_publisher(t_steps):
    reward = Reward()
    pub = rospy.Publisher('joints_trajectories', JointState, queue_size=10)
    rospy.init_node('joint_state_publisher')
    rate = rospy.Rate(1)
    joints = JointState()
    joints.header = Header()
    joints.name = ["shoulder_yaw","shoulder_pitch","shoulder_roll","elbow"]

    i = 0
    while not rospy.is_shutdown(): 
        
        joints.header.stamp =rospy.Time.now()
        
        
        if i < 9:
            i+=1
        else:
            i=0
        t_step = t_steps[i]
        weights = np.random.rand(12,4) *2
        joints_state = reward.publich_joint_trajector(weights=weights,t_step=t_step)
        joints_state = joints_state.numpy().flatten().tolist()
        joints.position = joints_state
        joints.velocity = []
        joints.effort = []
        rospy.loginfo(joints)
        pub.publish(joints)
        rate.sleep()

if __name__ == "__main__":
    i=0
    
    t_steps = [1/(10-i) for i in range(10)]
    
    
    
    # reward = Reward()
    # joints_state = reward.publich_joint_trajector(weights=weights,t_step=t_step)
    try:
        joint_publisher(t_steps=t_steps)
    except rospy.ROSInterruptException:
        pass

