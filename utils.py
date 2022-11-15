import numpy as np

def get_indexes(arr_time,time_optitrack):
        indices =[]
        for tt in arr_time:
            index = 0
            min_dist = 100
            
            for i,t in enumerate(time_optitrack):
                if (t-tt <min_dist) and (t-tt >0):
                    min_dist = t-tt
                    index=i
                i+=1
            indices.append(index)
        return indices

def to_transformation_matrix(trans_matrix,rot_matrix):  
    transformation_matrix = np.zeros((4,4))
    transformation_matrix[0:-1,-1]  = np.squeeze(trans_matrix)
    transformation_matrix[:-1,0:-1] = np.squeeze( rot_matrix)
    transformation_matrix[-1,-1] = 1
    return transformation_matrix

def to_rot_trans(matrix):
    return matrix[:,0:-1,0:-1],matrix[:,:-1,-1:]  # rot, trans