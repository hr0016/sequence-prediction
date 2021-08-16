'''
A module for storing the tools we will work with, alongside the models.

To use this in another place, you can say
    from ml_utils import tools
    
    
'''

import numpy as np


def lr_scheduler(epoch, lr):
    '''
    Learning rate scheduler, to be used with the LearningRateScheduler
    Keras callback.
    
    '''
    decay_rate = 0.95
    decay_step = 10
    if epoch % decay_step == 0 and epoch:
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr

# ensure the input is in the shape of (number of in_cycle points, original features) and scaled 
def cycle_scale_feature_extract(sample_point_gap, in_cycle_data):
    num_points, num_features = in_cycle_data.shape[0], in_cycle_data.shape[1]
    max_point_ind = np.argmax(in_cycle_data, axis=0)
    t_to_max = max_point_ind * sample_point_gap
    max_val = in_cycle_data[max_point_ind,]
    start_point = in_cycle_data[0,:]
    end_point = in_cycle_data[-1,:]
    mean_capa = np.mean(in_cycle_data, axis=0)
    AUC_est = [0] * num_features
    for m in range(num_points-1):
        AUC_est += (in_cycle_data[m,]+in_cycle_data[m+1,])/2 * sample_point_gap 
    return max_point_ind, t_to_max, max_val, start_point, end_point, mean_capa, AUC_est

def cov_feature_extract(in_cycle_data):
    n_features = in_cycle_data.shape[1] 
    if n_features == 1:
        cm = np.outer(in_cycle_data, in_cycle_data.T)
    else:
        cm = []
        for i in range(n_features):
            for k in range(i+1):
                cm.append(np.outer(in_cycle_data[:,i], in_cycle_data[:,k].T)) 
        cm = np.stack(cm, axis=2)
    return cm

def create_window(in_cycle_data, w_size: int):
    num_points, num_features = in_cycle_data.shape[0], in_cycle_data.shape[1]
    std_size = np.round(np.sqrt(num_points))
    #if np.abs(w_size-std_size) / std_size >= 0.5:
    #    raise ValueError('window size is too large or too small')
    if num_points % w_size != 0:
        raise ValueError('window size not devidable by the number of in-cycle points')
    ws = int(num_points / w_size) 
    w_data = in_cycle_data.reshape((ws, w_size, num_features))
    return w_data

def cov_data_construct(original_data, w_size):
    in_cycle_points = original_data.shape[1]
    ws = in_cycle_points / w_size
    ws3d = []
    for m in range(original_data.shape[0]):
        wdata = create_window(original_data[m,], w_size)
        ws4d = np.stack([cov_feature_extract(wdata[i,]) for i in range(int(ws))], axis=0)
        ws3d.append(ws4d.reshape((int(ws*w_size), w_size, ws4d.shape[-1])))
    ws3d = np.stack(ws3d, axis=0)
    return ws3d