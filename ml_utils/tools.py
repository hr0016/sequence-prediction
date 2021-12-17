'''
A module for storing the tools we will work with, alongside the models.

To use this in another place, you can say
    from ml_utils import tools
    
    
'''

import numpy as np
from sklearn.model_selection import KFold


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


def kfold_gen(cells, index, n_folds, rdm_state=None):
    '''
    The "train" and "test" sets here are referred to within the context of 
    k-fold cross validation.
    
    
    '''
    
    if rdm_state == None:
        shuffle = False
    else:
        shuffle = True
    
    # The KFold sklearn instance is a generator that yields train/test indices
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=rdm_state)

    # Identify cell IDs from "cells" array
    for kf_train_cell_indices, kf_test_cell_indices in kf.split(cells):
        
        kf_train_cells = cells[kf_train_cell_indices]
        kf_test_cells = cells[kf_test_cell_indices]
        
        # Debugging print statements
        #print(f"Train cells: {kf_train_cells}")
        #print(f"Test cells: {kf_test_cells}")

        # Get bool arrays that state for each row in "index", whether or not
        # that row contains a train cell (or a test cell)
        kf_train_bool = np.in1d(index, kf_train_cells)
        kf_test_bool = np.in1d(index, kf_test_cells)
        
        # Find the indices of True occurrences in the bool arrays.
        # These are the indices of the array "X" of train and test cell instances
        kf_train_indices = np.where(kf_train_bool)[0]
        kf_test_indices = np.where(kf_test_bool)[0]

        yield kf_train_indices, kf_test_indices




"""
ensure the input is in the shape of (number of in_cycle points, original features) and scaled 
primal feature means unprocessed features - "V", "I", "T"...
"""
def cycle_scale_feature_extract(sample_point_gap, in_cycle_data):
    num_points, num_features = in_cycle_data.shape[0], in_cycle_data.shape[1]
    
    """output the position of the maximum values of all primal features in the designated cycle"""
    
    max_point_ind = np.argmax(in_cycle_data, axis=0)
    
    """time elapsed till maximum values of all primal features """
    t_to_max = max_point_ind * sample_point_gap
    
    """the maximum values of all primal features"""
    max_val = in_cycle_data[max_point_ind,]
    
    start_point = in_cycle_data[0,:]
    end_point = in_cycle_data[-1,:]
    
    """mean values of all primal features"""
    mean_capa = np.mean(in_cycle_data, axis=0)
    
    """compute an erea feature (AUC) of all primal features"""
    AUC_est = [0] * num_features
    for m in range(num_points-1):
        AUC_est += (in_cycle_data[m,]+in_cycle_data[m+1,])/2 * sample_point_gap 
    return max_point_ind, t_to_max, max_val, start_point, end_point, mean_capa, AUC_est


"""
a function builds auto-covariance and cross-covariance features for the input in-cycle data points
"""
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

"""
a function segements the in-cycle time-series data uniformly into windows (without overlaps)
w_size: window size; ws = number of windows constructed
"""
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

"""a function further reshapes the processed data (after all previous processing including cleaning, interpolation and spliting) into a 4D array with dimension [num_cycles, in_cycle_points, window_size, combinatorics of primal features], which includes new 2D covariance features
"""
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

"""
split the trajectory date into any number of cycles (but <= 100) as input and output.
only suitable for a maximum of 100 cycles.
specify the total number of cells, input cycles and trajectory of the variable.
"""

def split_cycles(num_cells, input_num_cycles, full_capa_traj):
    pad_x, pad_y = int(input_num_cycles/5+1), 40
    cell_traj, cell_traj_beyond = [], []
    if input_num_cycles % 5 != 0:
        raise ValueError('input_num_cycles must be multiple of 5')
    for i in range(num_cells):
        for c in range(2, pad_x):
            x = [0]*(pad_x-c) + list(full_capa_traj[i][:c])
            y = list(full_capa_traj[i][c::10]) + [0]*(pad_y-len(full_capa_traj[i][c::10]))
            cell_traj.append(x)
            cell_traj_beyond.append(y)
            X_data = np.array(cell_traj)
            y_data = np.array(cell_traj_beyond)
            X_data = X_data.reshape(len(cell_traj), pad_x, 1)  
            y_data = y_data.reshape(len(cell_traj_beyond), pad_y, 1)            
    return X_data, y_data



"""
a function computes the MAPE when there are zeros in the vectors
"""

def new_mape(Y_actual,Y_Predicted):
    Y_actual = Y_actual.flatten()
    Y_Predicted = Y_Predicted.flatten()
    mask = np.ones(len(Y_actual), dtype=bool)
    zero_ind = np.where(Y_actual == 0)
    mask[zero_ind]=False
    Y_actual = Y_actual[mask]
    Y_Predicted = Y_Predicted[mask]
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

