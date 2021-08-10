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