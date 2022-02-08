import os
import sys
#import math
#import time
#import numpy as np
#import pandas as pd
import pickle


#%%
python_version_major = sys.version_info.major


#%%
def openfile(filename, mode, makefolder=False):
    ## create a folder if not exist
    if (mode[0] == 'w' or mode[0] == 'a'):
        if (makefolder == True):
            foldername = os.path.dirname(filename)
            if (os.path.isdir(foldername) == False):
                os.makedirs(foldername)
                print('openfile creates folder {}'.format(foldername))
    ##
    fh = open(filename, mode)    
    return fh

    
## In the dart project, make it compatible to Python2 by setting encoding='latin1' and protocol=2
## python3 uses encoding='ASCII' and protocol=None    
def load_pickle(filename, encoding='latin1', printflag=False):
    fh = open(filename, 'rb')
    #param = pickle.load(fh)    
    if (python_version_major == 3):
        param = pickle.load(fh, encoding=encoding)
    else:
        param = pickle.load(fh)
    fh.close()
    if (printflag == True):
        print('load_pickle from {}'.format(filename))
    return param

    
def dump_pickle(filename, content, protocol=2, printflag=False, makefolder=False):
    fh = openfile(filename, 'wb', makefolder=makefolder)    
    if (python_version_major == 3):    
        pickle.dump(content, fh, protocol=protocol)
    else:
        pickle.dump(content, fh)
    if (printflag == True):
        print('dump_pickle to {}'.format(filename))
    fh.close()
 

#%%
def execfile(filename):
    return eval(open(filename).read())


def submit_unix_cmd(cmd):
    import subprocess
    print(cmd)    
    p = subprocess.call(cmd, shell=True)
    #print(p)


#%%
## for numpy
def copy_numberarray_container(container, new_dtype=None):
    if type(container) is list: 
        res = []
        for array in container:
            if array is None:
                res.append(None)
            elif new_dtype is None:
                res.append(array.copy())
            else:                
                res.append(array.astype(new_dtype))
    elif type(container) is dict: 
        res = {}
        for key in container.keys():
            array = container[key]
            if array is None:
                res[key] = None
            elif new_dtype is None:
                res[key] = array.copy()
            else:
                res[key] = array.astype(new_dtype)
    else:
        res = None
    
    return res


## for pytorch
def copy_tensor_container(container, new_dtype=None):
    if type(container) is list: 
        res = []
        for array in container:
            if array is None:
                res.append(None)
            elif new_dtype is None:
                res.append(array.clone().detach())
            else:                
                res.append(array.type(new_dtype))
    elif type(container) is dict: 
        res = {}
        for key in container.keys():
            array = container[key]
            if array is None:
                res[key] = None
            elif new_dtype is None:
                res[key] = array.clone().detach()
            else:
                res[key] = array.type(new_dtype)
    else:
        res = None
    
    return res
