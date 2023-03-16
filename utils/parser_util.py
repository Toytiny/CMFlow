import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R



def parse_args_from_yaml(yaml_path):
    
    with open(yaml_path, 'r') as fd:
        args = yaml.safe_load(fd)
        args = EasyDict(d=args)
    
    return args
   
class EasyDict(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__
    
    
def get_matrix_from_ext(ext):
    

    N = np.size(ext,0)
    if ext.ndim==2:
        rot = R.from_euler('ZYX', ext[:,3:], degrees=True)
        rot_m = rot.as_matrix()
        tr = np.zeros((N,4,4))
        tr[:,:3,:3] = rot_m
        tr[:,:3, 3] = ext[:,:3]
        tr[:, 3, 3] = 1
    if ext.ndim==1:
        rot = R.from_euler('ZYX', ext[3:], degrees=True)
        rot_m = rot.as_matrix()
        tr = np.zeros((4,4))
        tr[:3,:3] = rot_m
        tr[:3, 3] = ext[:3]
        tr[ 3, 3] = 1
    return tr

