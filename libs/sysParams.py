import os
import pickle
"""This module is to implement a class that usage is similar to the easydict library"""
"""=================================================================================================================="""
class SysParams:
    def __init__(self):
        pass
    """Attach new attributes"""
    def __setattr__(self, key, value):
        assert isinstance(key,str)
        if isinstance(value,dict):
            new_item = SysParams()
            for ikey in value.keys():
                new_item.__setattr__(ikey,value[ikey])
            self.__dict__.update({key:new_item})
        else:
            self.__dict__.update({key:value})
    """Implementing the print method"""
    def __repr__(self):
        return '{}'.format(self.__dict__)
    """Save object to file for futher usage"""
    def save(self,i_file_path=None):
        assert isinstance(i_file_path,str)
        assert i_file_path.endswith('.pkl')
        save_path = os.path.split(i_file_path)[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        with open(i_file_path,'wb') as file:
            pickle.dump(self,file)
        return i_file_path
    """Load object for use"""
    @staticmethod
    def load(i_file_path=None):
        assert isinstance(i_file_path,str)
        assert os.path.exists(i_file_path)
        assert i_file_path.endswith('.pkl')
        with open(i_file_path,'rb') as file:
            return pickle.load(file)
"""=================================================================================================================="""
if __name__ == '__main__':
    print('This module is to implement easy dictionary')
    params = SysParams()
    params.name    = 'Hell0'
    params.ex_dict = {'a':1,'b': {'c':1,'d':2}}
    params.train   = SysParams()
    params.train.anchor = [1,2,3]
    params.val     = SysParams()
    params.val.anchor   = [4,5,6]
    print(params)
    print(params.name)
    print(params.train.anchor)
    print(params.val.anchor)
    params.save(os.path.join(os.getcwd(),'config.pkl'))
    reload_object = SysParams.load(os.path.join(os.getcwd(),'config.pkl'))
    print('reload_object Type    = ',type(reload_object))
    print('reload_object Content = ',reload_object)
"""=================================================================================================================="""