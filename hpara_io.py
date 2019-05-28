import os
import keyword
import json
from collections import namedtuple


def _json_object_hook(in_dict):
    return namedtuple('X', in_dict.keys())(*in_dict.values())


def json2obj(json_str):
    """
        convert json string to python object
    """
    return json.loads(json_str, object_hook=ParametersFromJSON)

class ParametersFromJSON(object):
    '''
        Manage hyperparameters
    '''

    def __init__(self, hp_dict):
        self.update(hp_dict)

    def update(self, hp_dict):
        '''
            update the hps with dictionary
        '''
        for key in keyword.kwlist:
            word = hp_dict.pop(key, None)
            if word:
                print('The parameter {} is illegal and excluded'.format(word))
        # https://stackoverflow.com/questions/6578986/how-to-convert-json-data-into-a-python-object
        vars(self).update(hp_dict)

    def hyperparameter_parse(self, json_str):
        """
            parse hyperparameters from json_str to python objects
            into original hyperparameters (org_hps) object
        """
        json_dict = json.loads(json_str)
        self.update(json_dict)