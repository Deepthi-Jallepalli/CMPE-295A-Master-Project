# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:48:04 2019

@author: Keshik

Sources 
    https://github.com/eriklindernoren/PyTorch-YOLOv3
"""

def parse_model_config(path):
    """
        Parses the yolo-v3 layer configuration file and returns module definitions
        
        Args
            path: configuration file path
            
        Returns
            Module definition as a list
    
    """
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


def parse_data_config(path):
    """
        Parses the data configuration file
        
        Args
            path: data configuration file path
        
        Returns
            dictionary containing training options
            
    """
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
