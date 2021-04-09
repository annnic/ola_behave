import os

import yaml


def load_yaml(rootdir, name):
    """ (str, str) -> (dict)
    finds name.yaml file in given dir and opens and returns it as a  dict"""
    try:
        filename = os.path.join(rootdir, name + ".yaml")
        print(filename)
        with open(filename) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
            print(params)
            return params
    except:
        print("couldn't find " + name + ".yaml in folder: " + rootdir)
        params = {}
        return params
