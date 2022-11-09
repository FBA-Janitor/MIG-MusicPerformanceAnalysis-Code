
import os
import yaml


def read_yaml_to_dict(path: str) -> dict:
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    return data


def get_student_ids_from_constraints(initial_set=None, *args, **kwargs):
    #TODO: fill in
    
    # initial_set is used to limit the search space, setting to None search thru everything
    pass