import json
import os
import warnings
import json


def compare_synthesis(data, filename):
    """ Compares given data to a baseline stored in the file. """
    with open(filename, "w") as fp:
        baseline = json.dump(data, fp)
    return data == baseline

def check_synthesis(synthesis, hls_model, baseline_path):
    """Function to run synthesis and compare results."""
    if synthesis:
        data = hls_model.build()
        if not compare_synthesis(data, baseline_path):
            warnings.warn("Results don't match baseline")
        if data.keys() < {'CSimResults', 'CSynthesisReport'}:
            raise ValueError('Synthesis Failed')
    

        
       
