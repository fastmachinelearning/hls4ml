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
        
        # Assert that the synthesis results match the baseline
        assert compare_synthesis(data, baseline_path), \
            "Synthesis results do not match the baseline"

        # Assert that the required keys are present in the synthesis data
        assert {'CSimResults', 'CSynthesisReport'}.issubset(data.keys()), \
            "Synthesis failed: Missing expected keys in the synthesis report"
    

        
       
