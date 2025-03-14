import json
import os
import json


def get_baselines_dir():
    """
    Returns the baselines directory path.

    Checks the `BASELINES_DIR` environment variable, which must contain an absolute
    path to the baselines directory. If the environment variable is not set or is 
    invalid, raises a `ValueError`.

    Returns:
        str: Absolute path to the baseline directory.

    Raises:
        ValueError: If `BASELINES_DIR` is not set or is not a valid absolute path.
    """
    baselines_dir = os.getenv('BASELINE_DIR')
    
    if not baselines_dir:
        raise ValueError("The 'BASELINES_DIR' environment variable must be set.")
    
    if not os.path.isabs(baselines_dir):
        raise ValueError(f"The path provided in 'BASELINES_DIR' must be an absolute path. Got: {baselines_dir}")
    
    return baselines_dir


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
    

        
       
