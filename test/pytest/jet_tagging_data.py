"""
Download the hls4ml LHC jet tagging dataset (OpenML 42468) used by the QKeras accuracy tests.

The Parquet copy on OpenML's object store downloads several times faster than the ARFF file that
fetch_openml reads, and arrives with a Content-Length, so a truncated download is detected.
fetch_openml remains the fallback when pyarrow is not installed.
"""

import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, get_data_home

OPENML_NAME = 'hls4ml_lhc_jets_hlf'
PARQUET_URL = 'https://data.openml.org/datasets/0004/42468/dataset_42468.pq'
TIMEOUT = 60
RETRIES = 3


def _download(url, path):
    tmp = path.with_suffix('.part')
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as response, tmp.open('wb') as f:
                expected = response.headers.get('Content-Length')
                shutil.copyfileobj(response, f)
            if expected is not None and tmp.stat().st_size != int(expected):
                raise OSError(f'{url}: received {tmp.stat().st_size} of {expected} bytes')
            tmp.replace(path)
            return
        except OSError:
            tmp.unlink(missing_ok=True)
            if attempt == RETRIES:
                raise
            time.sleep(5 * attempt)


def fetch_jet_tagging_data():
    """Return the 16 high-level features as a float64 array and the jet class labels as strings."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        data = fetch_openml(OPENML_NAME)
        return data['data'], data['target']

    path = Path(get_data_home()) / 'hls4ml' / Path(PARQUET_URL).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _download(PARQUET_URL, path)

    table = pq.read_table(path)
    features = [name for name in table.column_names if name != 'class']
    X = np.column_stack([table.column(name).to_numpy() for name in features]).astype(np.float64)
    y = table.column('class').cast(pa.string()).to_numpy()
    return X, y
