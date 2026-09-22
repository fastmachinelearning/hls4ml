"""Run bitstreams built by test_vitis_unified.py on a PYNQ board and compare with the saved software prediction.

For each project copy export/system.bit, export/system.hwh, the generated driver, and the x_input*.npy and
y_pred_sw*.npy files of the test output directory into one folder, then run on the board:

    python3 vitis_unified_hw_test.py axi_stream axi_master multi_io
"""

import glob
import importlib.util
import os
import sys
import time

import numpy as np

RTOL, ATOL = 1e-3, 1e-4


def load_driver(project_dir):
    driver = [f for f in os.listdir(project_dir) if f.endswith('_driver.py')][0]
    spec = importlib.util.spec_from_file_location('hls4ml_driver', os.path.join(project_dir, driver))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return driver, module


def load_arrays(project_dir):
    if os.path.exists(os.path.join(project_dir, 'x_input.npy')):
        xs = [np.load(os.path.join(project_dir, 'x_input.npy'))]
        ys = [np.load(os.path.join(project_dir, 'y_pred_sw.npy'))]
    else:
        xs = [np.load(f) for f in sorted(glob.glob(os.path.join(project_dir, 'x_input_*.npy')))]
        ys = [np.load(f) for f in sorted(glob.glob(os.path.join(project_dir, 'y_pred_sw_*.npy')))]
    return xs, ys


def run_project(project_dir):
    driver, drv = load_driver(project_dir)
    xs, ys = load_arrays(project_dir)
    print(f'== {project_dir}: {driver}, {len(xs)} input(s), {len(ys)} output(s), batch {xs[0].shape[0]}')

    x_shape = xs[0].shape if len(xs) == 1 else [x.shape for x in xs]
    y_shape = ys[0].shape if len(ys) == 1 else [y.shape for y in ys]
    t0 = time.time()
    nn = drv.NeuralNetworkOverlay(os.path.join(project_dir, 'system.bit'), x_shape, y_shape, input_dtype=xs[0].dtype)
    print(f'   overlay loaded in {time.time() - t0:.1f} s, kernel {type(nn.ip).__name__} at 0x{nn.ip.mmio.base_addr:x}')

    x_arg = xs[0] if len(xs) == 1 else xs
    y_hw, exec_time, rate = nn.predict(x_arg, profile=True)
    y_hw = [np.array(y) for y in (y_hw if isinstance(y_hw, (list, tuple)) else [y_hw])]

    ok = True
    for idx, (hw, sw) in enumerate(zip(y_hw, ys)):
        hw = hw.reshape(sw.shape)
        close = np.allclose(hw, sw, rtol=RTOL, atol=ATOL)
        ok = ok and close
        print(f'   output {idx}: max |hw - sw| = {np.abs(hw - sw).max():.3e}, match = {close}')

    # a second run must give the same result: the kernel restarts cleanly
    y_again = nn.predict(x_arg)
    y_again = [np.array(y) for y in (y_again if isinstance(y_again, (list, tuple)) else [y_again])]
    repeat = all(np.array_equal(a, b) for a, b in zip(y_hw, y_again))
    print(f'   second run identical = {repeat}')
    return ok and repeat


def main(projects):
    results = {project: run_project(project) for project in projects}
    print()
    for project, passed in results.items():
        print(f'{project:12s} {"PASS" if passed else "FAIL"}')
    return all(results.values())


if __name__ == '__main__':
    projects = sys.argv[1:] or ['axi_stream', 'axi_master', 'multi_io']
    sys.exit(0 if main(projects) else 1)
