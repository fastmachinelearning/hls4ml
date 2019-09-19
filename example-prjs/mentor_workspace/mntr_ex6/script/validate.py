from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt

ATOL=5e-02
RTOL=5e-02

def validate(ref, imp, rtol, atol):
    return np.allclose(ref, imp, rtol=rtol, atol=atol)

def plot_histogram(absolute_error):
    n, bins, patches = plt.hist(x=absolute_error, bins='auto', color='#1B9E77', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Absolute error')
    plt.ylabel('Frequency')
    plt.title('abs(ref - imp)')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    plt.axvline(absolute_error.mean(), color='#D95F02', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(absolute_error.mean()*1.1, max_ylim*0.9, 'Mean: {:.6f}'.format(absolute_error.mean()))
    plt.show()

def main(args):
    ref_data = np.loadtxt(args.ref_file)
    imp_data = np.loadtxt(args.imp_file)
    absolute_error = np.absolute(ref_data - imp_data)


    print('INFO: Check if the implementation and reference predictions are equal within the given tolerance')
    print('INFO:     Implementation file:', args.imp_file)
    print('INFO:     Reference file:', args.ref_file)
    print('INFO:     Relative tolerance:', args.rtol)
    print('INFO:     Absolute tolerance:', args.atol)
    print('INFO:     Formula: absolute(ref - imp) <= (atol + rtol * absolute(imp))')

    res = validate(ref_data, imp_data, args.rtol, args.atol)
    if (res):
        print('INFO: Validation: PASS')
    else:
        print('ERROR: Validation: FAIL')

    print('INFO:     Absolute difference:', absolute_error)
    print('INFO:     Absolute difference (max):', np.max(absolute_error))
    print('INFO:     Absolute difference (min):', np.min(absolute_error))

    plot_histogram(absolute_error)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', action='store', dest='ref_file', help='Reference-prediction file.')
    parser.add_argument('-i', action='store', dest='imp_file', help='Implementation-prediction file.')
    parser.add_argument('-at', action='store', dest='atol', default=ATOL, help='Absolute tolerance parameter.')
    parser.add_argument('-rt', action='store', dest='rtol', default=RTOL, help='Relative tolerance parameter.')
    args = parser.parse_args()
    if not args.ref_file: parser.error('Reference file is required.')
    if not args.imp_file: parser.error('Implementation file is required.')

    main(args)
