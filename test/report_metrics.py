import hls4ml
import argparse

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('-d,--dir', dest='dir', default='./', help='The HLS project directory')
  parser.add_argument('-o,--ofile', dest='ofile', default='./metrics.txt', help='The output file')

  args = parser.parse_args()

  report = hls4ml.report.parse_vivado_report(args.dir)
  with open(args.ofile, 'w') as f:
    for key in report.keys():
      f.write("{} {}\n".format(key, report[key]))