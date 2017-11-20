import argparse
import sys

import conll

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int)
args = parser.parse_args()

parse = []
for line in sys.stdin:
  line = line.strip()
  if not line:
    parse = '\n'.join(parse)
    for i in range(args.n):
      print parse
      print
    sys.stdout.flush()
    parse = []
  else:
    parse.append(line) 
