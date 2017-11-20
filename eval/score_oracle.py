import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int)
args = parser.parse_args()

best = None
for i, line in enumerate(sys.stdin):
  score = float(line)
  if best == None or score < best:
    best = score
  if i % args.n == args.n - 1:
    print best
    best = None
