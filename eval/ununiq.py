import argparse
import itertools
import sys

parser = argparse.ArgumentParser()
parser.add_argument('scores')
parser.add_argument('uniq_input')
args = parser.parse_args()

scores = {}
with open(args.scores) as f:
  with open(args.uniq_input) as g:
    for score_line, input_line in itertools.izip(f, g):
      score = float(score_line.strip())
      input_line = input_line.strip()
      scores[input_line] = score

for line in sys.stdin:
  line = line.strip()
  if not line in scores:
    print >>sys.stderr, 'Line not found in uniq\'d list:', line
    sys.exit(1)

  print scores[line]
