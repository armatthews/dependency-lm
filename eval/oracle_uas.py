import argparse
import itertools
import sys

import conll

parser = argparse.ArgumentParser()
parser.add_argument('ref_conll')
parser.add_argument('n', type=int)
args = parser.parse_args()

def score(hyp_heads, ref_heads):
  assert len(hyp_heads) == len(ref_heads)

  s = sum(1 for h, r in zip(hyp_heads, ref_heads) if h == r)
  s = 1.0 * s / len(ref_heads)
  return s

hyp_parses = conll.read_parses(sys.stdin)
for ref_parse in conll.read_parses(open(args.ref_conll)):
  best_score = 0
  for i in range(args.n):
    hyp_parse = hyp_parses.next()
    s = score(hyp_parse.heads, ref_parse.heads)
    if s > best_score:
      best_score = s
  print best_score
  sys.stdout.flush()
