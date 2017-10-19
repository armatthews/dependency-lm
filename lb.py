import argparse
import math
import sys

parser = argparse.ArgumentParser()
parser.add_argument('gen_scores')
parser.add_argument('samples_per', type=int)
parser.add_argument('--neg', action='store_true')
args = parser.parse_args()

s = []
def finish():
  global s
  s = list(set(s))
  m = max(s)
  s = [v - m for v in s]
  p = sum([math.exp(v) for v in s])
  assert p > 0.0, 'Invalid total prob: %f' % p
  return m + math.log(p)

final_scores = []
with open(args.gen_scores) as gf:
  for g in gf:
    g = float(g)
    if args.neg:
      assert g >= 0, 'Scores should represent negative log probs.'
      g = -g
    else:
      assert g <= 0, 'Scores should represent log probs.'
    s.append(g)
    if len(s) == args.samples_per:
      score = finish()
      final_scores.append(score)
      print score
      s = []

assert len(s) == 0
print sum(final_scores), sum(final_scores) / len(final_scores)
