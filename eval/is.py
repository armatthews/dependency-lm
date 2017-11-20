import argparse
import math
import sys

parser = argparse.ArgumentParser()
parser.add_argument('gen_scores')
parser.add_argument('disc_scores')
parser.add_argument('samples_per', type=int)
parser.add_argument('--neg', action='store_true')
args = parser.parse_args()

def logsumexp(s):
  m = max(s)
  t = [v - m for v in s]
  p = sum([math.exp(v) for v in t])
  return m + math.log(p)

s = []
def finish():
  global s
  assert len(s) == 1000
  s = s[:2]
  s = [g - d for (g, d) in s]
  lp = logsumexp(s)
  return lp - math.log(len(s))

final_scores = []
with open(args.gen_scores) as gf:
  with open(args.disc_scores) as df:
    for g, d in zip(gf, df):
      g, d = float(g), float(d)
      if args.neg:
        assert g >= 0 and d >= 0, 'Scores should represent negative log probs.'
        g = -g
        d = -d
      else:
        assert g <= 0 and d <= 0, 'Scores should represent log probs.'
      if d == float('-inf'):
        print >>sys.stderr, 'issue!'
        #d = -1000000.0
      s.append((g, d))
      if len(s) == args.samples_per:
        score = finish()
        final_scores.append(score)
        print score
        s = []

assert len(s) == 0
print sum(final_scores), sum(final_scores) / len(final_scores)
