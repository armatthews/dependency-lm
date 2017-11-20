import argparse
import sys

import conll

def read_scores(filename):
  with open(filename) as f:
    for line in f:
      yield float(line)

def score(hyp_heads, ref_heads):
  assert len(hyp_heads) == len(ref_heads)

  s = sum(1 for h, r in zip(hyp_heads, ref_heads) if h == r)
  s = 1.0 * s / len(ref_heads)
  return s

parser = argparse.ArgumentParser()
parser.add_argument('samples')
parser.add_argument('gscores')
parser.add_argument('refs')
parser.add_argument('n', type=int)
args = parser.parse_args()

sample_parses = conll.read_parses(open(args.samples))
scores = read_scores(args.gscores)
refs = conll.read_parses(open(args.refs))

for ref_parse in refs:
  best_score = None
  best_hyp = None

  for i in range(args.n):
    hyp_parse = sample_parses.next()
    hyp_score = scores.next()
    if best_score == None or hyp_score < best_score:
      best_hyp = hyp_parse
      best_score = hyp_score

  assert best_hyp != None
  print score(best_hyp.heads, ref_parse.heads)
  sys.stdout.flush()
