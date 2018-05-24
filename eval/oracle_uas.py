import argparse
import itertools
import sys

import conll

parser = argparse.ArgumentParser()
parser.add_argument('ref_conll')
parser.add_argument('n', type=int)
parser.add_argument('--all', '-a', action='store_true', help='Show all scores, rather than just the one best per sentence')
parser.add_argument('--length', '-l', action='store_true', help='Multiply all scores by sentence length')
args = parser.parse_args()

def score(hyp_heads, ref_heads):
  assert len(hyp_heads) == len(ref_heads)

  s = sum(1 for h, r in zip(hyp_heads, ref_heads) if h == r)
  if not args.length:
    s = 1.0 * s / len(ref_heads)
  return s

total_correct = 0
total_words = 0

hyp_parses = conll.read_parses(sys.stdin)
for ref_parse in conll.read_parses(open(args.ref_conll)):
  best_score = 0
  for i in range(args.n):
    hyp_parse = hyp_parses.next()
    s = score(hyp_parse.heads, ref_parse.heads)
    if not args.length:
      total_correct += s * len(ref_parse.heads)
    else:
      total_correct += s
    total_words += len(ref_parse.heads)
    if args.all:
      print s
    if s > best_score:
      best_score = s
  if not args.all:
    print best_score
  sys.stdout.flush()

print 'Overall UAS:', 100 * total_correct / total_words
