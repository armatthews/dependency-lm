from __future__ import print_function

import itertools
import sys

from oracle import OracleComputer


def read_oracle(filename):
  with open(filename) as f:
    for line in f:
      yield line.split()


def get_pos_tags(lines):
  pos_tags = []
  for line in lines:
    parts = line.split('\t')
    pos = parts[4]
    pos_tags.append(pos)
  return pos_tags


def convert(actions, pos_tags):
  actions.append('shift-root')
  actions.append('left-root')
  pos_tags.append('root')

  shifts = [a for a in actions if a.startswith('shift')]
  if len(shifts) != len(pos_tags):
    print(len(shifts), len(pos_tags))
    print(' '.join([s.split('-', 1)[1] for s in shifts]))
    print(' '.join(pos_tags))
  assert len(shifts) == len(pos_tags)
  buf = ['%s-%s' % (a.split('-', 1)[1], pos) for (a, pos) in zip(shifts, pos_tags)]
  stack = []
  print()
  for action in actions:
    print('[%s][%s]' % (', '.join(reversed(stack)), ', '.join(buf)))
    if action.startswith('left'):
      assert len(stack) >= 2
      del stack[-2]
      print('LEFT-ARC(%s)' % (action.split('-', 1)[1]))
    elif action.startswith('right'):
      assert len(stack) >= 2
      del stack[-1]
      print('RIGHT-ARC(%s)' % (action.split('-', 1)[1]))
    elif action.startswith('shift'):
      assert len(buf) > 0
      stack.append(buf[0])
      buf = buf[1:]
      print('SHIFT')
    else:
      assert False
  print('[%s][%s]' % (', '.join(reversed(stack)), ', '.join(buf)))

def main():
  lines = []
  for line in sys.stdin:
    line = line.strip()
    if not line:
      oracle = OracleComputer(lines).compute()
      pos_tags = get_pos_tags(lines)
      convert(oracle, pos_tags)
      lines = []
    else:
      lines.append(line)

if __name__ == '__main__':
  main()

