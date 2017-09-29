from __future__ import print_function

from collections import defaultdict
import sys

class OracleComputer:
  def __init__(self, lines):
    words = []
    heads = []
    poses = []
    rels = []
    children = defaultdict(list)
    for line in lines:
      parts = line.split('\t')
      assert(len(parts) == 8)
      index, word, _, pos, __, ___, head, rel = parts
      index = int(index)
      head = int(head)
      assert index == len(words) + 1
      assert index == len(heads) + 1
      assert index == len(poses) + 1
      assert index == len(rels) + 1
      words.append(word)
      heads.append(head - 1)
      poses.append(pos)
      rels.append(rel)
      children[head - 1].append(index - 1)

    self.words = words
    self.heads = heads
    self.poses = poses
    self.rels = rels
    self.children = children


    self.stack = []
    self.done = [False for _ in words]
    self.actions = []

  def can_shift(self, head, child):
    return self.done[child] and self.heads[child] == head

  def can_shift_left(self):
    if len(self.stack) < 2:
      return False

    head = self.stack[-2]
    child = self.stack[-1]
    return self.can_shift(head, child)

  def can_shift_right(self):
    if len(self.stack) < 2:
      return False

    head = self.stack[-1]
    child = self.stack[-2]
    return self.can_shift(head, child)

  def shift_left(self):
    head = self.stack[-2]
    child = self.stack[-1]
    #print('left_%d-%d' % (head, child))
    self.actions.append('left-%s' % (self.rels[child]))
    self.stack.pop()
    self.stack.pop()
    self.stack.append(head)

  def shift_right(self):
    head = self.stack[-1]
    child = self.stack[-2]
    #print('right_%d-%d' % (head, child))
    self.actions.append('right-%s' % (self.rels[child]))
    self.stack.pop()
    self.stack.pop()
    self.stack.append(head)

  def check_if_done(self, index):
    for child in self.children[index]:
      if not self.done[child]:
        return False
    return True

  def compute(self):
    for i in range(len(self.words)):
      #print('Shifting %d (head=%d)' % (i, self.heads[i]))
      self.actions.append('shift-%s' % (self.words[i]))
      self.stack.append(i)

      p = i
      while p >= 0:
        if self.check_if_done(p):
          self.done[p] = True
          p = self.heads[p]
        else:
          break

      while True:
        did_something = False

        if self.can_shift_left():
          self.shift_left()
          did_something = True
        elif self.can_shift_right():
          self.shift_right()
          did_something = True

        if not did_something:
          break

    assert len(self.stack) == 1
    assert self.done == [True for _ in self.words]
    assert self.rels[self.stack[-1]] == 'ROOT'
    return self.actions

lines = []
for line in sys.stdin:
  line = line.strip()
  if not line:
    comp = OracleComputer(lines)
    print(' '.join(comp.compute()))
    lines = []
  else:
    lines.append(line)
