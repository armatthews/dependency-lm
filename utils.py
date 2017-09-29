from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys
import dynet as dy


class Vocabulary:
  def __init__(self):
    self.i2w = []
    self.w2i = {}
    self.frozen = False

  def convert(self, word):
    if word not in self.w2i:
      if self.frozen:
        assert False, ('Invalid attempt to convert unknown word "%s" on a '
                        'frozen vocabulary.' % (word))
      self.w2i[word] = len(self.i2w)
      self.i2w.append(word)
    return self.w2i[word]

  def to_word(self, word_id):
    return self.i2w[word_id]

  def __len__(self):
    return len(self.i2w)


class LinearLayer:
  def __init__(self, pc, in_size, out_size):
    self.spec = (in_size, out_size)
    self.pc = pc.add_subcollection()
    self.pw = self.pc.add_parameters((out_size, in_size))
    self.pb = self.pc.add_parameters((out_size,))
    self.w = None
    self.b = None

  def new_graph(self):
    self.w = dy.parameter(self.pw)
    self.b = dy.parameter(self.pb)

  def __call__(self, x):
    assert x != None
    return self.w * x + self.b

  def param_collection(self):
    return self.pc

  @staticmethod
  def from_spec(spec, pc):
    return LinearLayer(pc, *spec)


class MLP:
  def __init__(self, pc, sizes):
    self.pc = pc.add_subcollection()
    self.spec = sizes
    self.layers = []
    for i in range(0, len(sizes) - 1):
      in_size = sizes[i]
      out_size = sizes[i + 1]
      layer = LinearLayer(self.pc, in_size, out_size)
      self.layers.append(layer)
    self.dropout_rate = 0.0

  def new_graph(self):
    for layer in self.layers:
      layer.new_graph()

  def set_dropout(self, r):
    self.dropout_rate = r

  def __call__(self, x):
    assert x != None
    h = x
    for i, layer in enumerate(self.layers):
      h = layer(h)
      if i != len(self.layers) - 1:
        h = dy.tanh(h)
        if self.dropout_rate > 0.0:
          h = dy.dropout(h, self.dropout_rate)
    return h

  def param_collection(self):
    return self.pc

  @staticmethod
  def from_spec(spec, pc):
    return MLP(pc, spec)
 

def read_corpus(filename, vocab):
  eos = vocab.convert('</s>')
  corpus = []
  with open(filename) as f:
    for line in f:
      words = line.split()
      word_ids = [vocab.convert(word) for word in words]
      word_ids.append(eos)
      corpus.append(word_ids)
  return corpus

def sample(dist):
  r = random.random()
  for i, p in enumerate(dist):
    if r < p:
      return i
    r -= p

  print('WARNING: sample random was 1.0', file=sys.stderr)
  return len(dist) - 1
