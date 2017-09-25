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
        assert False, ('Invalid attempt to convert unknown word "%s" on a'
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


class RNNLM:
  def __init__(self, pc, layers, emb_dim, hidden_dim, vocab_size):
    self.spec = (layers, emb_dim, hidden_dim, vocab_size)
    self.pc = pc.add_subcollection()
    self.rnn = dy.LSTMBuilder(layers, emb_dim, hidden_dim, self.pc)
    self.initial_state_params = [self.pc.add_parameters((hidden_dim,)) for _ in range(2 * layers)]
    self.word_embs = self.pc.add_lookup_parameters((vocab_size, emb_dim))
    self.output_mlp = MLP(self.pc, [hidden_dim, hidden_dim, vocab_size])

  def new_graph(self):
    self.output_mlp.new_graph()
    self.initial_state = [dy.parameter(p) for p in self.initial_state_params]

  def set_dropout(self, r):
    self.output_mlp.set_dropout(r)
    self.rnn.set_dropout(r)

  def build_graph(self, sent):
    state = self.rnn.initial_state()
    state = state.set_s(self.initial_state)

    losses = []
    for word in sent:
      assert state != None
      so = state.output()
      assert so != None
      output_dist = self.output_mlp(so)
      loss = dy.pickneglogsoftmax(output_dist, word)
      losses.append(loss)
      word_emb = dy.lookup(self.word_embs, word)
      state = state.add_input(word_emb)
    return dy.esum(losses)

  def sample(self, eos, max_len):
    #dy.renew_cg()
    #self.new_graph()
    state = self.rnn.initial_state()
    state = state.set_s(self.initial_state)
    sent = []
    while len(sent) < max_len:
      assert state != None
      so = state.output()
      assert so != None
      output_dist = dy.softmax(self.output_mlp(so))
      output_dist = output_dist.vec_value()
      word = sample(output_dist)
      sent.append(word)
      if word == eos:
        break
      word_emb = dy.lookup(self.word_embs, word)
      state = state.add_input(word_emb)
    return sent

  def param_collection(self):
    return self.pc

  @staticmethod
  def from_spec(spec, pc):
    rnnlm = RNNLM(pc, *spec)
    return rnnlm
    

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
