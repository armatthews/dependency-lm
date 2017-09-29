from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import sys

import dynet_config
dynet_config.set(mem=4096, weight_decay=0.0)
dynet_config.set_gpu()
import dynet as dy

from utils import Vocabulary
from utils import read_corpus
from utils import MLP
from harness import train


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
    #self.exp = dy.scalarInput(-0.5)

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
      #word_emb_norm = dy.pow(dy.dot_product(word_emb, word_emb), self.exp)
      #word_emb = word_emb * word_emb_norm
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
      #word_emb_norm = dy.pow(dy.dot_product(word_emb, word_emb), self.exp)
      #word_emb = word_emb * word_emb_norm
      state = state.add_input(word_emb)
    return sent

  def param_collection(self):
    return self.pc

  @staticmethod
  def from_spec(spec, pc):
    rnnlm = RNNLM(pc, *spec)
    return rnnlm


def sample_sentence(rnnlm, vocab):
  eos = vocab.convert('</s>')
  sampled_sent = rnnlm.sample(eos, 100)
  sampled_sent = [vocab.to_word(word_id) for word_id in sampled_sent]
  return ' '.join(sampled_sent)


def main():
  print('Invoked as:', ' '.join(sys.argv), file=sys.stderr)
  parser = argparse.ArgumentParser()
  parser.add_argument('corpus')
  parser.add_argument('dev_corpus')
  parser.add_argument('--layers', type=int, default=1)
  parser.add_argument('--emb_dim', type=int, default=128)
  parser.add_argument('--hidden_dim', type=int, default=128)
  parser.add_argument('--minibatch_size', type=int, default=1)
  parser.add_argument('--autobatch', action='store_true')
  parser.add_argument('--dropout', type=float, default=0.0)
  parser.add_argument('--output', type=str, default='')
  args = parser.parse_args()

  if args.output == '':
    args.output = '/tmp/model%d' % random.randint(0, 0xFFFF)
  print('Output file:', args.output, file=sys.stderr)

  vocab = Vocabulary()
  train_corpus = read_corpus(args.corpus, vocab)
  dev_corpus = read_corpus(args.dev_corpus, vocab)
  print('Vocab size:', len(vocab), file=sys.stderr)

  with open(args.output + '.vocab', 'w') as f:
    for word in vocab.i2w:
      print(word, file=f)

  pc = dy.ParameterCollection()
  optimizer = dy.SimpleSGDTrainer(pc, 1.0)
  model = RNNLM(pc, args.layers, args.emb_dim, args.hidden_dim, len(vocab))
  print('Total parameters:', pc.parameter_count(), file=sys.stderr)

  train(model, train_corpus, dev_corpus, optimizer, args)

if __name__ == '__main__':
  main()
