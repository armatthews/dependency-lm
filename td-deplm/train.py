from __future__ import print_function
import argparse
import collections
import random
import sys

import dynet_config
dynet_config.set(mem=10*1024)
dynet_config.set_gpu()
import dynet as dy

sys.path.append('/home/austinma/git/rnnlm/')
sys.path.append('../')
from utils import Vocabulary
from utils import MLP
import harness

ParserState = collections.namedtuple(
    'ParserState', 'parent, stack_state, comp_state, left_done')

class TopDownDepLM:
  def __init__(self, pc, vocab, layers, state_dim, final_hidden_dim, tied):
    self.vocab = vocab
    self.tied = tied
    self.done_with_left = vocab.convert('</LEFT>')
    self.done_with_right = vocab.convert('</RIGHT>')
    vocab_size = len(self.vocab)

    self.pc = pc.add_subcollection()
    if not self.tied:
      self.word_embs = self.pc.add_lookup_parameters((vocab_size, state_dim))

    self.stack_lstm = dy.LSTMBuilder(layers, state_dim, state_dim, self.pc)
    self.comp_lstm = dy.LSTMBuilder(layers, state_dim, state_dim, self.pc)
    self.final_mlp = MLP(self.pc, [2 * state_dim, final_hidden_dim, vocab_size])

    self.stack_initial_state_params = [
        self.pc.add_parameters((state_dim,)) for _ in range(2 * layers)]
    self.comp_initial_state_params = [
        self.pc.add_parameters((state_dim,)) for _ in range(2 * layers)]

  def set_dropout(self, r):
    self.stack_lstm.set_dropout(r)
    self.comp_lstm.set_dropout(r)
    self.final_mlp.set_dropout(r)

  def new_graph(self):
    self.final_mlp.new_graph()
    self.stack_initial_state = [
        dy.parameter(p) for p in self.stack_initial_state_params]
    self.comp_initial_state = [
        dy.parameter(p) for p in self.comp_initial_state_params]

  def embed_word(self, word):
    if self.tied:
      word_embs = self.final_mlp.layers[-1].w
      word_emb = dy.select_rows(word_embs, [word])
      word_emb = dy.transpose(word_emb)
    else:
      word_emb = dy.lookup(self.word_embs, word)
    return word_emb

  def add_input(self, state, word):
    word_emb = self.embed_word(word)
    if word == self.done_with_left:
      assert not state.left_done
      stack_state = state.stack_state
      comp_state = state.comp_state.add_input(word_emb)
      left_done = True
      parent = state.parent
    elif word == self.done_with_right:
      assert state.left_done
      if state.parent == None:
        parent = None
        stack_state = None
        comp_state = None
        left_done = None
      else:
        stack_state = state.parent.stack_state
        comp_state = state.comp_state.add_input(word_emb)
        this_word_subtree = state.comp_state.output()
        comp_state = state.parent.comp_state.add_input(this_word_subtree)
        left_done = state.parent.left_done
        parent = state.parent.parent
    else:
      stack_state = state.stack_state.add_input(word_emb)
      # We also seed the comp LSTM with the word.
      # This is debatable, but somehow the head word needs to make it into the
      # composed representation before it gets added to its parent.
      comp_state = self.comp_lstm.initial_state().set_s(
          self.comp_initial_state).add_input(word_emb)
      left_done = False
      parent = state
    return ParserState(parent, stack_state, comp_state, left_done)

  def new_sent(self):
    stack_state = self.stack_lstm.initial_state()
    stack_state = stack_state.set_s(self.stack_initial_state)
    comp_state = self.comp_lstm.initial_state()
    comp_state = comp_state.set_s(self.comp_initial_state)
    return ParserState(None, stack_state, comp_state, True)

  warned = False
  def compute_loss(self, state, word):
    stack_output = state.stack_state.output()
    comp_output = state.comp_state.output()
    final_input = dy.concatenate([stack_output, comp_output])
    logits = self.final_mlp(final_input)
    #loss = dy.pickneglogsoftmax(logits, word)

    if not self.warned:
      sys.stderr.write('WARNING: compute_loss hacked to not include actual terminals.\n')
      self.warned = True
    if word != 0 and word != 1:
      probs = -dy.softmax(logits)
      left_prob = dy.pick(probs, 0)
      right_prob = dy.pick(probs, 1)
      loss = dy.log(1 - left_prob - right_prob)
    else:
      loss = dy.pickneglogsoftmax(logits, word)
    
    return loss

  def build_graph(self, sent):
    state = self.new_sent()

    losses = []
    for word in sent:
      loss = self.compute_loss(state, word)
      losses.append(loss)
      state = self.add_input(state, word)

    return dy.esum(losses)

def read_corpus(filename, vocab):
  corpus = []
  with open(filename) as f:
    for line in f:
      words = line.split()
      word_ids = [vocab.convert(word) for word in words]
      corpus.append(word_ids)
  return corpus

def main():
  print('Invoked as:', ' '.join(sys.argv), file=sys.stderr)
  parser = argparse.ArgumentParser()
  parser.add_argument('train_corpus')
  parser.add_argument('dev_corpus')
  parser.add_argument('--layers', type=int, default=1)
  parser.add_argument('--hidden_dim', type=int, default=128)
  parser.add_argument('--minibatch_size', type=int, default=1)
  parser.add_argument('--autobatch', action='store_true')
  parser.add_argument('--tied', action='store_true')
  parser.add_argument('--dropout', type=float, default=0.0)
  parser.add_argument('--output', type=str, default='')
  harness.add_optimizer_args(parser)
  args = parser.parse_args()

  if args.output == '':
    args.output = '/tmp/model%d' % random.randint(0, 0xFFFF)
  print('Output file:', args.output, file=sys.stderr)

  vocab = Vocabulary()
  train_corpus = read_corpus(args.train_corpus, vocab)
  vocab.frozen = True
  dev_corpus = read_corpus(args.dev_corpus, vocab) 

  print('Vocabulary size:', len(vocab), file=sys.stderr)

  pc = dy.ParameterCollection()
  optimizer = harness.make_optimizer(args, pc)
  model = TopDownDepLM(pc, vocab, args.layers, args.hidden_dim, args.hidden_dim, args.tied)
  print('Total parameters:', pc.parameter_count(), file=sys.stderr)

  harness.train(model, train_corpus, dev_corpus, optimizer, args)

if __name__ == '__main__':
  main()

