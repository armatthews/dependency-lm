from __future__ import print_function
import argparse
import collections
import random
import sys

import dynet_config
dynet_config.set(mem="5120,5120,1,512", profiling=0)
dynet_config.set_gpu()
#dynet_config.set(mem=512,profiling=1)
#dynet_config.set(mem=32.0*1024)
import dynet as dy
import numpy as np

sys.path.append('/home/austinma/git/rnnlm/')
sys.path.append('../')
from utils import Vocabulary
from utils import MLP
import harness

ParserState = collections.namedtuple(
    'ParserState', 'open_constits, spine')

class TopDownDepLM:
  def __init__(self, pc, vocab, layers, state_dim, final_hidden_dim, tied, residual):
    self.vocab = vocab
    self.layers = layers
    self.state_dim = state_dim
    self.tied = tied
    self.residual = residual
    self.done_with_left = vocab.convert('</LEFT>')
    self.done_with_right = vocab.convert('</RIGHT>')
    vocab_size = len(self.vocab)

    self.pc = pc.add_subcollection()
    if not self.tied:
      self.word_embs = self.pc.add_lookup_parameters((vocab_size, state_dim))

    self.top_lstm = dy.LSTMBuilder(layers, state_dim, state_dim, self.pc)
    self.vertical_lstm = dy.LSTMBuilder(layers, state_dim, state_dim, self.pc)
    self.gate_mlp = MLP(self.pc, [2 * state_dim, state_dim, state_dim])
    self.open_constit_lstms = []
    self.debug_stack = []
    self.spine = []
    self.final_mlp = MLP(self.pc, [state_dim, final_hidden_dim, vocab_size])

    self.top_initial_state = [
        self.pc.add_parameters((state_dim,)) for _ in range(2 * layers)]
    self.open_initial_state = [
        self.pc.add_parameters((state_dim,)) for _ in range(2 * layers)]

  def set_dropout(self, r):
    self.dropout_rate = r
    self.top_lstm.set_dropout(r)
    self.vertical_lstm.set_dropout(r)
    self.final_mlp.set_dropout(r)

  def new_graph(self):
    # Do LSTM builders need reset?
    self.final_mlp.new_graph()
    self.gate_mlp.new_graph()

  def embed_word(self, word):
    if self.tied:
      word_embs = self.final_mlp.layers[-1].w
      word_emb = dy.select_rows(word_embs, [word])
      word_emb = dy.transpose(word_emb)
    else:
      word_emb = dy.lookup(self.word_embs, word)
    return word_emb

  def add_to_last(self, word):
    assert len(self.open_constit_lstms) > 0
    word_emb = self.embed_word(word)
    new_rep = self.open_constit_lstms[-1].add_input(word_emb)
    self.open_constit_lstms[-1] = new_rep

    self.debug_stack[-1].append(self.vocab.to_word(word))

  def pop_and_add(self, word):
    assert len(self.open_constit_lstms) >= 1
    word_emb = self.embed_word(word)
    child_state = self.open_constit_lstms[-1].add_input(word_emb)
    child_emb = child_state.output()
    self.open_constit_lstms.pop()
    if len(self.open_constit_lstms) > 0:
      self.open_constit_lstms[-1] = self.open_constit_lstms[-1].add_input(child_emb)
    self.spine.pop()

    self.debug_stack[-1].append(self.vocab.to_word(word))
    debug_child = self.debug_stack.pop()
    if len(self.debug_stack) > 0:
      self.debug_stack[-1].append(debug_child)

  def push(self, word):
    word_emb = self.embed_word(word)

    new_state = self.vertical_lstm.initial_state()
    new_state = new_state.set_s(self.open_initial_state)
    new_state = new_state.add_input(word_emb)
    self.open_constit_lstms.append(new_state)
    self.spine.append(word)

    self.debug_stack.append([self.vocab.to_word(word)])

  def add_input(self, state, word):
    word_emb = self.embed_word(word)
    if word == self.done_with_left:
      self.add_to_last(word)
    elif word == self.done_with_right:
      self.pop_and_add(word)
    else:
      self.push(word)
    #print('After:', self.debug_stack)
    assert len(self.debug_stack) == len(self.open_constit_lstms)
    return ParserState(self.open_constit_lstms, self.spine)

  def new_sent(self):
    new_state = self.vertical_lstm.initial_state()
    new_state = new_state.set_s(self.open_initial_state)
    self.open_constit_lstms = [new_state]
    self.spine = [-1]
    self.debug_stack = [[]]
    return ParserState(self.open_constit_lstms, self.spine)

  def debug_embed_vertical(self, vertical):
    state = self.vertical_lstm.initial_state()
    state = state.set_s(self.open_initial_state)
    for word in vertical:
      if type(word) == list:
        emb = self.debug_embed_vertical(word)
      else:
        emb = self.embed_word(self.vocab.convert(word))
      state = state.add_input(emb)
    return state.output()

  def debug_embed(self):
    top_state = self.top_lstm.initial_state()
    top_state = top_state.set_s(self.top_initial_state)

    assert len(self.open_constit_lstms) == len(self.debug_stack)
    for i, open_constit in enumerate(self.debug_stack):
      emb = self.debug_embed_vertical(open_constit)
      top_state = top_state.add_input(emb)
      alt = self.open_constit_lstms[i]
      #c = 'O' if np.isclose(emb.npvalue(), alt.output().npvalue()).all() else 'X'
      #print(c, emb.npvalue(), alt.output().npvalue())
      #assert np.isclose(emb.npvalue(), alt.output().npvalue()).all()
    #print()
    return top_state

  warned = False
  def compute_loss(self, state, word):
    top_state = self.top_lstm.initial_state()
    top_state = top_state.set_s(self.top_initial_state)
    assert len(state.open_constits) == len(state.spine)
    for open_constit, spine_word in zip(state.open_constits, state.spine):
      constit_emb = open_constit.output()
      if self.residual and spine_word != -1:
        spine_word_emb = self.embed_word(spine_word)
        if False:
          constit_emb += spine_word_emb
        else:
          inp = dy.concatenate([constit_emb, spine_word_emb])
          mask = self.gate_mlp(inp)
          mask = dy.logistic(mask)
          constit_emb = dy.cmult(1 - mask, constit_emb)
          constit_emb = constit_emb + dy.cmult(mask, spine_word_emb)
      top_state = top_state.add_input(constit_emb)
    #debug_top_state = self.debug_embed()
    #assert np.isclose(top_state.output().npvalue(), debug_top_state.output().npvalue()).all()

    logits = self.final_mlp(top_state.output())
    loss = dy.pickneglogsoftmax(logits, word)

    #if not self.warned:
    #  sys.stderr.write('WARNING: compute_loss hacked to not include actual terminals.\n')
    #  self.warned = True
    #if word != 0 and word != 1:
    #  probs = -dy.softmax(logits)
    #  left_prob = dy.pick(probs, 0)
    #  right_prob = dy.pick(probs, 1)
    #  loss = dy.log(1 - left_prob - right_prob)
    #else:
    #  loss = dy.pickneglogsoftmax(logits, word)
    
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
  parser.add_argument('--residual', action='store_true')
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
  model = TopDownDepLM(pc, vocab, args.layers, args.hidden_dim, args.hidden_dim, args.tied, args.residual)
  print('Total parameters:', pc.parameter_count(), file=sys.stderr)

  harness.train(model, train_corpus, dev_corpus, optimizer, args)

if __name__ == '__main__':
  main()

