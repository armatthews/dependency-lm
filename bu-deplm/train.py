from __future__ import print_function
import argparse
import random
import sys

import dynet_config
#dynet_config.set(mem=8*1024)
dynet_config.set_gpu()
import dynet as dy

sys.path.append('..')
from utils import Vocabulary
from utils import MLP

from harness import train

class BottomUpDepLM:
  def __init__(self, pc, action_vocab, word_vocab_size, rel_vocab_size, layers, hidden_dim, labelled=True, tied=False):
    self.labelled = labelled
    self.tied = tied
    self.action_vocab = action_vocab
    self.pc = pc.add_subcollection()
    action_vocab_size = len(action_vocab)

    if not self.tied:
      self.word_embs = self.pc.add_lookup_parameters((word_vocab_size, hidden_dim))
    self.action_mlp = MLP(self.pc, [hidden_dim, hidden_dim, action_vocab_size])
    self.word_mlp = MLP(self.pc, [hidden_dim, hidden_dim, word_vocab_size])

    self.combine_mlp = MLP(self.pc, [2 * hidden_dim, hidden_dim, hidden_dim])

    self.stack_lstm = dy.LSTMBuilder(layers, hidden_dim, hidden_dim, self.pc)
    self.initial_state_params = [self.pc.add_parameters((hidden_dim,)) for _ in range(2 * layers)]
    self.stack_embs = []

    if labelled:
      self.rel_embs = self.pc.add_lookup_parameters((rel_vocab_size, hidden_dim))
      self.rel_mlp = MLP(self.pc, [hidden_dim, hidden_dim, rel_vocab_size])

  def new_graph(self):
    self.action_mlp.new_graph()
    self.word_mlp.new_graph()
    self.combine_mlp.new_graph()
    if self.labelled:
      self.rel_mlp.new_graph()
    self.initial_state = [dy.parameter(p) for p in self.initial_state_params]

  def new_sent(self):
    self.stack_embs = []
    self.stack = []
    state = self.stack_lstm.initial_state()
    state = state.set_s(self.initial_state)
    self.stack_embs.append(state)

  def set_dropout(self, r):
    self.action_mlp.set_dropout(r)
    self.word_mlp.set_dropout(r)
    self.combine_mlp.set_dropout(r)
    self.stack_lstm.set_dropout(r)
    if self.labelled:
      self.rel_mlp.set_dropout(r)

  def combine(self, head, child, direction):
    head_and_child = dy.concatenate([head, child])
    return self.combine_mlp(head_and_child)

  def embed_word(self, word):
    if self.tied:
      word_embs = self.word_mlp.layers[-1].w
      word_emb = dy.select_rows(word_embs, [word])
      word_emb = dy.transpose(word_emb)
    else:
      word_emb = dy.lookup(self.word_embs, word)
    return word_emb

  def embed_stack_naive(self):
    state = self.stack_lstm.initial_state()
    state = state.set_s(self.initial_state)
    for item in self.stack:
      state = state.add_input(item)
    return state.output()

  def embed_stack(self):
    return self.stack_embs[-1].output()

  def pop(self):
    self.stack.pop()
    self.stack_embs.pop()

  def push(self, v):
    self.stack.append(v)
    state = self.stack_embs[-1]
    state = state.add_input(v)
    self.stack_embs.append(state)

  def shift(self, word):
    word_emb = self.embed_word(word)
    self.push(word_emb)

  def reduce_right(self):
    assert len(self.stack) >= 2
    head = self.stack[-1]
    child = self.stack[-2]
    self.pop()
    self.pop()
    combined = self.combine(head, child, 'right')
    self.push(combined)

  def reduce_left(self):
    assert len(self.stack) >= 2
    head = self.stack[-2]
    child = self.stack[-1]
    self.pop()
    self.pop()
    combined = self.combine(head, child, 'left')
    self.push(combined)

  def build_graph(self, sent):
    losses = []
    self.new_sent()
    for action, subtype in sent:
      action_str = self.action_vocab.to_word(action)

      # predict action
      hidden_state = self.embed_stack()
      action_logits = self.action_mlp(hidden_state)
      action_nlp = dy.pickneglogsoftmax(action_logits, action)

      loss = action_nlp
      if action_str == 'shift':
        word_logits = self.word_mlp(hidden_state)
        word_nlp = dy.pickneglogsoftmax(word_logits, subtype)
        loss += word_nlp
      elif self.labelled:
        rel_logits = self.rel_mlp(hidden_state)
        rel_nlp = dy.pickneglogsoftmax(rel_logits, subtype)
        #loss += rel_nlp
      losses.append(loss)

      # Do the reference action
      if action_str == 'shift':
        self.shift(subtype)
      elif action_str == 'right':
        self.reduce_right()
      elif action_str == 'left':
        self.reduce_left()
      else:
        assert 'Unknown action: %s' % action_str

    return dy.esum(losses)

def read_corpus(filename, action_vocab, terminal_vocab, rel_vocab):
  corpus = []
  with open(filename) as f:
    for line in f:
      words = line.split()
      words = [word.split('-', 1) for word in words]

      for word in words:
        if len(word) != 2:
          print(word)
          assert False

      sent_actions = []
      for action, subtype in words:
        action_id = action_vocab.convert(action)
        if action == 'shift':
          subtype_id = terminal_vocab.convert(subtype)
        else:
          #subtype_id = rel_vocab.convert(subtype)
          subtype_id = None
        sent_actions.append((action_id, subtype_id))
      corpus.append(sent_actions)
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
  args = parser.parse_args()

  if args.output == '':
    args.output = '/tmp/model%d' % random.randint(0, 0xFFFF)
  print('Output file:', args.output, file=sys.stderr)

  action_vocab = Vocabulary()
  terminal_vocab = Vocabulary()
  rel_vocab = Vocabulary()
  train_corpus = read_corpus(
      args.train_corpus, action_vocab, terminal_vocab, rel_vocab)
  action_vocab.frozen = True
  terminal_vocab.frozen = True
  rel_vocab.frozen = True
  dev_corpus = read_corpus(
      args.dev_corpus, action_vocab, terminal_vocab, rel_vocab)

  print('Vocabulary sizes:',
        len(action_vocab), len(terminal_vocab), len(rel_vocab),
        file=sys.stderr)

  pc = dy.ParameterCollection()
  optimizer = dy.SimpleSGDTrainer(pc, 1.0)
  model = BottomUpDepLM(pc, action_vocab, len(terminal_vocab), len(rel_vocab),
                        args.layers, args.hidden_dim, False, args.tied)
  print('Total parameters:', pc.parameter_count(), file=sys.stderr)

  train(model, train_corpus, dev_corpus, optimizer, args)

if __name__ == '__main__':
  main()
