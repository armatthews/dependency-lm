from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import sys

import dynet_config
dynet_config.set(mem=4096)
import dynet as dy

from train import RNNLM
from utils import Vocabulary
from utils import read_corpus


def run_test_set(rnnlm, corpus, args):
  total_loss = 0.0
  word_count = 0
  losses = []
  for sent in corpus:
    if len(losses) == 0:
      dy.renew_cg(autobatching = args.autobatch)
      rnnlm.new_graph()
    loss = rnnlm.build_graph(sent)
    losses.append(loss)
    word_count += len(sent)

    if len(losses) == args.minibatch_size:
      total_loss += dy.esum(losses).scalar_value()
      losses = []
  if len(losses) > 0:
    total_loss += dy.esum(losses).scalar_value()
    losses = []

  print('Loss: %f total, %f per sent, %f per word' % (total_loss, total_loss / len(corpus), total_loss / word_count))
  sys.stdout.flush()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('model')
  parser.add_argument('vocab')
  parser.add_argument('corpus')
  parser.add_argument('--layers', type=int, default=1)
  parser.add_argument('--emb_dim', type=int, default=128)
  parser.add_argument('--hidden_dim', type=int, default=128)
  parser.add_argument('--minibatch_size', type=int, default=1)
  parser.add_argument('--autobatch', action='store_true')
  args = parser.parse_args()

  vocab = Vocabulary()
  with open(args.vocab) as f:
    for line in f:
      word = line.strip()
      vocab.convert(word)
  print('Loaded a vocabulary of size %d' % (len(vocab)))
  eos = vocab.convert('</s>')

  pc = dy.ParameterCollection()
  rnnlm = RNNLM(pc, args.layers, args.emb_dim, args.hidden_dim, len(vocab))
  pc.populate_from_textfile(args.model)
  #rnnlm, = dy.load(args.model, pc)
  print('Total parameters:', pc.parameter_count())

  """for i in range(100):
    rnnlm.new_graph()
    sampled_sent = rnnlm.sample(eos, 100)
    sampled_sent = [vocab.to_word(word_id) for word_id in sampled_sent]
    print(' '.join(sampled_sent))
    sys.stdout.flush()
  sys.exit(0)"""

  rnnlm.set_dropout(0.0)
  vocab.frozen = True
  corpus = read_corpus(args.corpus, vocab)
  run_test_set(rnnlm, corpus, args)

if __name__ == '__main__':
  main()
