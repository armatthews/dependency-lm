from __future__ import print_function
import argparse
import random
import sys

import dynet_config
dynet_config.set(mem=6*1024)
dynet_config.set_gpu()
import dynet as dy

sys.path.append('/home/austinma/git/rnnlm/')
sys.path.append('..')
from utils import Vocabulary
from train import BottomUpDepLM
from train import read_corpus
from utils import run_test_set


def main():
  print('Invoked as:', ' '.join(sys.argv), file=sys.stderr)
  parser = argparse.ArgumentParser()
  parser.add_argument('model')
  parser.add_argument('train_corpus')
  parser.add_argument('test_corpus')
  parser.add_argument('--layers', type=int, default=1)
  parser.add_argument('--hidden_dim', type=int, default=128)
  parser.add_argument('--minibatch_size', type=int, default=1)
  parser.add_argument('--autobatch', action='store_true')
  parser.add_argument('--tied', action='store_true')
  parser.add_argument('--sent_level', action='store_true')
  args = parser.parse_args()

  action_vocab = Vocabulary()
  terminal_vocab = Vocabulary()
  rel_vocab = Vocabulary()
  train_corpus = read_corpus(
      args.train_corpus, action_vocab, terminal_vocab, rel_vocab)
  action_vocab.frozen = True
  terminal_vocab.frozen = True
  rel_vocab.frozen = True
  test_corpus = read_corpus(
      args.test_corpus, action_vocab, terminal_vocab, rel_vocab)

  print('Vocabulary sizes:',
        len(action_vocab), len(terminal_vocab), len(rel_vocab),
        file=sys.stderr)

  pc = dy.ParameterCollection()
  model = BottomUpDepLM(pc, action_vocab, len(terminal_vocab), len(rel_vocab),
                        args.layers, args.hidden_dim, False, args.tied)
  pc.populate_from_textfile(args.model)
  print('Total parameters:', pc.parameter_count(), file=sys.stderr)

  run_test_set(model, test_corpus, args)

if __name__ == '__main__':
  main()
