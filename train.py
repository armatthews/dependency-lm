from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import sys

import dynet_config
dynet_config.set(mem=4096)
dynet_config.set_gpu()
import dynet as dy

from model import RNNLM
from model import Vocabulary
from model import read_corpus


def run_dev_set(rnnlm, corpus, args):
  total_loss = 0.0
  word_count = 0
  losses = []
  rnnlm.set_dropout(0.0)
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

  print('Dev loss: %f total, %f per sent (%d), %f per word (%d)' % (
      total_loss,
      total_loss / len(corpus), len(corpus),
      total_loss / word_count, word_count))
  sys.stdout.flush()
  return total_loss


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
  corpus = read_corpus(args.corpus, vocab)
  dev_corpus = read_corpus(args.dev_corpus, vocab)
  print('Vocab size:', len(vocab), file=sys.stderr)

  with open(args.output + '.vocab', 'w') as f:
    for word in vocab.i2w:
      print(word, file=f)

  pc = dy.ParameterCollection()
  optimizer = dy.SimpleSGDTrainer(pc)
  rnnlm = RNNLM(pc, args.layers, args.emb_dim, args.hidden_dim, len(vocab))
  print('Total parameters:', pc.parameter_count())

  losses = []
  word_count = 0
  updates_done = 0
  best_dev_score = None
  learning_rate_changes = 0

  while True:
    random.shuffle(corpus)
    for sent in corpus:
      if len(losses) == 0:
        dy.renew_cg(autobatching=args.autobatch)
        rnnlm.new_graph()

      rnnlm.set_dropout(args.dropout)
      loss = rnnlm.build_graph(sent)
      word_count += len(sent)
      losses.append(loss)

      if len(losses) == args.minibatch_size:
        total_loss = dy.esum(losses)
        total_loss.forward()
        per_word = total_loss.scalar_value() / word_count
        per_sent = total_loss.scalar_value() / len(losses)
        print(per_word, per_sent)
        sys.stdout.flush()
        total_loss.backward()
        optimizer.update()
        losses = []
        word_count = 0
        updates_done += 1

        if updates_done % 50 == 0:
          print(sample_sentence(rnnlm, vocab))
          sys.stdout.flush()

        if updates_done % 150 == 0:
          dev_score = run_dev_set(rnnlm, dev_corpus, args)
          if best_dev_score == None or dev_score < best_dev_score:
            best_dev_score = dev_score
            pc.save(args.output)
            print('Model saved!')
            sys.stdout.flush()
          else:
            f = (learning_rate_changes + 1) / (learning_rate_changes + 2)
            optimizer.learning_rate *= f
            learning_rate_changes += 1

    print('=== END EPOCH ===')

if __name__ == '__main__':
  main()
