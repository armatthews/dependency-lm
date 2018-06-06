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

def add_optimizer_args(parser):
  parser.add_argument('--adam', action='store_true')

def make_optimizer(args, pc):
  if args.adam:
    return dy.AdamTrainer(pc, 2.e-4)
  return dy.SimpleSGDTrainer(pc, 1.0)


def run_dev_set(model, corpus, args):
  if len(corpus) == 0:
    return None

  total_loss = 0.0
  word_count = 0
  losses = []
  model.set_dropout(0.0)

  for sent in corpus:
    if len(losses) == 0:
      dy.renew_cg(autobatching=args.autobatch)
      model.new_graph()

    loss = model.build_graph(sent)
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
      total_loss / word_count, word_count),
      file=sys.stderr)
  return total_loss


def train(model, train_corpus, dev_corpus, optimizer, args):
  losses = []
  word_count = 0
  updates_done = 0
  best_dev_score = None
  learning_rate_changes = 0
  bad_devs_in_row = 0

  while True:
    random.shuffle(train_corpus)
    for sent in train_corpus:
      if len(losses) == 0:
        dy.renew_cg(autobatching=args.autobatch)
        model.new_graph()

      model.set_dropout(args.dropout)
      loss = model.build_graph(sent)
      word_count += len(sent)
      losses.append(loss)

      if len(losses) == args.minibatch_size:
        total_loss = dy.esum(losses)
        total_loss.forward()
        per_word = total_loss.scalar_value() / word_count
        per_sent = total_loss.scalar_value() / len(losses)
        print(per_word, per_sent, file=sys.stderr)
        total_loss.backward()
        if len(losses) == args.minibatch_size or True:
          optimizer.update()
          updates_done += 1
        losses = []
        word_count = 0

        if updates_done % 150 == 0:
          dev_score = run_dev_set(model, dev_corpus, args)
          if best_dev_score == None or dev_score < best_dev_score:
            bad_devs_in_row = 0
            best_dev_score = dev_score
            model.pc.save(args.output)
            print('Model saved!', file=sys.stderr)
          else:
            pass
            #bad_devs_in_row += 1
            #if bad_devs_in_row >= 3:
            #  f = (learning_rate_changes + 1) / (learning_rate_changes + 2)
            #  optimizer.learning_rate *= f
            #  learning_rate_changes += 1
            #  bad_devs_in_row = 0

    print('=== END EPOCH ===')
