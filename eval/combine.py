import argparse
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.spatial

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int)
parser.add_argument('w', type=float)
args = parser.parse_args()
dims = None
features = []
scores = []

for line in sys.stdin:
  parts = [float(part) for part in line.split()]
  assert len(parts) >= 3
  feats = tuple(parts[:-1])
  feats = [-f for f in feats]
  if dims == None:
    dims = len(feats)
  else:
    assert len(feats) == dims

  features.append(feats)
  scores.append(parts[-1])
  
  if len(features) == args.n:
    best = None
    for f, s in zip(features, scores):
      model_score = args.w * f[0] + f[1]
      if best is None or model_score > best[0]:
        best = (model_score, s)
    assert best is not None
    print best[1]
    features = []
    scores = []
