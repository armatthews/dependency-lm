import argparse
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.spatial

def find_intersection(line1, line2):
  if line1[0] == line2[0]:
    return None
  return -(line1[1] - line2[1]) / (line1[0] - line2[0])

def plot_lines(hyps):
  intersections = []
  for i in range(len(hyps)):
    for j in range(i + 1, len(hyps)):
      xi = find_intersection(hyps[i], hyps[j])
      intersections.append(xi)

  intersections = [i for i in intersections if i is not None]
  intersections = sorted(intersections)

  x_min = intersections[0]
  x_max = intersections[-1]
  dist = x_max - x_min
  x_min -= 0.1 * dist
  x_max += 0.1 * dist

  for m, b in hyps:
    y1 = m * x_min + b
    y2 = m * x_max + b
    plt.plot((x_min, x_max), (y1, y2), 'k-')

  plt.show()

def find_upper_envelope(points, scores):
  # Start by solving the dual problem: the convex hull
  # If there are two or fewer points, all of them are on the hull
  # (and the scipy library barfs), so we special case this.
  assert len(points) > 0
  if len(points) == 1:
    vertices = [0]
  elif len(points) == 2:
    if points[0][0] > points[1][0]:
      vertices = [0, 1]
    else:
      vertices = [1, 0]
  else:
    hull = scipy.spatial.ConvexHull(points)
    vertices = hull.vertices
    #for i in range(len(vertices)):
    #  print '  %d:' % i, points[vertices[i]], scores[vertices[i]], '(%d)' % (vertices[i])

  # Now start with the point with the highest slope
  assert len(points) > 0
  highest_slope_index = 0
  for i in range(1, len(vertices)):
    if points[vertices[i]][0] > points[vertices[highest_slope_index]][0]:
      highest_slope_index = i
 
  # Take lines on the convex hull in order of decreasing slope
  # When the slopes start increasing again, we've hit the bottom
  # part of the convex hull and can quit
  envelope = []
  prev_slope = float('inf')
  for i in range(0, len(vertices)):
    j = (highest_slope_index + i) % len(vertices)
    point = points[vertices[j]]
    slope = point[0]
    if slope > prev_slope:
      break
    envelope.append(vertices[j])
    prev_slope = slope

  # Reverse the list, so we get points with increasing slopes instead of decreasing
  return list(reversed(envelope))

def find_intersections(lines):
  intersections = []
  for i in range(1, len(lines)):
    line1 = lines[i - 1]
    line2 = lines[i]
    intersection = find_intersection(line1, line2)
    intersections.append(intersection)
  assert len(intersections) == len(lines) - 1
  return intersections

def combine(scores1, intersections1, scores2, intersections2):
  assert len(scores1) == len(intersections1) + 1
  assert len(scores2) == len(intersections2) + 1

  i, j = 0, 0
  new_scores = []
  new_scores.append(scores1[i] + scores2[j])
  new_intersections = []
  while i < len(intersections1) or j < len(intersections2):
    if i >= len(intersections1):
      new_intersections.append(intersections2[j])
      j += 1
      new_scores.append(scores1[i] + scores2[j])
    elif j >= len(intersections2):
      new_intersections.append(intersections1[i])
      i += 1
      new_scores.append(scores1[i] + scores2[j])
    elif intersections1[i] < intersections2[j]:
      new_intersections.append(intersections1[i])
      i += 1
      new_scores.append(scores1[i] + scores2[j])
    elif intersections2[j] < intersections1[i]:
      new_intersections.append(intersections2[j])
      j += 1
      new_scores.append(scores1[i] + scores2[j])
    else:
      assert intersections1[i] == intersections2[j]
      new_intersections.append(intersections1[i])
      i += 1
      j += 1
      new_scores.append(scores1[i] + scores2[j])
  return new_scores, new_intersections

def uniqify(features, scores):
  hyps = zip(features, scores)
  hyps = set(hyps)
  hyps = list(hyps)
  return [hyp[0] for hyp in hyps], [hyp[1] for hyp in hyps]

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int)
args = parser.parse_args()

overall_scores = [0.0]
overall_intersections = []

dims = None
features = []
scores = []
for line in sys.stdin:
  parts = [float(part) for part in line.split()]
  assert len(parts) >= 3
  feats = tuple(parts[:-1])
  feats = tuple(-f for f in feats)
  if dims == None:
    dims = len(feats)
  else:
    assert len(feats) == dims

  features.append(feats)
  scores.append(parts[-1])
  
  if len(features) == args.n:
    features, scores = uniqify(features, scores)
    assert len(features) == len(scores)
    assert len(features) > 0
    #print 'Features:', features
    #plot_lines(features)
    #print 'Scores:', scores
    envelope = find_upper_envelope(features, scores)
    #print 'Envelope indices:', envelope
    envelope_lines = [features[i] for i in envelope]
    envelope_scores = [scores[i] for i in envelope]
    intersections = find_intersections(envelope_lines)
    #print 'Envelope lines:', envelope_lines
    #print 'Envelope scores:', envelope_scores
    #print 'Intersection points:', intersections
    overall_scores, overall_intersections = combine(
        envelope_scores, intersections, overall_scores, overall_intersections)
    features = []
    scores = []

#print overall_scores
#print overall_intersections

best_i = max(range(len(overall_scores)), key=lambda i: overall_scores[i])
#print best_i, len(overall_scores)
best_score = overall_scores[best_i]
if best_i == 0:
  best_lambda = float('-inf')
elif best_i == len(overall_scores) - 1:
  best_lambda = float('inf')
else:
  best_lambda = overall_intersections[best_i - 1] / 2.0 + overall_intersections[best_i] / 2.0
print 'Best score:', best_score
print 'Best lambda:', best_lambda
