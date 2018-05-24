from collections import namedtuple
Parse = namedtuple('Parse', 'words, tags, heads, rels')

def parse_to_string(parse):
  output = []
  for i, (word, tag, head, rel) in enumerate(zip(parse.words, parse.tags, parse.heads, parse.rels)):
    parts = [str(i + 1), word, '_', tag, tag, '_', str(head + 1), rel, '_', '_']
    output.append('\t'.join(parts))
  output.append('')
  return '\n'.join(output)

def parse(lines):
  words = []
  tags = []
  heads = []
  rels = []

  for line in lines:
    parts = line.split('\t')
    id = parts[0]
    if '-' in id or '.' in id:
      continue
    id = int(id) - 1 # subtract one so we index from 0
    word = parts[1]
    tag = parts[3]
    head = int(parts[6]) - 1
    rel = parts[7]
    assert len(words) == len(tags)
    assert len(words) == len(heads)
    assert len(words) == len(rels)
    assert len(words) == id
    words.append(word)
    tags.append(tag)
    heads.append(head)
    rels.append(rel)
  return Parse(words, tags, heads, rels)

def read_parses(stream):
  parse_lines = []
  while True:
    line = stream.readline()
    if not line:
      break

    line = line.strip()
    if not line:
      yield parse(parse_lines)
      parse_lines = []
    elif '\t' not in line:
      # Skip extraneous nonsense
      # e.g. logprobs from the sampler
      continue
    else:
      parse_lines.append(line)
