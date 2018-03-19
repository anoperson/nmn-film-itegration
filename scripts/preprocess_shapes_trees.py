import h5py
import numpy
import os
import argparse
import json
from vr.preprocess import tokenize, encode, build_vocab
import sexpdata
from scripts.preprocess_questions import program_to_str, program_to_arity, program_to_depth

def extract_parse(p):
  if isinstance(p, sexpdata.Symbol): return p.value()
  elif isinstance(p, int): return str(p)
  elif isinstance(p, bool): return str(p).lower()
  elif isinstance(p, float): return str(p).lower()
  return tuple(extract_parse(q) for q in p)

def parse_tree(p):
  if "'" in p: p = "none"
  parsed = sexpdata.loads(p)
  extracted = extract_parse(parsed)
  return extracted

def layout_from_parsing(parse):
  if isinstance(parse, str): return ("_Find",)
  head = parse[0]
  if len(parse) > 2:  # fuse multiple tokens with "_And"
    assert(len(parse)) == 3
    below = ("_And", layout_from_parsing(parse[1]),
                 layout_from_parsing(parse[2]))
  else:
    below = layout_from_parsing(parse[1])
  if head == "is":
    module = "_Answer"
  elif head in ["above", "below", "left_of", "right_of"]:
    module = "_Transform"
  return (module, below)

#('_Answer', ('_And', ('_Find',), ('_Transform', ('_Find',))))

def layout_tree(module_layout):
  # Postorder traversal to generate Reverse Polish Notation
  tree = []
  def build_tree(cur):
    if isinstance(cur, str):
      tree.append({'function' : cur, 'value_inputs' : [], 'inputs' : []})
      return cur
    name = cur[0]
    node = {'function' : name, 'value_inputs' : [], 'inputs' : []}
    for m in cur[1:]:
      fn = build_tree(m)
      node['value_inputs'].append(fn)
      node['inputs'].append(len(tree)-1)
    tree.append(node)
    return name
  build_tree(module_layout)
  return tree

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', default='prefix',
                    choices=['chain', 'prefix', 'postfix'])
  parser.add_argument('--shapes_data', type=str,
                      help="Path to the SHAPES dataset")
  parser.add_argument('--size', type=str,
                      help="Which version of the training set to use")

  args = parser.parse_args()
  parts = ['train', 'val', 'test']
  part_prefixes = ['train.' + args.size, 'val', 'test']
  part_prefixes = [os.path.join(args.shapes_data, prefix)
                   for prefix in part_prefixes]

  for part, prefix in zip(parts, part_prefixes):
    image_path = prefix + '.input.npy'
    images = numpy.load(image_path)

    questions_path = prefix + '.query_str.txt'
    questions_encoded = []
    with open(questions_path) as src:
      questions = [str_ for str_ in src]
      if part == 'train':
        question_vocab = build_vocab(questions, delim=None)
      for qe in questions:
        tkn = tokenize(qe, delim=None)
        questions_encoded.append(encode(tkn, question_vocab, allow_unk=True))
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
      while len(qe) < max_question_length:
        qe.append(question_vocab['<NULL>'])
    
    answers_path = prefix + '.output'
    with open(answers_path) as src:
      answers = [1 if w.strip() == 'true' else 0 for w in src]
    
    programs_path = prefix + '.query'
    all_program_strs = []
    with open(programs_path) as src:
      for line in src:
        line = line.strip()
        program = layout_tree(layout_from_parsing(parse_tree(line)))
        program_str = program_to_str(program, args.mode)
        if program_str is not None:
          all_program_strs.append(program_str)
    if part == 'train':
      program_vocab = build_vocab(all_program_strs)
    
    programs_encoded = []
    programs_arities = []
    programs_depths = []
    
    with open(programs_path) as src:
      for line in src:
        line = line.strip()
        program = layout_tree(layout_from_parsing(parse_tree(line)))
        program_str = program_to_str(program, args.mode)
        program_tokens = tokenize(program_str, delim=None)
        program_encoded = encode(program_tokens, program_vocab, allow_unk=True)
        programs_encoded.append(program_encoded)
      
        programs_arities.append(program_to_arity(program, args.mode))
        programs_depths.append(program_to_depth(program, args.mode))
    
    if len(programs_encoded) > 0:
      max_program_length = max(len(x) for x in programs_encoded)
      for pe in programs_encoded:
        while len(pe) < max_program_length:
          pe.append(vocab['program_token_to_idx']['<NULL>'])
    
      max_program_arity_length = max(len(x) for x in programs_arities)
      for ar in programs_arities:
        while len(ar) < max_program_arity_length:
          ar.append(-1)
    
      max_program_depth_length = max(len(x) for x in programs_depths)
      for de in programs_depths:
        while len(de) < max_program_depth_length:
          de.append(-1)

      assert(max_program_length == max_program_arity_length) and (max_program_length == max_program_depth_length)
    
    # Create h5 file
    print('Writing output')
    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    programs_encoded = np.asarray(programs_encoded, dtype=np.int32)
    programs_arities = np.asarray(programs_arities, dtype=np.int32)
    programs_depths = np.asarray(programs_depths, dtype=np.int32)
    print(questions_encoded.shape)
    print(programs_encoded.shape)
    print(programs_arities.shape)
    print(programs_depths.shape)
    
    with h5py.File(part + '_features.h5', 'w') as f:
      features = images.transpose(0, 3, 1, 2)
      features_dataset = f.create_dataset('features', (features.shape), dtype=numpy.float32)
      features_dataset[:] = features
      
    with h5py.File(part + '_questions.h5', 'w') as f:
      f.create_dataset('questions', data=questions_encoded)
      
      image_idxs_dataset = f.create_dataset('image_idxs', (len(questions_encoded),), dtype=numpy.int32)
      image_idxs_dataset[:] = range(len(questions_encoded))
      
      if len(programs_encoded) > 0:
        f.create_dataset('programs', data=programs_encoded)
        f.create_dataset('programs_arities', data=programs_arities)
        f.create_dataset('programs_depths', data=programs_depths)
      
      if len(answers) > 0:
        f.create_dataset('answers', data=np.asarray(answers))

  with open('vocab.json', 'w') as f:
    json.dump({'question_token_to_idx': question_vocab,
                'program_token_to_idx': program_vocab,
                'answer_token_to_idx': {'false': 0, 'true': 1}},
              f)

if __name__ == '__main__':
  main()
