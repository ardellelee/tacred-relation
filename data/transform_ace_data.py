'''
Description:
Transform ACE data with corenlp annotations to meet requirements of LSTM model.


History:
    Apri 18, 2019       Initial version.

'''
import json
from argparse import ArgumentParser, FileType
from utils.borrowed.spans import find_token_span
from utils.borrowed.relation_pair import RelationPairGeneratorTransformer
from utils.borrowed.relation_pair import RelationPairGeneratorTransformer03

parser = ArgumentParser()
parser.add_argument('--input', type=FileType('r'), help='Input ace-style json data')
parser.add_argument('--output', type=FileType('w'), help='Dump processed data')
parser.add_argument('--mode', type=str, default='stat', help='Dump processed data')
args = parser.parse_args()


def transform_one(relation_instance):
  paired_mentions = relation_instance.pop('paired_mentions')
  annotations = relation_instance.pop('corenlp_annotations')

  m1_span = find_token_span(annotations, paired_mentions['m1'], offsetAdjust=2)
  m2_span = find_token_span(annotations, paired_mentions['m2'], offsetAdjust=2)

  assert m1_span[0][0] == m1_span[1][0] == m2_span[0][0] == m2_span[1][0]

  target_sent = annotations['sentences'][m1_span[0][0]]

  stanford_deprel = [d['dep'] for d in target_sent['basicDependencies']]
  stanford_head = [d['governor'] for d in target_sent['basicDependencies']]
  stanford_ner = [d['ner'] for d in target_sent['tokens']]
  stanford_pos = [d['pos'] for d in target_sent['tokens']]
  token = [d['originalText'] for d in target_sent['tokens']]

  transformed = dict(docid=relation_instance['docid'],
                     id=relation_instance['id'],
                     relation=relation_instance['relation'],
                     subrelation=relation_instance['subrelation'],
                     subj_start=m1_span[0][1],
                     subj_end=m1_span[1][1],
                     obj_start=m2_span[0][1],
                     obj_end=m2_span[0][1],
                     subj_type=paired_mentions['m1']['entity_type'],
                     obj_type=paired_mentions['m2']['entity_type'],
                     stanford_deprel=stanford_deprel,
                     stanford_head=stanford_head,
                     stanford_ner=stanford_ner,
                     stanford_pos=stanford_pos,
                     token=token,
                     )

  return transformed


def count_stat(values):
  frequency = {}
  for v in set(values):
    frequency[v] = values.count(v)
  frequency = dict(sorted(frequency.items()))
  return frequency


def main():

  ace_docs = [json.loads(line) for line in args.input]
  print('%d docments in %s' % (len(ace_docs), args.input.name))

  # relation_instances = RelationPairGeneratorTransformer(ordered=True).transform(ace_docs)     # args are using the default
  relation_instances = RelationPairGeneratorTransformer03(ordered=True).transform(ace_docs)     # args are using the default
  print('%d relation instances generated.' % len(relation_instances))

  n_error = 0
  transformed = []
  for r in relation_instances:
    try:
      transformed.append(transform_one(r))
    except Exception:
      n_error += 1
  # end for
  if n_error:
    print('Skipped %d error instances.' % n_error)

  if args.mode == 'stat':
    print('Relation:\n', count_stat([t['relation'] for t in transformed]))
    print('Subrelation:\n', count_stat([t['subrelation'] for t in transformed]))
    print('Subject:\n', count_stat([t['subj_type'] for t in transformed]))
    print('Object:\n', count_stat([t['obj_type'] for t in transformed]))

    all_ner, all_pos, all_deprel = [], [], []
    for t in transformed:
      all_ner += t['stanford_ner']
      all_pos += t['stanford_pos']
      all_deprel += t['stanford_deprel']
    print('NER:\n', count_stat(all_ner))
    print('POS:\n', count_stat(all_pos))
    print('Deprel:\n', count_stat(all_deprel))

  elif args.mode == 'write':
    json.dump(transformed, args.output)
    print('Processed data saved to %s' % args.output.name)


if __name__ == '__main__':
  main()
