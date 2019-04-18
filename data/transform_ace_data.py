'''
Description:
Transform ACE data with corenlp annotations to meet requirements of LSTM model.


History:
    Apri 18, 2019       Initial version.

Usage:
python -m data.transform_ace_data --input ../relation-extraction-ly-dev/data/ace05_json/ace05_511set/w_head_sys_0131/ace2005.511set.test.nlp.dual.json
python -m data.transform_ace_data --input ../relation-extraction-ly-dev/data/ace05_json/ace05_511set/w_head_sys_0131/ace2005.511set.validation.nlp.dual.json
python -m data.transform_ace_data --input ../relation-extraction-ly-dev/data/ace05_json/ace05_511set/w_head_sys_0131/w_gold_mentions/ace2005.511set.train.nlp.dual.json

'''
import json
from argparse import ArgumentParser, FileType
from utils.borrowed.spans import find_token_span
from utils.borrowed.relation_pair import RelationPairGeneratorTransformer

parser = ArgumentParser()
parser.add_argument('--input', type=FileType('r'), help='Input ace-style json data')
parser.add_argument('--output', type=FileType('w'), help='Dump processed data')
args = parser.parse_args()


def transform_one(relation_instance):
    paired_mentions = relation_instance.pop('paired_mentions')
    annotations = relation_instance.pop('corenlp_annotations')

    m1_span = find_token_span(annotations, paired_mentions['m1'], offsetAdjust=2)
    m2_span = find_token_span(annotations, paired_mentions['m2'], offsetAdjust=2)

    assert m1_span[0][0] == m1_span[1][0] == m2_span[0][0] == m2_span[1][0]

    target_sent = annotations['sentences'][m1_span[0][0]]

    stanford_deprel = [d['dep'] for d in target_sent['enhancedPlusPlusDependencies']]
    stanford_head = [d['governor'] for d in target_sent['enhancedPlusPlusDependencies']]
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


def main():

    ace_docs = [json.loads(line) for line in args.input]
    print('%d docments in %s' % (len(ace_docs), args.input.name))

    relation_instances = RelationPairGeneratorTransformer(ordered=True).transform(ace_docs)     # args are using the default
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

    for r in transformed:
        args.output.write(json.dumps(r, sort_keys=True))
        args.output.write('\n')
    print('Processed data saved to %s' % args.output.name)


if __name__ == '__main__':
    main()
