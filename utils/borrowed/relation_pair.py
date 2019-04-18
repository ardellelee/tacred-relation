__all__ = ['RelationPairGeneratorTransformer']

'''
This file is borrowed from relation-extraction project, location: i2r/relation_extraction/transformers/relation_pair.py.
Main logic keeps equivalent. Some default parameter values are changed to fit the TACRED project.


History:
    Apr 18, 2019    Initial Version.

'''
from itertools import combinations
from itertools import permutations
from operator import itemgetter
import random

from sklearn.base import BaseEstimator, TransformerMixin


class RelationPairGeneratorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ordered=True, symmetric=True, gold_only=False, none_label='no_relation', symmetric_list=['PER-SOC'], tolerance=None, downsampling=False, ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.ordered = ordered
        self.symmetric = symmetric
        self.gold_only = gold_only
        self.none_label = none_label
        self.tolerance = tolerance
        self.downsampling = downsampling
        self.ratio = ratio
        self.symmetric_list = symmetric_list
        # print('ordered={}, symmetric={}, symmetric_list={}, gold_only={}, tolerance={}, downsampling={}, ratio={}'.format(self.ordered, self.symmetric, self.symmetric_list, self.gold_only, self.tolerance, self.downsampling, self.ratio))
    # end def

    def fit(self, instances):
        return self

    def transform(self, instances):
        relation_instances = []
        for instance in instances:
            r = self.findRelationsInDocument(doc=instance)
            relation_instances.extend(r)
        # print(len(relation_instances))

        if (not self.gold_only) and self.downsampling:
            relation_instances = self.downsampleNegativeInstances(relation_instances, self.ratio)

        return relation_instances
    # end def

    def _find_relation(self, relations, m1_id, m2_id):
        for r in relations:
            if r['arg1_id'] == m1_id and r['arg2_id'] == m2_id:
                return r
            elif not self.ordered and r['arg2_id'] == m1_id and r['arg1_id'] == m2_id:
                return r if self.symmetric else False
        # end for
        return False
    # end def

    def _find_relation_symmetric(self, relations, m1_id, m2_id):    # @Jan14, model argument order for assymmetric relations
        for r in relations:
            if r['type'] in self.symmetric_list:
                symmetric_relation = True
            else:
                symmetric_relation = False
            match = r['arg1_id'] == m1_id and r['arg2_id'] == m2_id
            reverse_match = r['arg2_id'] == m1_id and r['arg1_id'] == m2_id

            if match:
                return r
            elif not self.ordered and reverse_match:
                if symmetric_relation:
                    return r
                else:
                    new_r = r.copy()
                    new_r['type'] = new_r['type'] + '_R'
                    return new_r
        # end for
        return False
    # end def

    def findRelationsInSentence(self, mentions, relations):
        if self.ordered:
            def combination_func(L): return permutations(L, 2)
        else:
            def combination_func(L): return combinations(L, 2)

        for m1, m2 in combination_func(mentions):
            if self.symmetric:
                relation = self._find_relation(relations, m1['id'], m2['id'])
            else:
                relation = self._find_relation_symmetric(relations, m1['id'], m2['id'])     # @Jan 14
            if self.gold_only and relation is False:
                continue

            d = {}
            d['paired_mentions'] = dict(m1=m1, m2=m2)
            d['relation'] = relation['type'] if relation else self.none_label
            d['subrelation'] = relation['subtype'] if relation else self.none_label
            d['id'] = relation['id'] if relation else ''        # add relation id for TACRED code
            yield d
    # end def

    def filterInstanceWithinTolerance(self, relation_instances, tolerance):
        for r in relation_instances:
            m1_id = r['paired_mentions']['m1']['id_in_sentence']
            m2_id = r['paired_mentions']['m2']['id_in_sentence']
            # if r['relation'] is self.none_label and abs(m1_id - m2_id) > tolerance:  # @Jan 11, '>=' to '>'
            if abs(m1_id - m2_id) > tolerance:  # @Jan 15,  apply to all relation types including gold and None
                relation_instances.remove(r)
        return relation_instances
    # end def

    def findRelationsInDocument(self, doc=None):
        docid = doc.get('docid', '')
        # content = doc.get('content', '')
        mentions = doc.get('mentions', [])
        relations = doc.get('relations', [])
        corenlp_annotations = doc.get('corenlp_annotations', [])
        sentences = corenlp_annotations.get('sentences', [])
        sentSplitPoints = [0] + [sent['tokens'][-1]['characterOffsetEnd'] for sent in sentences]

        relation_instances = []
        for i in range(len(sentences)):
            sent_mentions = [m for m in mentions if (m['end_char'] <= sentSplitPoints[i + 1] and m['start_char'] >= sentSplitPoints[i])]
            for m in sent_mentions:
                m['id_in_sentence'] = sent_mentions.index(m)
            if sent_mentions and sent_mentions is not []:
                _relation_instances = list(self.findRelationsInSentence(mentions=sent_mentions, relations=relations))
                if self.tolerance:
                    _relation_instances = self.filterInstanceWithinTolerance(_relation_instances, tolerance=self.tolerance)
                for d in _relation_instances:
                    d.update({'docid': docid})
                    d.update({'corenlp_annotations': corenlp_annotations})

                # end for
                relation_instances.extend(_relation_instances)
            # end if
        # end for
        return relation_instances

    def downsampleNegativeInstances(self, relation_instances, ratio):
        negative_indices = []
        positive_indices = []
        for i, r in enumerate(relation_instances):
            if r['relation'] == 'None':
                negative_indices.append(i)
            else:
                positive_indices.append(i)
        negative_indices = negative_indices[::ratio]
        positive_instances = list(itemgetter(*positive_indices)(relation_instances))
        negative_instances = list(itemgetter(*negative_indices)(relation_instances))
        print('positive:{}, negative:{}'.format(len(positive_instances), len(negative_instances)))
        new_relation_instances = positive_instances + negative_instances
        random.shuffle(new_relation_instances)
        return new_relation_instances
    # end def

# end class


class RelationPairGeneratorTransformer03(BaseEstimator, TransformerMixin):

    def __init__(self, ordered=True, symmetric=False, gold_only=False, none_label='None', tolerance=2, downsampling=False, ratio=1, **kwags):
        # super().__init__(**kwargs)
        self.ordered = ordered
        self.symmetric = symmetric
        self.gold_only = gold_only
        self.none_label = none_label
        self.tolerance = tolerance
        self.downsampling = downsampling
        self.ratio = ratio
        self.symmetric_list = ['RELATIVE-LOCATION', 'ASSOCIATE', 'OTHER-RELATIVE', 'OTHER-PROFESSIONAL', 'SIBLING', 'SPOUSE']
        print('gold_only={}, tolerance={}, downsampling={}, ratio={}'.format(self.gold_only, self.tolerance, self.downsampling, self.ratio))
    # end def

    def fit(self, instances):
        return self

    def transform(self, instances):
        relation_instances = []
        for instance in instances:
            r = self.findRelationsInDocument(doc=instance)
            relation_instances.extend(r)
            # end for
        # print(len(relation_instances))

        if (not self.gold_only) and self.downsampling:
            relation_instances = self.downsampleNegativeInstances(relation_instances, self.ratio)

        relation_instances = self.distinguishTwoOtherSubtype(relation_instances)  # @Nov29
        return relation_instances
    # end def

    def distinguishTwoOtherSubtype(self, relation_instances):
        for r in relation_instances:
            if r['subrelation'] == 'OTHER' or r['subrelation'] == 'OTHER_R':
                r['subrelation'] = r['relation'] + '_' + r['subrelation']
        # end for
        return relation_instances
    # end def

    def _find_relation(self, relations, m1_id, m2_id):
        for r in relations:
            if r['arg1_id'] == m1_id and r['arg2_id'] == m2_id:
                return r
            elif not self.ordered and r['arg2_id'] == m1_id and r['arg1_id'] == m2_id:
                return r if self.symmetric else False
        # end for
        return False
    # end def

    def _find_relation_symmetric(self, relations, m1_id, m2_id):  # @Nov 14. distinguish 6 symetric relations and 18 non-symmetric
        for r in relations:
            if r['subtype'] in self.symmetric_list:
                self.symmetric = True
            else:
                self.symmetric = False

            match = r['arg1_id'] == m1_id and r['arg2_id'] == m2_id
            reverse_match = r['arg2_id'] == m1_id and r['arg1_id'] == m2_id

            if match:
                return r
            elif not self.ordered and reverse_match:
                if self.symmetric:
                    return r
                else:
                    new_r = r.copy()
                    new_r['subtype'] = new_r['subtype'] + '_R'
                    return new_r
        # end for
        return False
    # end def

    def findRelationsInSentence(self, mentions, relations):
        if self.ordered:
            def combination_func(L): return permutations(L, 2)
        else:
            def combination_func(L): return combinations(L, 2)

        for m1, m2 in combination_func(mentions):
            # relation = self._find_relation(relations, m1['entity_id'], m2['entity_id'])
            relation = self._find_relation_symmetric(relations, m1['entity_id'], m2['entity_id'])  # @Nov 14
            if self.gold_only and relation is False:
                continue

            d = {}
            d['paired_mentions'] = dict(m1=m1, m2=m2)
            d['relation'] = relation['type'] if relation else self.none_label
            d['subrelation'] = relation['subtype'] if relation else self.none_label
            yield d
    # end def

    def filterInstanceWithinTolerance(self, relation_instances, tolerance):
        for r in relation_instances:
            m1_id = r['paired_mentions']['m1']['id_in_sentence']
            m2_id = r['paired_mentions']['m2']['id_in_sentence']
            # if r['relation'] is self.none_label and abs(m1_id - m2_id) >= tolerance:
            if abs(m1_id - m2_id) > tolerance:              # @Jan 15, apply to all relation types including gold and None
                relation_instances.remove(r)
        return relation_instances
    # end def

    # @Nov 14  follow GD's method
    def filterImplicitRelations(self, relations):
        filtered = [r for r in relations if r['relation_class'] == 'EXPLICIT']
        return filtered

    def findRelationsInDocument(self, doc=None):
        docid = doc.get('docid', '')
        source = doc.get('source', '')
        # content = doc.get('content', '')
        mentions = doc.get('mentions', [])
        relations = self.filterImplicitRelations(doc.get('relations', []))
        corenlp_annotations = doc.get('corenlp_annotations', [])
        spacy_annotations = doc.get('nlp', [])
        sentences = corenlp_annotations.get('sentences', [])
        sentSplitPoints = [0] + [sent['tokens'][-1]['characterOffsetEnd'] for sent in sentences]

        relation_instances = []
        for i in range(len(sentences)):
            sent_mentions = [m for m in mentions if (m['end_char'] <= sentSplitPoints[i + 1] and m['start_char'] >= sentSplitPoints[i])]
            for m in sent_mentions:
                m['id_in_sentence'] = sent_mentions.index(m)
            if sent_mentions and sent_mentions is not []:
                _relation_instances = list(self.findRelationsInSentence(mentions=sent_mentions, relations=relations))
                _relation_instances = self.filterInstanceWithinTolerance(_relation_instances, tolerance=self.tolerance)
                # print('_relation_instances:', len(_relation_instances))
                for d in _relation_instances:
                    d.update({'corenlp_annotations': corenlp_annotations})
                    d.update({'nlp': spacy_annotations})
                    d.update({'source': source})
                    d.update({'docid': docid})
                # end for
                relation_instances.extend(_relation_instances)
            # end if
        # end for
        return relation_instances

    def downsampleNegativeInstances(self, relation_instances, ratio):
        negative_indices = []
        positive_indices = []
        for i, r in enumerate(relation_instances):
            if r['relation'] == 'None':
                negative_indices.append(i)
            else:
                positive_indices.append(i)
        negative_indices = negative_indices[::ratio]
        positive_instances = list(itemgetter(*positive_indices)(relation_instances))
        negative_instances = list(itemgetter(*negative_indices)(relation_instances))
        print('positive:{}, negative:{}'.format(len(positive_instances), len(negative_instances)))
        new_relation_instances = positive_instances + negative_instances
        random.shuffle(new_relation_instances)
        return new_relation_instances
    # end def
# end class
