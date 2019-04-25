"""
Define common constants.
"""
TRAIN_JSON = 'train.json'
DEV_JSON = 'dev.json'
TEST_JSON = 'test.json'

GLOVE_DIR = 'dataset/glove'

EMB_INIT_RANGE = 1.0
MAX_LEN = 100

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'FAC': 2, 'GPE': 3, 'LOC': 4, 'ORG': 5, 'PER': 6, 'VEH': 7, 'WEA': 8}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'FAC': 2, 'GPE': 3, 'LOC': 4, 'ORG': 5, 'PER': 6, 'VEH': 7, 'WEA': 8}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14, 'CAUSE_OF_DEATH': 15, 'CITY': 16, 'COUNTRY': 17, 'CRIMINAL_CHARGE': 18, 'IDEOLOGY': 19, 'NATIONALITY': 20, 'RELIGION': 21, 'STATE_OR_PROVINCE': 22, 'TITLE': 23, 'URL': 24}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

# LABEL_TO_ID = {'no_relation': 0, 'ART': 1, 'GEN-AFF': 2, 'ORG-AFF': 3, 'PART-WHOLE': 4, 'PER-SOC': 5, 'PHYS': 6}  # ace03
LABEL_TO_ID = {'no_relation': 0, 'AT': 1, 'NEAR': 2, 'PART': 3, 'ROLE': 4, 'SOC': 5}  # ace03

SUB_LABEL_TO_ID = {'no_relation': 0, 'Artifact': 1, 'Business': 2, 'Citizen-Resident-Religion-Ethnicity': 3, 'Employment': 4, 'Family': 5, 'Founder': 6, 'Geographical': 7, 'Investor-Shareholder': 19, 'Lasting-Personal': 8, 'Located': 9, 'Membership': 10, 'Near': 11, 'Org-Location': 12, 'Ownership': 13, 'Sports-Affiliation': 14, 'Student-Alum': 15, 'Subsidiary': 16, 'User-Owner-Inventor-Manufacturer': 17}

INFINITY_NUMBER = 1e12
