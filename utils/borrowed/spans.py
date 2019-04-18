__all__ = ['find_token_span']


def find_token_span(annotations, m_dict, offsetAdjust=0):
    start_char = m_dict['start_char'] - offsetAdjust
    end_char = m_dict['end_char'] - offsetAdjust
    span_start, span_end = None, None

    for i, sent in enumerate(annotations['sentences']):
        for j, token in enumerate(sent['tokens']):
            if start_char >= token['characterOffsetBegin'] and start_char <= token['characterOffsetEnd']:
                span_start = (i, j)
            if end_char >= token['characterOffsetBegin'] and end_char <= token['characterOffsetEnd']:
                span_end = (i, j)
        # end for
    # end for

    # assert span_start is not None and span_end is not None
    sentences = annotations['sentences']
    if span_start is None:
        span_start = find_nearest_token(sentences, start_char, location='start')
    if span_end is None:
        span_end = find_nearest_token(sentences, end_char, location='end')

    return (span_start, span_end)
# end def


def find_nearest_token(sentences, char_index, location='start'):
    '''
    In ACE04 corpus, some instances cannot find token span for a mention. Such instacnes usually occur in the beginning/end of a sentence, and will result in an assert error.
    This is a workaround to find the token which is nearest to the mention start/end in terms on character offset.
    '''
    # find sentence
    for idx, sent in enumerate(sentences):
        if sent['tokens'][0]['characterOffsetBegin'] <= char_index and sent['tokens'][-1]['characterOffsetEnd'] >= char_index:
            sent_idx = idx
    # end for

    # use the nearest token as alternative
    tokens = sentences[sent_idx]['tokens']
    if location == 'start':
        distance = [abs(tok['characterOffsetBegin'] - char_index) for tok in tokens]
    if location == 'end':
        distance = [abs(tok['characterOffsetEnd'] - char_index) for tok in tokens]

    tok_idx = distance.index(min(distance))
    return (sent_idx, tok_idx)
# end def





