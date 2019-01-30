from collections import Counter
import torch

def vectorize(ex,model, single_answer=False):
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict

    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])

    if len(feature_dict) > 0:
        plain_features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        plain_features = None

    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                plain_features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                plain_features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                plain_features[i][feature_dict['in_question_lemma']] = 1.0

    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                plain_features[i][feature_dict[f]] = 1.0

    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                plain_features[i][feature_dict[f]] = 1.0

    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            plain_features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    if single_answer:
        start = torch.FloatTensor(1).fill_(ex['label'])
    else:
        start = [ex['label']]

    return document, question, plain_features, start,  ex['id'], ex['document'], ex['question']

def batchify(batch):

    docs = [ex[0] for ex in batch]
    questions = [ex[1] for ex in batch]
    features = [ex[2] for ex in batch]
    ids = [ex[4] for ex in batch]
    doc_real = [ex[5] for ex in batch]
    que_real = [ex[6] for ex in batch]

    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)


    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :d.size(0)].copy_(features[i])

    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)


    if torch.is_tensor(batch[0][3]):
        y_s = torch.cat([ex[3] for ex in batch])

    else:
        y_s = [ex[3] for ex in batch]


    return x1, x1_f, x1_mask, x2, x2_mask, y_s, ids, que_real, doc_real

