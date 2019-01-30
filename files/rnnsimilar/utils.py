import json
import logging
import string
import regex as re
import time

from collections import Counter
from .data import Dictionary

logger = logging.getLogger(__name__)


def load_data(args, filename, mode):
    with open(filename) as f:
        examples = [json.loads(line) for line in f]


    if args.uncased_question or args.uncased_doc:
        for ex in examples:
            if args.uncased_question:
                ex['question'] = [w.lower() for w in ex['question']]
            if args.uncased_doc:
                ex['document'] = [w.lower() for w in ex['document']]


    examples = [ex for ex in examples if len(ex['document']) > 0]

    #to write a args case here
    # if mode == 'Train':
    #     examples = [ex for ex in examples if ("max_entity_left" in ex['question']) or ("entity_left" in ex['question']) or (("wikipage_hum" in ex['question'])) or ("wikipage_loc" in ex['question']) or ("wikipage_enty" in ex['question']) or ("wikipage_num" in ex['question']) or ("wikipage_desc" in ex['question']) or ("wikipage_abbr" in ex['question'])]

    return examples

def load_text(filename):
    with open(filename) as f:
        examples = json.load(f)['data']

    texts = {}
    for article in examples:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                texts[qa['id']] = paragraph['context']
    return texts


def load_answers(filename):
    with open(filename) as f:
        examples = json.load(f)['data']

    ans = {}
    for article in examples:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                ans[qa['id']] = list(map(lambda x: x['text'], qa['answers']))
    return ans


def index_embedding_words(embedding_file):
    words = set()
    with open(embedding_file) as f:
        for line in f:
            w = line.rstrip().split(' ')[0]
            w = w.lower()
            words.add(w)
            w = Dictionary.normalize(line.rstrip().split(' ')[0])
            words.add(w)
    return words


def load_words(args, examples):
    def _insert(iterable):
        for w in iterable:
            vv = Dictionary.normalize(w.lower())
            w = w.lower()
            tt = vv

            if valid_words and w not in valid_words:
                if valid_words and vv not in valid_words:
                    if valid_words and tt not in valid_words:
                        continue

            words.add(w)

    if args.restrict_vocab and args.embedding_file:
        logger.info('Restricting to words in %s' % args.embedding_file)
        valid_words = index_embedding_words(args.embedding_file)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None

    words = set()
    for ex in examples:
        _insert(ex['question'])
        _insert(ex['document'])

    special_tokens = ["wikipage_hum", "wikipage_loc", "wikipage_enty", "wikipage_num", "wikipage_desc", "wikipage_abbr", "max_entity_left", "entity_left",
                      "wikipage_entity", "max_entity_hum", "max_entity_loc", "max_entity_enty", "max_entity_num", "max_entity_desc", "max_entity_abbr"]

    for items in special_tokens:
        words.add(items)

    return words



def build_word_dict(args, examples):
    word_dict = Dictionary()
    for w in load_words(args, examples):
        word_dict.add(w)
    return word_dict

def top_question_words(args, examples, word_dict):
    word_count = Counter()

    for ex in examples:
        for w in ex['question']:
            w = Dictionary.normalize(w)
            if w in word_dict:
                word_count.update([w])

    return word_count.most_common(args.tune_partial)


def build_feature_dict(args, examples):
    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}

    # Exact match features
    if args.use_in_question:

        _insert('in_question')
        _insert('in_question_uncased')
        if args.use_lemma:
            _insert('in_question_lemma')

    # Part of speech tag features
    if args.use_pos:
        for ex in examples:
            for w in ex['pos']:
                _insert('pos=%s' % w)

    # Named entity tag features
    if args.use_ner:
        for ex in examples:
            for w in ex['ner']:
                _insert('ner=%s' % w)

    # Term frequency feature
    if args.use_tf:

        _insert('tf')

    return feature_dict



def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def strip(text):
        return text.rstrip()

    return white_space_fix(remove_articles(remove_punc(lower(strip(s)))))


def f1_score(prediction, pred_score, ground_truth, idss, miss, corrects):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return (0, miss,corrects)
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return (f1, miss, corrects)


def exact_match_score(prediction, pred_score, ground_truth, idss, miss,corrects):

    if normalize_answer(prediction) != normalize_answer(ground_truth):
        miss[idss] = str(str(prediction) + "$$" + str(pred_score))

    if normalize_answer(prediction) == normalize_answer(ground_truth):
        corrects[idss] = 1
        miss[idss] = str(str(prediction)+"$$"+str(pred_score))

    return (normalize_answer(prediction) == normalize_answer(ground_truth), miss, corrects)


def regex_match_score(prediction, pattern):
    try:
        compiled = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE
        )
    except BaseException:
        logger.warn('Regular expression failed to compile: %s' % pattern)
        return False
    return compiled.match(prediction) is not None


def metric_max_over_ground_truths(metric_fn, prediction, pred_score, ground_truths, idss, miss, corrects):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score, miss, corrects = metric_fn(prediction, pred_score, ground_truth, idss, miss, corrects)
        scores_for_ground_truths.append(score)
    return (max(scores_for_ground_truths), miss, corrects)



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
