
import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

from .vector import vectorize, batchify
from .model import RnnSimilarModel
from . import DEFAULTS, utils
from .. import tokenizers

logger = logging.getLogger(__name__)


PROCESS_TOK = None


def init(tokenizer_class, annotators):
    global PROCESS_TOK
    PROCESS_TOK = tokenizer_class(annotators=annotators)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)



class Predictor(object):

    def __init__(self, args, model=None, tokenizer=None, normalize=True,
                 embedding_file=None, num_workers=None):

        logger.info('Initializing model...')

        self.model = RnnSimilarModel.load(model or DEFAULTS['model'], new_args=args,
                                    normalize=normalize)

        if embedding_file:
            logger.info('Expanding dictionary...')
            words = utils.index_embedding_words(embedding_file)
            added = self.model.expand_dictionary(words)
            self.model.load_embeddings(added, embedding_file)

        logger.info('Initializing tokenizer...')
        annotators = tokenizers.get_annotators_for_model(self.model)
        if not tokenizer:
            tokenizer_class = DEFAULTS['tokenizer']
        else:
            tokenizer_class = tokenizers.get_class(tokenizer)

        if num_workers is None or num_workers > 0:
            self.workers = ProcessPool(
                num_workers,
                initializer=init,
                initargs=(tokenizer_class, annotators),
            )
        else:
            self.workers = None
            self.tokenizer = tokenizer_class(annotators=annotators)

    def predict(self, document, question, candidates=None, top_n=1):
        results = self.predict_batch([(document, question, candidates,)], top_n)
        return results[0]

    def predict_batch(self, args, batch, top_n=1):
        documents, questions, candidates, labels = [], [], [], []
        for b in batch:
            documents.append(b[0])
            questions.append(b[1])
            labels.append(b[2])
            candidates.append(b[2] if len(b) == 3 else None)
        candidates = candidates if  any(candidates) else None

        # Tokenize the inputs, perhaps multi-processed.
        if self.workers:
            q_tokens = self.workers.map_async(tokenize, questions)
            d_tokens = self.workers.map_async(tokenize, documents)
            q_tokens = list(q_tokens.get())
            d_tokens = list(d_tokens.get())
        else:
            q_tokens = list(map(self.tokenizer.tokenize, questions))
            d_tokens = list(map(self.tokenizer.tokenize, documents))

        examples = []

        for i in range(len(questions)):
            examples.append({
                'id': i,
                'question': q_tokens[i].words(),
                'qlemma': q_tokens[i].lemmas(),
                'document': d_tokens[i].words(),
                'lemma': d_tokens[i].lemmas(),
                'pos': d_tokens[i].pos(),
                'ner': d_tokens[i].entities(),
                'label': labels[i],
            })

        # Stick document tokens in candidates for decoding
        if candidates:
            candidates = [{'input': d_tokens[i], 'cands': candidates[i]}
                          for i in range(len(candidates))]

        batch_exs = batchify([vectorize(e, self.model) for e in examples])
        score = self.model.predict(batch_exs, candidates, top_n)
        labels = [e['label'] for e in examples]
        results = []
        for i in range(len(score)):
            predictions = []
            for j in range(len(score[i])):
                predictions.append((score[i][j], labels[i]))
            results.append(predictions)
        return results

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
