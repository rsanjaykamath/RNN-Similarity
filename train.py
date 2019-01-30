import argparse
import torch
import numpy as np
import os
import sys
import subprocess
import json
import logging
import string
import regex as re

from utils.relevancy_metrics import get_map_mrr
from files.rnnsimilar import utils, vector, config, data
from files.rnnsimilar import RnnSimilarModel
from files import DATA_DIR as RNN_DATA

logger = logging.getLogger()


DATA_DIR = os.path.join(RNN_DATA, 'datasets/')
MODEL_DIR = 'models/'
EMBED_DIR = os.path.join(RNN_DATA, 'embeddings')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


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

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def add_train_args(parser):

    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=128,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--train-file', type=str,
                       help='Preprocessed train file')
    files.add_argument('--dev-file', type=str,
                       help='Preprocessed dev file')
    files.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str,
                       default='glove.840B.300d.txt',
                       help='Space-separated pretrained embeddings file')

    files.add_argument('--ner-dim', type=int,
                       default=8,)


    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')
    save_load.add_argument('--expand-dictionary', type='bool', default=False,
                           help='Expand dictionary of pretrained model to ' +
                                'include training/dev words of new data')

    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--uncased-question', type='bool', default=True,
                            help='Question words will be lower-cased')
    preprocess.add_argument('--uncased-doc', type='bool', default=True,
                            help='Document words will be lower-cased')
    preprocess.add_argument('--restrict-vocab', type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')


    general = parser.add_argument_group('General')
    general.add_argument('--official-eval', type='bool', default=True,
                         help='Validate with official SQuAD eval')
    general.add_argument('--valid-metric', type=str, default='f1',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--sort-by-len', type='bool', default=True,
                         help='Sort batches by length for speed')


def set_defaults(args):

    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)
    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)

    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    # Embeddings options
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')



    # Make sure tune_partial and fix_embeddings are consistent.
    if args.tune_partial > 0 and args.fix_embeddings:
        logger.warning('WARN: fix_embeddings set to False as tune_partial > 0.')
        args.fix_embeddings = False

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not (args.embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    return args



def init_from_scratch(args, train_exs, dev_exs):

    logger.info('-' * 100)
    logger.info('Generate features')
    feature_dict = utils.build_feature_dict(args, train_exs)

    logger.info('Num features = %d' % len(feature_dict))
    logger.info(feature_dict)

    logger.info('-' * 100)
    logger.info('Build dictionary')
    word_dict = utils.build_word_dict(args, train_exs + dev_exs)

    logger.info('Num words = %d' % len(word_dict))
    model = RnnSimilarModel(config.get_model_args(args), word_dict, feature_dict)

    if args.embedding_file:
        model.load_embeddings(word_dict.tokens(), args.embedding_file)

    return model




def train(args, data_loader, model, global_stats):

    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex))

        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)



def validate_unofficial(args, data_loader, model, global_stats, mode):

    eval_time = utils.Timer()
    top1 = utils.AverageMeter()

    examples = 0
    qids = []
    predictions_class=[]
    predictions = []
    labels = []
    queses = []
    docses = []
    real = []
    qids_tids = {}
    if mode == 'train':
        with open(args.data_dir + 'train_mapping.txt') as f:
            temp = f.readlines()

            for items in range(len(temp)):
                tempqids = temp[items].split("\t")[0]
                temptids = temp[items].split("\t")[1].rstrip()
                qids_tids[tempqids] = temptids

        with open(args.data_dir + 'dev_mapping.txt') as f:
            temp = f.readlines()

            for items in range(len(temp)):
                tempqids = temp[items].split("\t")[0]
                temptids = temp[items].split("\t")[1].rstrip()
                qids_tids[tempqids] = temptids

    if mode == 'dev':
        with open(args.data_dir + 'test_mapping.txt') as f:
            temp = f.readlines()

            for items in range(len(temp)):
                tempqids = temp[items].split("\t")[0]
                temptids = temp[items].split("\t")[1].rstrip()
                qids_tids[tempqids] = temptids

    ids_vals = {}

    for ex in data_loader:
        batch_size = ex[0].size(0)
        pred_s  = model.predict(ex)
        target_s = ex[5]
        ids = ex[6]
        ques = ex[7]
        doc = ex[8]


        if torch.is_tensor(target_s):
            target_s1 = [[e] for e in target_s]
        else:
            target_s1 = target_s

        batch_size = len(pred_s)
        for i in range(batch_size):

            pred_class = np.argmax(pred_s[i])

            pred = pred_s[i][0]#[pred_class]

            targ = target_s1[i][0]

            ids11 = qids_tids[ids[i]]
            real.append(ids[i])
            qids.append(float(ids11))
            predictions_class.append(pred)
            predictions.append(pred)
            labels.append(targ)
            queses.append(ques[i])
            docses.append(doc[i])

            if ids[i] in ids_vals:
                ids_vals[ids[i]].append((pred,targ,pred))
            else:
                ids_vals[ids[i]] = [(pred,targ,pred)]

        examples += batch_size
        if mode == 'train' and examples >= 1e4:
            break

    start = utils.AverageMeter()

    for items in ids_vals:
        t2 = ids_vals[items]

        t = sorted(t2, key=lambda x: x[0], reverse=True)

        if int(t[0][1]) == 1.0 or int(t[0][1]) == 1:
            start.update(1)
        else:
            start.update(0)

    dev_map, dev_mrr = get_map_mrr(qids, predictions, labels)

    logger.info('%s valid unofficial: Epoch = %d | start = %.2f | MAP = %.2f | MRR = %.2f ' %
                (mode, global_stats['epoch'], start.avg *100, dev_map *100, dev_mrr*100) +
                'valid time = %.2f (s)' % eval_time.time())

    return {'top1': start.avg *100,  'MAP': dev_map * 100, 'MRR': dev_mrr * 100}, real, predictions, labels, queses, docses, predictions_class

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])



def main(args):

    logger.info('-' * 100)
    logger.info('Load data files')

    dev_exs = utils.load_data(args, args.dev_file, 'Dev')
    logger.info('Num dev examples = %d' % len(dev_exs))

    train_exs = utils.load_data(args, args.train_file, 'Train')
    logger.info('Num train examples = %d' % len(train_exs))


    logger.info('-' * 100)
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        logger.info('Found a checkpoint...')
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = RnnSimilarModel.load_checkpoint(checkpoint_file, args)
    else:

        if args.pretrained:
            logger.info('Using pretrained model...')
            model = RnnSimilarModel.load(args.pretrained, args)
            if args.expand_dictionary:
                logger.info('Expanding dictionary for new data...')
                words = utils.load_words(args, train_exs + dev_exs)
                added = model.expand_dictionary(words)
                if args.embedding_file:
                    model.load_embeddings(added, args.embedding_file)


        else:
            logger.info('Training model from scratch...')
            model = init_from_scratch(args, train_exs, dev_exs)

        if args.tune_partial > 0:
            logger.info('-' * 100)
            logger.info('Counting %d most frequent question words' %
                        args.tune_partial)
            top_words = utils.top_question_words(
                args, train_exs, model.word_dict
            )
            for word in top_words[:5]:
                logger.info(word)
            logger.info('...')
            for word in top_words[-6:-1]:
                logger.info(word)
            model.tune_embeddings([w[0] for w in top_words])

        model.init_optimizer()

    if args.cuda:
        model.cuda()

    if args.parallel:
        model.parallelize()

    logger.info('-' * 100)
    logger.info('Make data loaders')
    train_dataset = data.ReaderDataset(train_exs, model, single_answer=True)
    if args.sort_by_len:

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
        )

    dev_dataset = data.ReaderDataset(dev_exs, model, single_answer=False)
    if args.sort_by_len:
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
            )
    else:
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
        )

    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))


    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        train(args, train_loader, model, stats)

        validate_unofficial(args, train_loader, model, stats, mode='train')

        result, qids, predictions, labels, queses, docses, predictions_class = validate_unofficial(args, dev_loader, model, stats, mode='dev')


        if result[args.valid_metric] > stats['best_valid']:
            logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                        (args.valid_metric, result[args.valid_metric],
                         stats['epoch'], model.updates))
            model.save(args.model_file)
            stats['best_valid'] = result[args.valid_metric]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'RNN Similarity model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)


    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))


    main(args)
