import argparse
import os
import sys
import json
import time
import spacy
nlp = spacy.load('en_core_web_sm')

from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial
from files import tokenizers

import spotlight
spotlighturl = 'http://localhost:2222/rest/annotate'

import string
import re


conv_1 = {"": "0" , "PERSON" : "HUM", "ORG" : "HUM" , "GPE" : "LOC", "LOC" : "LOC", "PRODUCT" : "ENTY", "EVENT" : "ENTY", "LANGUAGE" : "ENTY", "DATE" : "NUM", "TIME" : "NUM", "PERCENT" : "NUM", "MONEY" : "NUM", "QUANTITY" : "NUM", "ORDINAL" : "NUM", "CARDINAL" : "NUM", "NORP" : ["LOC", "ENTY", "HUM"], "FAC" : "ENTY", "WORK_OF_ART" : "ENTY", "LAW" : "ENTY"}

filen = "dev"
filenn_lat = "LAT_SQUAD/annotated_data_" + str(filen) + ".txt"

type_lats = ["HUM", "LOC", "ENTY", "NUM", "DESC", "ABBR"]
ent_words = ["wikipage_hum", "wikipage_loc", "wikipage_enty", "wikipage_num", "wikipage_desc", "wikipage_abbr"]
max_ent_words = ["max_entity_hum", "max_entity_loc", "max_entity_enty", "max_entity_num",  "max_entity_desc", "max_entity_abbr"]



with open(filenn_lat) as f:
    lats_read = [json.loads(line) for line in f]

accepts = {}
for ex in lats_read:
    try:
        if ex['major_type'] in type_lats:
            accepts[str(ex['id'])] = ex['major_type']
    except  KeyError:
        pass

#print (accepts)
single = 11 # 11 for just once nothing special



def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

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


TOK = None


def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)

    Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': tokens.words(),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
        'plain_text': text,
    }
    return output


def load_dt1(data):
    global TOK
    # print (TOK.)
    tokens = TOK.tokenize_just(data)
    return tokens


# ------------------------------------------------------------------------------
# Process dataset examples
# ------------------------------------------------------------------------------


def spotlight_annotate(ip_text, c, tagmes, ex_id, question_txt):
    global flag
    flag = ""
    global enc

    ent_offset = []
    doc_offset = []
    count = 0

    for items in range(len(tagmes['mention'])):

        spot_ent = tagmes['mention'][items]
        gold = spot_ent
        doc = nlp(gold)
        gold_lat = None

        if len(doc.ents) > 0:
            for ent in doc.ents:
                gold_lat = str(ent.label_)
                gold_lat = conv_1[gold_lat]

        if gold in question_txt:
            pass
        else:
            if gold_lat == accepts[ex_id]:

                if tagmes['entity'][items] == "max_entity_left":
                    wikipage_word = max_ent_words[type_lats.index(accepts[ex_id])]
                else:
                    wikipage_word = ent_words[type_lats.index(accepts[ex_id])]

                # if tagmes['entity'][items] == "max_entity_left":
                #     wikipage_word = "max_entity_left"
                offsets = tagmes['offset'][items]
                URI = tagmes['entity'][items]
                if len(str("Wikipage_" + URI).lower()) > len(spot_ent):  # (str("Wikipage_" + URI).lower() in embed_words)
                    t1 = spot_ent
                    start = offsets + count
                    end = offsets + count + len(spot_ent)


                    if start == 0 or len(ip_text) == end:

                        if start == 0 and len(ip_text) == end:
                            doc_offset.append((offsets, offsets + len(spot_ent)))
                            URI = re.sub('([^0-9a-zA-Z]+)', r'\1$', URI)

                            if single == 0:
                                repl = (str(spot_ent) + str(" ") + str("Wikipage_$" + URI))
                            elif single == 11:
                                repl = (str(wikipage_word))
                                #repl = (str("Wikipage_$" + URI))

                            elif single == 5:
                                repl = (str("Wikipage_$" + URI) + str(" ")) * times
                                repl = repl[:-1]
                            elif single == 6:

                                repl = (str("Wikipage_$" + URI) + str(" ")) + ((str(0) + str(" ")) * (times - 1))
                                repl = repl[:-1]

                            else:
                                repl = (str("Wikipage_$" + URI) + str(" ") + str(spot_ent) + str(" ") + str("Wikipage_$" + URI))

                            ip_text = ip_text[:start] + str(repl) + ip_text[end:]
                            count = count + len(str(repl)) - len(spot_ent)
                            ent_offset.append((start, start + len(str(repl))))

                        if start == 0 and ip_text[end] == " ":
                            doc_offset.append((offsets, offsets + len(spot_ent)))
                            URI = re.sub('([^0-9a-zA-Z]+)', r'\1$', URI)
                            if single == 0:
                                repl = (str(spot_ent) + str(" ") + str("Wikipage_$" + URI))
                            elif single == 11:
                                repl = (str(wikipage_word))
                                # repl = (str("Wikipage_$" + URI))
                            elif single == 5:
                                repl = (str("Wikipage_$" + URI) + str(" ")) * times
                                repl = repl[:-1]
                            elif single == 6:

                                repl = (str("Wikipage_$" + URI) + str(" ")) + ((str(0) + str(" ")) * (times - 1))
                                repl = repl[:-1]

                            else:
                                repl = (str("Wikipage_$" + URI) + str(" ") + str(spot_ent) + str(" ") + str("Wikipage_$" + URI))
                            ip_text = ip_text[:start] + str(repl) + ip_text[end:]
                            count = count + len(str(repl)) - len(spot_ent)
                            ent_offset.append((start, start + len(str(repl))))

                        elif len(ip_text) == end and ip_text[start - 1] == " ":
                            doc_offset.append((offsets, offsets + len(spot_ent)))
                            URI = re.sub('([^0-9a-zA-Z]+)', r'\1$', URI)
                            if single == 0:
                                repl = (str(spot_ent) + str(" ") + str("Wikipage_$" + URI))
                            elif single == 11:
                                repl = (str(wikipage_word))
                                # repl = (str("Wikipage_$" + URI))
                            elif single == 5:
                                repl = (str("Wikipage_$" + URI) + str(" ")) * times
                                repl = repl[:-1]
                            elif single == 6:

                                repl = (str("Wikipage_$" + URI) + str(" ")) + ((str(0) + str(" ")) * (times - 1))
                                repl = repl[:-1]

                            else:
                                repl = (str("Wikipage_$" + URI) + str(" ") + str(spot_ent) + str(" ") + str("Wikipage_$" + URI))
                            ip_text = ip_text[:start] + str(repl) + ip_text[end:]
                            count = count + len(str(repl)) - len(spot_ent)
                            ent_offset.append((start, start + len(str(repl))))

                        else:
                            pass

                    else:

                        if ip_text[start - 1] == " " and ip_text[end] == " ":

                            doc_offset.append((offsets, offsets + len(spot_ent)))
                            URI = re.sub('([^0-9a-zA-Z]+)', r'\1$', URI)
                            if single == 0:
                                repl = (str(spot_ent) + str(" ") + str("Wikipage_$" + URI))
                            elif single == 11:
                                repl = (str(wikipage_word))
                                # repl = (str("Wikipage_$" + URI))
                            elif single == 5:
                                repl = (str("Wikipage_$" + URI) + str(" ")) * times
                                repl = repl[:-1]
                            elif single == 6:

                                repl = (str("Wikipage_$" + URI) + str(" ")) + ((str(0) + str(" ")) * (times - 1))
                                repl = repl[:-1]

                            else:
                                repl = (str("Wikipage_$" + URI) + str(" ") + str(spot_ent) + str(" ") + str("Wikipage_$" + URI))
                            ip_text = ip_text[:start] + str(repl) + ip_text[end:]
                            count = count + len(str(repl)) - len(spot_ent)
                            ent_offset.append((start, start + len(str(repl))))


                        elif len(ip_text) == int(end + 1) and str(ip_text[-1]) == "?":

                            doc_offset.append((offsets, offsets + len(spot_ent)))
                            URI = re.sub('([^0-9a-zA-Z]+)', r'\1$', URI)
                            if single == 0:
                                repl = (str(spot_ent) + str(" ") + str("Wikipage_$" + URI))
                            elif single == 11:
                                repl = (str(wikipage_word))
                                # repl = (str("Wikipage_$" + URI))
                            elif single == 5:
                                repl = (str("Wikipage_$" + URI) + str(" ")) * times
                                repl = repl[:-1]
                            elif single == 6:

                                repl = (str("Wikipage_$" + URI) + str(" ")) + ((str(0) + str(" ")) * (times - 1))
                                repl = repl[:-1]

                            else:
                                repl = (str("Wikipage_$" + URI) + str(" ") + str(spot_ent) + str(" ") + str("Wikipage_$" + URI))

                            ip_text = ip_text[:start] + str(repl + " " + "?")
                            count = count + len(str(repl + "?")) - len(spot_ent)
                            ent_offset.append((start, start + len(str(repl))))


    # print (ip_text)
    return ip_text, flag, ent_offset, doc_offset, c


def load_dataset(path):
    """Load json file and store fields separately."""
    with open(path) as f:
        data = json.load(f)['data']

    output = {'qids': [], 'questions': [], 'answers': [], 'binary_answers': [],
              'contexts': [], 'qid2cid': [], 'answers_starts': [], 'answers_texts': [], 'full_para': []}
    c = 0
    count = 0
    cdd = 0
    dd = 0
    for article in data:
        for paragraph in article['paragraphs']:
            c = c + 1
            for qa in paragraph['qas']:
                cdd = cdd + 1

                context = paragraph['context']
                count += 1
                print (count)

                doc = nlp(context)

                sent_tokenize_list = [sent.string.strip() for sent in doc.sents]
                sent_offsets = [context.find(sent.string.strip()) for sent in doc.sents]

                off = [context.find(sent.string.strip()) for sent in doc.sents]

                # sent_tokenize_list = sent_tokenize(context)
                # off = [0]

                # for ind, items in enumerate(sent_tokenize_list):
                # 	off.append(off[ind] + len(items) + 1)

                if 'answers' in qa:
                    if len(qa['answers']) > 0:
                        answer_start = qa['answers'][0]['answer_start']

                        sent_offsets = off
                        ans_posti = 0
                        for inx, items in enumerate(sent_offsets):

                            if len(sent_offsets) == 1:

                                ans_posti = 0
                                break
                            elif answer_start == sent_offsets[-1]:

                                ans_posti = sent_offsets.index(sent_offsets[-1])

                                break
                            elif answer_start == sent_offsets[0]:

                                ans_posti = sent_offsets.index(sent_offsets[0])
                                break
                            elif answer_start > sent_offsets[-1]:

                                ans_posti = sent_offsets.index(sent_offsets[-1])
                                break

                            elif answer_start in sent_offsets:

                                ans_posti = sent_offsets.index(answer_start)
                                break
                            else:

                                try:
                                    if items < answer_start and sent_offsets[inx + 1] > answer_start:
                                        ans_posti = sent_offsets.index(items)
                                        break
                                except IndexError:

                                    pass




                # doc = nlp(context)
                #
                # sent_offsets = [context.find(sent.string.strip()) for sent in doc.sents]
                #
                # spacy_sents = []
                #
                # for index, val in enumerate(sent_offsets):
                # 	try:
                # 		spacy_sents.append(context[sent_offsets[index]:sent_offsets[index + 1]])
                # 	except:
                # 		spacy_sents.append(context[sent_offsets[index]:])

                for index, items in enumerate(sent_tokenize_list):
                    if index == ans_posti:
                        if items.find(qa['answers'][0]['text']) == -1:
                            dd += 1
                        output['answers_starts'].append(items.find(qa['answers'][0]['text']))
                        output['answers_texts'].append(qa['answers'][0]['text'])
                        output['binary_answers'].append(1)
                    else:
                        output['answers_starts'].append(-1)
                        output['answers_texts'].append('')
                        output['binary_answers'].append(0)
                    # print ("Ques: " + str(question))
                    # print ("Context: " + str(items))


                    output['contexts'].append(items)
                    output['full_para'].append(context)
                    output['qids'].append(qa['id'])
                    output['questions'].append(qa['question'])
                    output['qid2cid'].append(len(output['contexts']) - 1)



    print ("Missing ones: " + str(dd))
    return output


def find_answer(offsets, begin_offset, end_offset):
    """Match token offsets with the char begin/end offsets of the answer."""
    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
    assert (len(start) <= 1)
    assert (len(end) <= 1)
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]

import operator
def process_dataset(data, tokenizer, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""

    ids = []
    questions = []
    answers = []
    labels = []

    c = 0
    missed_ques = {}

    full_spots = {}
    replaces = {}
    for idx in range(len(data['qids'])):

        if str(data['qids'][idx]) in accepts:

            missed_ques[data['qids'][idx]] = 1
            ip_text = data['contexts'][idx]
            # print (ip_text)

            tags = []
            ex1 = {}
            try:
                temp1 = spotlight.annotate(spotlighturl, ip_text, confidence=0.7, support=20, spotter='Default')
            except spotlight.SpotlightException:
                temp1 = []
            except:
                temp1 = []

            for items in temp1:
                spot_ent = items['surfaceForm']
                URI = items['URI'].replace("http://dbpedia.org/resource/", "")

                tags.append((
                    spot_ent,
                    items['offset'],
                    URI.replace(" ", "_")
                ))

            ex1['mention'] = [t[0] for t in tags]
            ex1['offset'] = [t[1] for t in tags]
            ex1['entity'] = [t[2] for t in tags]

            ques_dict = ex1

            if data['qids'][idx] in full_spots:
                repla = full_spots[data['qids'][idx]]
                #paradict = full_spots[data['qids'][idx]]
            else:
                ip_text1 = data['full_para'][idx]
                # print (ip_text)

                tags1 = []
                ex2 = {}
                try:
                    temp12 = spotlight.annotate(spotlighturl, ip_text1, confidence=0.7, support=20, spotter='Default')
                except spotlight.SpotlightException:
                    temp12 = []
                except:
                    temp12 = []

                for items in temp12:
                    spot_ent = items['surfaceForm']
                    URI = items['URI'].replace("http://dbpedia.org/resource/", "")

                    tags1.append((
                        spot_ent,
                        items['offset'],
                        URI.replace(" ", "_")
                    ))

                ex2['mention'] = [t[0] for t in tags1]
                ex2['offset'] = [t[1] for t in tags1]
                ex2['entity'] = [t[2] for t in tags1]

                t1 = ex2['entity']
                sorts = {}
                for items in t1:
                    try:
                        sorts[items] += 1
                    except:
                        sorts[items] = 1

                t2 = sorted(sorts.items(), key=operator.itemgetter(1), reverse=True)

                if len(t2) > 1:
                    if t2[0][1] >= 2 * t2[1][1]:
                        t1 = ["max_entity_left" if x == t2[0][0] else x for x in t1]
                        reee = {t2[0][0] : "max_entity_left" }
                        ex2['entity'] = t1



                full_spots[data['qids'][idx]] = reee

                #paradict = ex2
                repla = reee





            ques_dict['entity'] = ["max_entity_left" if x in repla else x for x in ques_dict['entity']]


            ques_annot, flag, ques_ent_offset, ques_doc_offset, c = spotlight_annotate(data['contexts'][idx], c,
                                                                                       ques_dict,
                                                                                       str(data['qids'][idx]), data['questions'][idx])

            if (ques_annot):
                ids.append(data['qids'][idx])
                wikipage_word = ent_words[type_lats.index(accepts[str(data['qids'][idx])])]
                max_wikipage_word = max_ent_words[type_lats.index(accepts[str(data['qids'][idx])])]
                #questions.append((data['questions'][idx]) + ' ' + str(wikipage_word) )
                questions.append((data['questions'][idx]) + ' ' + str(wikipage_word) + ' ' + str(max_wikipage_word))

                answers.append(ques_annot)
                labels.append(data['binary_answers'][idx])

        # 'id': data['qids'][idx],
        # 'question': (data['questions'][idx]) + str(" Wikipage_entity"),
        # 'answer': ques_annot,
        # 'label': data['binary_answers'][idx],


        else:

            if (data['contexts'][idx]):
                ids.append(data['qids'][idx])
                questions.append(data['questions'][idx])
                answers.append(data['contexts'][idx])
                labels.append(data['binary_answers'][idx])

        # 'id': data['qids'][idx],
        # 'question': data['questions'][idx],
        # 'answer': data['contexts'][idx],
        # 'label': data['binary_answers'][idx],

    tokenizer_class = tokenizers.get_class(tokenizer)
    make_pool = partial(Pool, workers, initializer=init)
    workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}}))
    q_tokens = workers.map(tokenize, questions)
    workers.close()
    workers.join()

    workers = make_pool(
        initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}})
    )
    c_tokens = workers.map(tokenize, answers)

    workers.close()
    workers.join()

    for idx in range(len(ids)):
        question = q_tokens[idx]['words']
        qlemma = q_tokens[idx]['lemma']
        q_pos = q_tokens[idx]['pos']
        q_ner = q_tokens[idx]['ner']

        document = c_tokens[data['qid2cid'][idx]]['words']
        offsets = c_tokens[data['qid2cid'][idx]]['offsets']
        lemma = c_tokens[data['qid2cid'][idx]]['lemma']
        pos = c_tokens[data['qid2cid'][idx]]['pos']
        ner = c_tokens[data['qid2cid'][idx]]['ner']
        paragraph_text = c_tokens[data['qid2cid'][idx]]['plain_text']
        paragraph_ques = q_tokens[idx]['plain_text']

        ans_tokens = []

        # print ('answer_start: '+str(data['answers_starts'][idx]))
        # print ('real: ' + str(data['binary_answers'][idx]))
        found = find_answer(offsets,
                            data['answers_starts'][idx],
                            data['answers_starts'][idx] + len(data['answers_texts'][idx]))
        if found:
            ans_tokens.append(found)

        yield {
            'id': data['qids'][idx],
            'question': question,
            'document': document,
            'paragraph_text': paragraph_text,
            'paragraph_ques': paragraph_ques,
            'label': data['binary_answers'][idx],
            'offsets': offsets,
            'answers': ans_tokens,
            'qlemma': qlemma,
            'lemma': lemma,
            'pos': pos,
            'ner': ner,
            'q_pos': q_pos,
            'q_ner': q_ner,
        }


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to SQuAD data directory')
parser.add_argument('out_dir', type=str, help='Path to output file dir')
parser.add_argument('--split', type=str, help='Filename for train/dev split',
                    default='SQuAD-v1.1-train')
parser.add_argument('--workers', type=int, default=None)
parser.add_argument('--tokenizer', type=str, default='corenlp')
parser.add_argument('--random', type=bool, default=False)
args = parser.parse_args()

t0 = time.time()

in_file = os.path.join(args.data_dir, args.split + '.json')
print('Loading dataset %s' % in_file, file=sys.stderr)
dataset = load_dataset(in_file)

if args.random == True:

    out_file = os.path.join(
        args.out_dir, '%s-processed-%s-rerankdata-spotlighted-harishway.txt' % (filen, args.tokenizer)
    )

else:
    out_file = os.path.join(
        args.out_dir, '%s-processed-%s-rerankdata-spotlighted-harishway.txt' % (filen, args.tokenizer)
    )

print('Will write to file %s' % out_file, file=sys.stderr)
with open(out_file, 'w') as f:
    for ex in process_dataset(dataset, args.tokenizer, args.workers):
        f.write(json.dumps(ex) + '\n')
print('Total time: %.4f (s)' % (time.time() - t0))
print ("lenght of fuck yes: " + str(len(fuck_yes)))
print ("length of singles: " + str(len(singles)))
print ("lenght of total: " + str(len(total)))



