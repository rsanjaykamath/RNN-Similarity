import spacy
import copy
from .tokenizer import Tokens, Tokenizer


class SpacyTokenizer(Tokenizer):
	def __init__(self, **kwargs):
		model = kwargs.get('model', 'en')
		self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
		nlp_kwargs = {'parser': False}
		if not any([p in self.annotators for p in ['lemma', 'pos', 'ner']]):
			nlp_kwargs['tagger'] = False
		if 'ner' not in self.annotators:
			nlp_kwargs['entity'] = False
		self.nlp = spacy.load(model, **nlp_kwargs)

	def tokenize(self, text):
		clean_text = text.replace('\n', ' ')
		tokens = self.nlp.tokenizer(clean_text)
		if any([p in self.annotators for p in ['lemma', 'pos', 'ner']]):
			self.nlp.tagger(tokens)
		if 'ner' in self.annotators:
			self.nlp.entity(tokens)

		data = []
		for i in range(len(tokens)):
			# Get whitespace
			start_ws = tokens[i].idx
			if i + 1 < len(tokens):
				end_ws = tokens[i + 1].idx
			else:
				end_ws = tokens[i].idx + len(tokens[i].text)

			data.append((
				tokens[i].text,
				text[start_ws: end_ws],
				(tokens[i].idx, tokens[i].idx + len(tokens[i].text)),
				tokens[i].tag_,
				tokens[i].lemma_,
				tokens[i].ent_type_,
				tokens[i].idx,
				tokens[i].idx + len(tokens[i].text),
			))

		return Tokens(data, self.annotators, opts={'non_ent': ''})

