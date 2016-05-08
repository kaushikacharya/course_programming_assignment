#! /usr/bin/python

__author__ = 'Kaushik Acharya'
__date__ = '$Sep 07, 2014'

import json
from collections import defaultdict


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class MachineTranslation:
    def __init__(self, eng_corpus_filename='corpus.en', foreign_corpus_filename='corpus.es', debug_flag=True):
        self.count_struct = []
        self.word_translation_param = []
        self.alignment_param = []

        # index = 0 :: English as native language
        # index = 1 :: English as foreign language
        for language_i in range(0, 2):
            self.count_struct.append(Struct(translation_dict=None, distortion_dict=None))
            self.word_translation_param.append({})
            self.alignment_param.append({})

        self.null_word = unicode('NULL', 'utf8')
        self.debug_flag = debug_flag

        # corpus files
        self.eng_corpus_filename = eng_corpus_filename
        self.foreign_corpus_filename = foreign_corpus_filename

        self.param_filename = 'param.json'

    def initialize_count(self, language_index):
        # translation
        self.count_struct[language_index].translation_dict = dict()
        self.count_struct[language_index].translation_dict['native_foreign'] = defaultdict(float)
        self.count_struct[language_index].translation_dict['native'] = defaultdict(float)
        # distortion
        self.count_struct[language_index].distortion_dict = dict()

    def save_param(self):
        param = dict()
        param['word_translation'] = self.word_translation_param
        param['alignment'] = self.alignment_param
        json.dump(param, open(self.param_filename, 'w'))

    def load_param(self):
        param = json.load(open(self.param_filename))
        self.word_translation_param = param['word_translation']
        self.alignment_param = param['alignment']

    def initialize_param(self, language_index, flag_translation, flag_alignment):
        # IBM model 1: initialize word translation param
        # IBM model 2: initialize alignment param (uses word translation param set by IBM model 1)
        if flag_translation:
            self.word_translation_param[language_index] = dict()
        if flag_alignment:
            self.alignment_param[language_index] = dict()

        # go through the lines of both the corpus simultaneously and collect all the possible foreign word for
        # each native language word.
        with open(self.eng_corpus_filename, 'r') as fd_eng:
            with open(self.foreign_corpus_filename, 'r') as fd_foreign:
                sentence_index = 0
                for line_eng in fd_eng:
                    line_foreign = fd_foreign.readline()
                    line_eng = unicode(line_eng.strip('\n'), 'utf8')
                    line_foreign = unicode(line_foreign.strip('\n'), 'utf8')

                    if (len(line_eng) == 0) or (len(line_foreign) == 0):
                        sentence_index += 1
                        continue  # ignoring since translation is absent

                    english_sentence_tokens = line_eng.split()
                    foreign_sentence_tokens = line_foreign.split()

                    if flag_translation:
                        if language_index == 0:
                            # English as native language
                            self.add_translation_param_keys(language_index, [self.null_word], foreign_sentence_tokens)

                            self.add_translation_param_keys(language_index, english_sentence_tokens,
                                                              foreign_sentence_tokens)
                        else:
                            # English as foreign language
                            self.add_translation_param_keys(language_index, [self.null_word], english_sentence_tokens)

                            self.add_translation_param_keys(language_index, foreign_sentence_tokens,
                                                              english_sentence_tokens)

                    if flag_alignment:
                        if language_index == 0:
                            # English as native language
                            lk = len(english_sentence_tokens)
                            mk = len(foreign_sentence_tokens)
                        else:
                            # English as foreign language
                            lk = len(foreign_sentence_tokens)
                            mk = len(english_sentence_tokens)

                        self.add_alignment_param_keys(language_index, lk, mk)

                    sentence_index += 1

        # now assign the params based on uniform distribution over the foreign words that could be
        # aligned to native word

        # translation param
        if flag_translation:
            self.initialize_translation_param(language_index)

        # alignment param
        if flag_alignment:
            self.initialize_alignment_param(language_index)

    def compute_count_for_training_pair_of_sentence(self, language_index, native_sentence_tokens,
                                                    foreign_sentence_tokens, ibm_model=1):
        lk = len(native_sentence_tokens)
        mk = len(foreign_sentence_tokens)
        tuple_lk_mk = tuple([lk, mk])

        # using 0-indexed unlike in pseudo-code
        for i in range(0, mk):
            word_foreign = foreign_sentence_tokens[i]

            # compute denominator for delta_k_i_j
            denominator_delta_k_i_j = 0
            for j in range(-1, lk):
                if j == -1:
                    word_native = self.null_word
                else:
                    word_native = native_sentence_tokens[j]

                if ibm_model == 1:
                    denominator_delta_k_i_j += self.word_translation_param[language_index][word_native][word_foreign]
                else:
                    denominator_delta_k_i_j += self.alignment_param[language_index][tuple_lk_mk][tuple([j, i])] * \
                                               self.word_translation_param[language_index][word_native][word_foreign]

            for j in range(-1, lk):
                # iterating over the words in the native language sentence
                if j == -1:
                    word_native = self.null_word
                else:
                    word_native = native_sentence_tokens[j]

                if ibm_model == 1:
                    delta_k_i_j = self.word_translation_param[language_index][word_native][word_foreign]/denominator_delta_k_i_j
                else:
                    delta_k_i_j = (self.alignment_param[language_index][tuple_lk_mk][tuple([j, i])] *
                                   self.word_translation_param[language_index][word_native][word_foreign])/denominator_delta_k_i_j

                self.count_struct[language_index].translation_dict['native_foreign'][tuple([word_native, word_foreign])] += delta_k_i_j
                self.count_struct[language_index].translation_dict['native'][word_native] += delta_k_i_j

                tuple_lk_mk = tuple([lk, mk])
                if tuple_lk_mk not in self.count_struct.distortion_dict[language_index].keys():
                    self.count_struct.distortion_dict[language_index][tuple_lk_mk] = defaultdict(float)

                self.count_struct.distortion_dict[language_index][tuple_lk_mk][tuple([j, i])] += delta_k_i_j
                self.count_struct.distortion_dict[language_index][tuple_lk_mk][i] += delta_k_i_j

    def estimate_param_based_on_ibm_model(self, language_index, ibm_model=1, n_iteration=5):
        # initialize translation & alignment params
        if ibm_model == 1:
            flag_translation = True
            flag_alignment = False
        else:
            flag_translation = False
            flag_alignment = True

    def add_translation_param_keys(self, language_index, native_sentence_tokens, foreign_sentence_tokens):
        for word_native in native_sentence_tokens:
            if word_native not in self.word_translation_param[language_index]:
                self.word_translation_param[language_index][word_native] = {}
            for word_foreign in foreign_sentence_tokens:
                self.word_translation_param[language_index][word_native][word_foreign] = 0

    def add_alignment_param_keys(self, language_index, lk, mk):
        # lk: corresponds to native language
        # mk: corresponds to foreign language
        tuple_lk_mk = tuple([lk, mk])
        if tuple_lk_mk not in self.alignment_param[language_index].keys():
            self.alignment_param[language_index][tuple_lk_mk] = {}

    def initialize_translation_param(self, language_index):
        for word_native in self.word_translation_param[language_index].keys():
            n_e = len(self.word_translation_param[language_index][word_native].keys())
            param_val = 1.0/n_e
            for word_foreign in self.word_translation_param[language_index][word_native].keys():
                self.word_translation_param[language_index][word_native][word_foreign] = param_val

    def initialize_alignment_param(self, language_index):
        for tuple_lk_mk in self.alignment_param[language_index].keys():
            lk = tuple_lk_mk[0]
            mk = tuple_lk_mk[1]

            for i in range(0, mk):
                for j in range(-1, lk):
                    self.alignment_param[language_index][tuple_lk_mk][tuple([j, i])] = 1.0/(lk + 1)
