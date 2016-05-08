#! /usr/bin/python

__author__ = 'Kaushik Acharya'
__date__ = '$Aug 03, 2014'

from count_cfg_freq import Counts
import json
import re
from collections import defaultdict


class PCFG:
    def __init__(self, original_train_file, modified_train_file, count_threshold=5):
        self.count_threshold = count_threshold
        self.word_freq = defaultdict(int)
        self.cfg_counter = Counts()
        self.bp_table = {}

    def compute_cfg_frequency_in_train_file(self, train_file):
        self.cfg_counter = Counts()
        with open(train_file, 'r') as fd:
            for line in fd:
                cfg_tree = json.loads(line)
                self.cfg_counter.count(cfg_tree)

    def compute_word_frequency_in_cfg(self):
        self.word_freq = defaultdict(int)
        # case sensitive
        # https://class.coursera.org/nlangp-001/forum/thread?thread_id=631

        # Terminal word can be assigned to multiple non-Terminals
        # https://class.coursera.org/nlangp-001/forum/thread?thread_id=620#post-2613
        for (sym, word), count in self.cfg_counter.unary.iteritems():
            self.word_freq[word] += count

    def is_rare_word(self, word):
        return self.word_freq[word] < self.count_threshold

    def replace_infrequent_words_in_parse_tree(self, tree):
        if isinstance(tree, basestring):
            return
        if len(tree) == 3:
            # binary rule
            self.replace_infrequent_words_in_parse_tree(tree[1])
            self.replace_infrequent_words_in_parse_tree(tree[2])
        elif len(tree) == 2:
            # unary rule
            word = tree[1]
            if self.is_rare_word(word):
                tree[1] = '_RARE_'

    def compute_binary_parameter(self, symbol, y1, y2):
        return self.cfg_counter.binary[(symbol, y1, y2)]*1.0/self.cfg_counter.nonterm[symbol]

    def compute_unary_parameter(self, symbol, word):
        return self.cfg_counter.unary[(symbol, word)]*1.0/self.cfg_counter.nonterm[symbol]

    def CKY_algorithm(self, sentence):
        sentence_tokens = re.split(r'[ ]+', sentence.rstrip())
        n_tokens = len(sentence_tokens)

        max_prob_table = defaultdict(float)  # pi table
        self.bp_table = {}

        # now build the dynamic programming table bottom-up
        for symbol in self.cfg_counter.nonterm.iterkeys():
            for i in range(0, n_tokens):
                word = sentence_tokens[i]
                if self.is_rare_word(word):
                    word = '_RARE_'
                key = (symbol, word)
                if key in self.cfg_counter.unary.keys():
                    max_prob_table[(i, i, symbol)] = self.compute_unary_parameter(symbol, word)

        for step in range(1, n_tokens):
            for i in range(0, n_tokens-step):
                j = i + step

                for (sym, y1, y2) in self.cfg_counter.binary.iterkeys():
                    binary_param = self.compute_binary_parameter(sym, y1, y2)
                    max_prob_mult = 0
                    max_prob_s = None
                    for s in range(i, j):
                        prob_mult = max_prob_table[(i, s, y1)]*max_prob_table[(s+1, j, y2)]
                        if max_prob_mult < prob_mult:
                            max_prob_mult = prob_mult
                            max_prob_s = s

                    prob_with_current_binary_rule_over_i_j = binary_param*max_prob_mult

                    if max_prob_table[(i, j, sym)] < prob_with_current_binary_rule_over_i_j:
                        max_prob_table[(i, j, sym)] = prob_with_current_binary_rule_over_i_j
                        self.bp_table[(i, j, sym)] = tuple([max_prob_s, y1, y2])

        parse_tree = self.create_parse_tree(0, n_tokens-1, 'SBARQ', sentence_tokens)
        return parse_tree

    def create_parse_tree(self, i, j, sym, sentence_tokens):
        # [sym, func(i,s,y1), func(s+1,j,y2) ]
        parse_sub_tree = []
        if i == j:
            parse_sub_tree = [sym, sentence_tokens[i]]
            # parse_sub_tree = '[' + sym + ', ' + sentence_tokens[i] + ']'
        else:
            split_tuple = self.bp_table[(i, j, sym)]
            s = split_tuple[0]
            y1 = split_tuple[1]
            y2 = split_tuple[2]

            parse_sub_tree = [sym, self.create_parse_tree(i, s, y1, sentence_tokens),
                              self.create_parse_tree(s+1, j, y2, sentence_tokens)]
            '''
            parse_sub_tree = '['
            parse_sub_tree += sym
            parse_sub_tree += ', '
            parse_sub_tree += self.create_parse_tree(i, s, y1, sentence_tokens)
            parse_sub_tree += ', '
            parse_sub_tree += self.create_parse_tree(s+1, j, y2, sentence_tokens)
            parse_sub_tree += ']'
            '''

        return parse_sub_tree


def process(train_file_original, train_file_modified, dev_input_file, dev_output_file, threshold_count):
    pcfg_obj = PCFG(train_file_original, train_file_modified, threshold_count)
    pcfg_obj.compute_cfg_frequency_in_train_file(train_file_original)
    pcfg_obj.compute_word_frequency_in_cfg()

    # now replace the rare words with _RARE_
    with open(train_file_modified, 'w') as wfd:
        with open(train_file_original, 'r') as rfd:
            for line in rfd:
                parse_tree = json.loads(line)
                pcfg_obj.replace_infrequent_words_in_parse_tree(parse_tree)
                wfd.write('%s\n' % json.dumps(parse_tree))

    # now compute cfg frequency using the modified train file
    pcfg_obj.compute_cfg_frequency_in_train_file(train_file_modified)

    with open(dev_output_file, 'w') as wfd:
        with open(dev_input_file, 'r') as rfd:
            for line in rfd:
                parse_tree = pcfg_obj.CKY_algorithm(line)
                wfd.write('%s\n' % json.dumps(parse_tree))


if __name__ == '__main__':
    import sys
    # parse_train.dat parse_train_with_rare.dat 5 parse_dev.dat parse_dev.out
    train_file_orig = sys.argv[1]
    train_file_mod = sys.argv[2]
    count_thresh = int(sys.argv[3])
    input_dev_file = sys.argv[4]
    output_dev_file = sys.argv[5]

    process(train_file_orig, train_file_mod, input_dev_file, output_dev_file, count_thresh)

'''
CYK Performance for PA2 part 3
https://class.coursera.org/nlangp-001/forum/thread?thread_id=686
'''