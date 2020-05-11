#! /usr/bin/python

__author__ = 'Kaushik Acharya'
__date__ = "$Jun 29, 2014"

from count_freqs import Hmm
import sys
from collections import defaultdict
from math import log
import re


def enum(**enums):
    # type creates a dynamic form of the class statement
    return type('Enum', (object,), enums)


class Emission:
    def __init__(self, original_train_file, modified_train_file, count_threshold=5, categorize_rare=False, epsilon=1e-07):
        # in modified_train_file we map infrequent words to a common class
        self.original_train_file = original_train_file
        self.modified_train_file = modified_train_file
        self.count_threshold = count_threshold
        self.emission_counts = defaultdict(int)
        self.ne_tags = []
        self.categorize_rare = categorize_rare
        self.rare_type = enum(NUMERIC=0, ALL_CAPS=1, LAST_CAPS=2, RARE=3)
        self.epsilon = epsilon

    def train_ngram_and_emission_freq_from_corpus_file(self, corpus_file):
        counter = Hmm(3)
        counter.train(corpus_file)
        self.emission_counts = counter.emission_counts
        self.ngram_counts = counter.ngram_counts

    def collect_ne_tags(self):
        for ngram in self.ngram_counts[0]:
            self.ne_tags.append(ngram[0])

    def is_rare_word(self, word):
        freq_count = 0
        for ne_tag in self.ne_tags:
            elem = (word, ne_tag)
            freq_count += self.emission_counts[elem]

        return freq_count < self.count_threshold

    def type_rare_word(self, word):
        # Note: the requirement of ^ and $
        # http://stackoverflow.com/questions/9999726/python-regular-expression-matching-either-all-uppercase-letter-and-number-or-jus
        # http://stackoverflow.com/questions/14471177/python-check-if-the-last-characters-in-a-string-are-numbers#
        if re.search(r'\d', word) is not None:
            return self.rare_type.NUMERIC
        elif re.match(r'^[A-Z]+$', word) is not None:
            return self.rare_type.ALL_CAPS
        elif re.search(r'[A-Z]+$', word) is not None:
            return self.rare_type.LAST_CAPS
        else:
            return self.rare_type.RARE

    def rare_word(self, word):
        if self.categorize_rare is False:
            return '_RARE_'
        else:
            type_rare = self.type_rare_word(word)
            if type_rare == self.rare_type.NUMERIC:
                return '_RARE_NUMERIC_'
            elif type_rare == self.rare_type.ALL_CAPS:
                return '_RARE_ALL_CAPS_'
            elif type_rare == self.rare_type.LAST_CAPS:
                return '_RARE_LAST_CAPS_'
            else:
                return '_RARE_'

    def assign_rare_symbol_to_infrequent_words(self):
        with open(self.modified_train_file, 'w') as wfd:
            with open(self.original_train_file, 'r') as rfd:
                for line in rfd:
                    tokens = line.split()
                    if len(tokens) == 0:
                        wfd.write('%s' % line)  # blank line
                        continue
                    word = tokens[0]
                    ne_tag = tokens[1]
                    if self.is_rare_word(word):
                        word = self.rare_word(word)
                    wfd.write('%s %s\n' % (word, ne_tag))

    def gene_tagger_using_emission_param(self, word):
        if self.is_rare_word(word):
            word = self.rare_word(word)

        max_emission_parameter = 0
        max_emission_tag = None
        for ne_tag in self.ne_tags:
            emission_parameter = self.compute_emission_parameter(word, ne_tag)
            if max_emission_parameter < emission_parameter:
                max_emission_parameter = emission_parameter
                max_emission_tag = ne_tag

        assert (max_emission_tag is not None), "max_emission_parameter not computed"
        return max_emission_tag

    def compute_emission_parameter(self, word, ne_tag):
        elem = (word, ne_tag)
        freq_emission = self.emission_counts[elem]
        freq_tag = self.ngram_counts[0][(ne_tag,)]

        return freq_emission*1.0/freq_tag

    def compute_trigram_transition_parameter(self, ngram):
        if self.ngram_counts[1][ngram[0:2]] == 0:
            return self.epsilon
        if self.ngram_counts[2][ngram] == 0:
            return self.epsilon
        return self.ngram_counts[2][ngram]*1.0/self.ngram_counts[1][ngram[0:2]]

    def assign_gene_tag_using_emission_param(self, dev_input_file, dev_output_file):
        with open(dev_output_file, 'w') as wfd:
            with open(dev_input_file, 'r') as rfd:
                for line in rfd:
                    tokens = line.split()
                    if len(tokens) == 0:
                        wfd.write('%s'%(line))  # blank line
                        continue
                    word = tokens[0]
                    ne_tag = self.gene_tagger_using_emission_param(word)
                    wfd.write('%s %s\n' % (word, ne_tag))

    def assign_gene_tag_using_viterbi(self, dev_input_file, dev_output_file):
        with open(dev_output_file, 'w') as wfd:
            with open(dev_input_file, 'r') as rfd:
                sentence_tokens = []
                for line in rfd:
                    tokens = line.split()
                    if len(tokens) == 0:
                        final_tag_sequence = self.compute_max_prob_tag_sequence_for_all_pos(sentence_tokens)
                        # print 'tag sequence: ', self.final_tag_sequence
                        for token_i in range(0, len(sentence_tokens)):
                            wfd.write('%s %s\n' % (sentence_tokens[token_i], final_tag_sequence[token_i]))
                        wfd.write('%s' % (line))  # blank line
                        sentence_tokens = []
                        continue
                    word = tokens[0]
                    sentence_tokens.append(word)

    def max_prob_tag_sequence_at_pos(self, k, u, v, word_at_pos_k):
        # returns log prob
        # values assigned in dynamic programming table
        # output: max prob of a tag sequence ending in tags u,v at position k  i.e. u at position (k-1) and v at pos k
        arg_max_prob_tag = str()
        if k in [1, 2]:
            # w belongs to {*}
            bigram = tuple(['*', u])
            max_prob_tag_seq_at_prev_pos = self.max_prob_tag_sequence[k-1][bigram]
            trigram = tuple(['*', u, v])
            trigram_transition_parameter = self.compute_trigram_transition_parameter(trigram)
            # max_prob_tag_seq = max_prob_tag_seq_at_prev_pos * trigram_transition_parameter
            max_prob_tag_seq = max_prob_tag_seq_at_prev_pos + log(trigram_transition_parameter)
        else:
            # w belongs to S
            max_prob_tag_seq = float('-inf')
            for w in self.ne_tags:
                bigram = tuple([w, u])
                max_prob_tag_seq_at_prev_pos = self.max_prob_tag_sequence[k-1][bigram]
                trigram = tuple([w, u, v])
                trigram_transition_parameter = self.compute_trigram_transition_parameter(trigram)
                # prob_tag_seq = max_prob_tag_seq_at_prev_pos * trigram_transition_parameter
                prob_tag_seq = max_prob_tag_seq_at_prev_pos + log(trigram_transition_parameter)
                if max_prob_tag_seq < prob_tag_seq:
                    max_prob_tag_seq = prob_tag_seq
                    arg_max_prob_tag = w

        # now multiply with emission parameter
        if self.is_rare_word(word_at_pos_k):
            word_at_pos_k = self.rare_word(word_at_pos_k)  # '_RARE_'
        # max_prob_tag_seq *= self.compute_emission_parameter(word_at_pos_k, v)
        prob_emission = self.compute_emission_parameter(word_at_pos_k, v)
        if prob_emission == 0:
            max_prob_tag_seq = float('-inf')
        else:
            max_prob_tag_seq += log(prob_emission)

        bigram = tuple([u, v])
        self.max_prob_tag_sequence[k][bigram] = max_prob_tag_seq
        self.back_pointer[k][bigram] = arg_max_prob_tag

    def compute_max_prob_tag_sequence_for_all_pos(self, sentence_tokens):
        # log prob
        self.max_prob_tag_sequence = [defaultdict(int) for i in xrange(len(sentence_tokens)+2)]
        self.back_pointer = [defaultdict(str) for i in xrange(len(sentence_tokens)+2)]
        # base case
        bigram = tuple(['*', '*'])
        self.max_prob_tag_sequence[0][bigram] = log(1.0)

        for k in range(1, len(sentence_tokens)+1):
            if k < 2:
                tag_list_for_pos_u = ['*']
            else:
                tag_list_for_pos_u = self.ne_tags

            # u belongs to S(k-1) and v belongs to S(k)
            for u in tag_list_for_pos_u:
                for v in self.ne_tags:
                    self.max_prob_tag_sequence_at_pos(k, u, v, sentence_tokens[k-1])

        # Now assign the tag sequence backwards
        max_prob = float('-inf')
        # Note: final_tag_sequence[0,1, ...,(n-1)] whereas
        #       self.max_prob_tag_sequence and self.back_pointer are 1-based indexed
        final_tag_sequence = [str() for i in xrange(len(sentence_tokens))]
        # print 'sentence_tokens: ', sentence_tokens
        # print 'length(final_tag_sequence): ', len(final_tag_sequence)
        for u in tag_list_for_pos_u:
            for v in self.ne_tags:
                bigram = tuple([u, v])
                trigram = tuple([u, v, 'STOP'])
                prob = self.max_prob_tag_sequence[len(sentence_tokens)][bigram] + \
                       log(self.compute_trigram_transition_parameter(trigram))
                if max_prob < prob:
                    final_tag_sequence[len(sentence_tokens)-2] = u
                    final_tag_sequence[len(sentence_tokens)-1] = v
                    max_prob = prob

        for k in range(len(sentence_tokens)-3, -1, -1):
            bigram = tuple([final_tag_sequence[k+1], final_tag_sequence[k+2]])
            final_tag_sequence[k] = self.back_pointer[k+3][bigram]

        return final_tag_sequence

if __name__ == "__main__":
    original_train_file = sys.argv[1]
    modified_train_file = sys.argv[2]
    count_threshold = int(sys.argv[3])
    dev_input_file = sys.argv[4]
    dev_output_file = sys.argv[5]
    part_no = int(sys.argv[6])

    if part_no == 3:
        categorize_rare_flag = True
    else:
        categorize_rare_flag = False

    emission = Emission(original_train_file, modified_train_file, count_threshold, categorize_rare_flag)
    with open(original_train_file, 'r') as fd:
        emission.train_ngram_and_emission_freq_from_corpus_file(fd)

    emission.collect_ne_tags()
    emission.assign_rare_symbol_to_infrequent_words()

    # re-calculate counts again
    with open(modified_train_file, 'r') as fd:
        emission.train_ngram_and_emission_freq_from_corpus_file(fd)

    if part_no == 1:
        emission.assign_gene_tag_using_emission_param(dev_input_file, dev_output_file)
    elif part_no in [2, 3]:
        emission.assign_gene_tag_using_viterbi(dev_input_file, dev_output_file)


'''
Need for log transform, though staff claims that's not mandatory.
https://class.coursera.org/nlangp-001/forum/thread?thread_id=505

http://stackoverflow.com/questions/15721363/preserve-python-tuples-with-json

Enums:
    http://www.pythonexamples.org/2011/01/12/how-to-create-an-enum-in-python/
    http://www.pythoncentral.io/how-to-implement-an-enum-in-python/

Regular expressions:
    http://www.tutorialspoint.com/python/python_reg_expressions.htm
'''

'''
    corpus_file = sys.argv[1]
    counter = Hmm(3)
    with open(corpus_file) as fd:
        counter.read_counts(fd)

    # test
    elem = ("Sls1p", "I-GENE")
    print "emission count: {0}".format(counter.emission_counts[elem])
    ngram = tuple(elem[1:])
    print "tag count: {0}".format(counter.ngram_counts[0][ngram])
    elem = ("Sls1p", "O")
    if elem in counter.emission_counts:
        print "present with another tag: {0}".format(elem[1])
    else:
        print "absent with another tag: {0}".format(elem[1])

    # printing all the symbols which have both the tag
    for elem in counter.emission_counts.keys():
        if elem[1] ==  "I-GENE":
            other_tag = "O"
        else:
            other_tag = "I-GENE"
        other_elem = (elem[0], other_tag)
        if other_elem in counter.emission_counts:
            print 'symbol {0} present with both tags'.format(elem[0])

    count_key = 0
    for elem in counter.emission_counts.keys():
        print '{0} : {1} : {2}'.format(count_key,elem[0],counter.emission_counts[elem])
        count_key += 1
        if count_key > 6:
            break
'''