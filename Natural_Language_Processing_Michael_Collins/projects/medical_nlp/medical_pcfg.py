import xml.etree.ElementTree as ET
import glob
import random
import sys
import os
import json
from count_cfg_freq import Counts
from collections import defaultdict
from copy import deepcopy

__date__ = '$Aug 12, 2016'
__author__ = 'Kaushik Acharya'


class PCFG:
    def __init__(self, rare_count_threshold=5):
        self.rare_count_threshold = rare_count_threshold
        self.cfg_counter = Counts()
        self.word_freq = defaultdict(int)

    # This tree is in Penn Tree Bank format not in Chomsky Normal Form
    def create_parse_tree(self, elem_root_subtree):
        if 'null' in elem_root_subtree.attrib:
            return None
        subtree_array = [elem_root_subtree.attrib['cat']]

        if elem_root_subtree.tag == 'cons':
            for child in elem_root_subtree:
                child_subtree_array = self.create_parse_tree(child)
                if child_subtree_array is not None:
                    subtree_array.append(child_subtree_array)
        elif elem_root_subtree.tag == 'tok':
            subtree_array.append(elem_root_subtree.text)
        else:
            # TBD: print error message
            pass

        return subtree_array

    # right-factored
    # TBD: Provide option for left-factored too
    # TBD: Handle null as mentioned in section 6.3 Null Elements in GENIA corpus manual
    def convert_PTB_to_CNF(self, tree_array):
        if len(tree_array) > 3:
            # convert multi-child tree into tree with two children
            subtree_array = [tree_array[0]]
            for i in range(2, len(tree_array)):
                subtree_array.append(tree_array[i])
            tree_array[2] = subtree_array
            # tree_array[2] = [tree_array[0], tree_array[2], tree_array[3:]]
            del tree_array[3:]
            self.convert_PTB_to_CNF(tree_array)
        elif len(tree_array) == 3:
            # root of both the children should be non-terminals
            assert (type(tree_array[1]) is list), "expected list for left child: {0}".format(tree_array[1])
            assert (type(tree_array[2]) is list), "expected list for right child: {0}".format(tree_array[2])
            # convert left child into CNF form if its not already in that form
            self.convert_PTB_to_CNF(tree_array[1])
            # convert right child into CNF form if its not already in that form
            self.convert_PTB_to_CNF(tree_array[2])
        elif (len(tree_array) == 2) and (type(tree_array[1]) is list):
            # Form: X->Y where X,Y are non-terminals
            tree_array[0] = tree_array[0] + '+' + tree_array[1][0]
            if type(tree_array[1][1]) is list:
                subtree_array = tree_array[1][1:]
                del tree_array[1:]
                for subtree in subtree_array:
                    tree_array.append(subtree)
                self.convert_PTB_to_CNF(tree_array)
            else:
                # e.g. [NP [DET There]]
                tree_array[1] = tree_array[1][1]

    # Extract parse tree from the xml files. These parse trees are in Penn TreeBank format.
    # Randomly assign xml files into training and validation set.
    # Write the parse trees after converting them into Chomsky Normal Form
    def create_train_and_test_parse_tree(self, treebank_folder, train_key_file, test_file, test_key_file, n_train_file):
        file_list = glob.glob(treebank_folder+'/'+'*.xml')
        # randomly select n_train_file for training and rest for testing
        train_index_list = random.sample(range(len(file_list)), n_train_file)

        # write the train file
        count_parse_tree = 0
        with open(train_key_file, 'w') as fd:
            for file_i in train_index_list:
                train_file = file_list[file_i]
                try:
                    count_parse_tree_in_xml = self.write_parse_tree(fd, train_file)
                    count_parse_tree += count_parse_tree_in_xml
                    print('train file: {0} :: parse tree count till now: {1}'.format(train_file, count_parse_tree))
                except:
                    err = sys.exc_info()[0]
                    print('Error in {0}: {1}'.format(train_file, err))

        # test files are actually more of validation set
        # write the test parse tree file
        failure_parse_file_list = []
        with open(test_key_file, 'w') as fd:
            # optimize the search to O(n) by sorting the train_index_list first
            for file_i in range(len(file_list)):
                if file_i not in train_index_list:
                    test_xml_file = file_list[file_i]
                    try:
                        self.write_parse_tree(fd, test_xml_file)
                    except:
                        err = sys.exc_info()
                        print('Error in {0}: {1}'.format(test_xml_file, err))
                        parts_filename = os.path.split(test_xml_file)
                        failure_parse_file_list.append(parts_filename[1])

        # Now write the test sentences
        with open(test_file, 'w') as fd:
            # optimize the search to O(n) by sorting the train_index_list first
            for file_i in range(len(file_list)):
                if file_i not in train_index_list:
                    test_xml_file = file_list[file_i]
                    parts_filename = os.path.split(test_xml_file)
                    if parts_filename[1] in failure_parse_file_list:
                        print('ignoring sentence extraction from {0}'.format(test_xml_file))
                        continue
                    try:
                        self.write_sentences(fd, test_xml_file)
                    except:
                        err = sys.exc_info()[0]
                        print('Error in extracting sentence from {0}: {1}'.format(test_xml_file, err))

    # Create parse trees from xml file and write in the train/test file (fd)
    def write_parse_tree(self, fd, xml_filename):
        tree = ET.parse(xml_filename)
        root = tree.getroot()

        count_parse_tree_in_xml = 0
        # reading the sentences only from the section: AbstractText
        for abstractText in root.iter('AbstractText'):
            # iterate over each of the sentences
            for sentence in abstractText.iter('sentence'):
                # TBD: Following should have a single iteration
                #       Need to check if the root is an actual root tag e.g. 'S'
                for sentence_root in sentence:
                    tree_ptb_array = self.create_parse_tree(sentence_root)
                    tree_cnf_array = deepcopy(tree_ptb_array)
                    self.convert_PTB_to_CNF(tree_cnf_array)
                    # convert string into json (this converts single quotes to double quotes)
                    # required due to failure in json load of tree in count_cfg_freq.py
                    tree_cnf_json = json.dumps(tree_cnf_array)
                    fd.write('{0}\n'.format(tree_cnf_json))
                    count_parse_tree_in_xml += 1

        return count_parse_tree_in_xml

    # Extract sentences from xml file and write in fd
    @staticmethod
    def write_sentences(fd, xml_filename):
        tree = ET.parse(xml_filename)
        root = tree.getroot()

        for abstractText in root.iter('AbstractText'):
            for sentence in abstractText.iter('sentence'):
                token_array = []
                for token in sentence.iter('tok'):
                    token_array.append(token.text)
                fd.write('{0}\n'.format(' '.join(token_array)))

    def create_train_with_rare(self, orig_train_key_file, mod_train_key_file):
        # First check if self.word_freq is already populated or not
        if len(self.cfg_counter.unary) == 0:
            self.compute_cfg_frequency_in_train_file(orig_train_key_file)
        if len(self.word_freq) == 0:
            self.compute_word_frequency_in_cfg()

        # Now iterate through each parse tree and replace the rare word with rare symbol
        # Write the changed parse trees into new train file
        count_parse_tree = 0
        with open(mod_train_key_file, 'w') as wfd:
            with open(orig_train_key_file, 'r') as rfd:
                for line in rfd:
                    count_parse_tree += 1
                    tree = json.loads(line)
                    try:
                        self.replace_infrequent_words_in_parse_tree(tree)
                        tree_json = json.dumps(tree)
                        wfd.write('{0}\n'.format(tree_json))
                    except:
                        print('Error: create_train_with_rare(): parse tree # {0} :: line: {1}\n'.format(count_parse_tree, line))

    def compute_cfg_frequency_in_train_file(self, train_file):
        self.cfg_counter = Counts()
        with open(train_file, 'r') as fd:
            for line in fd:
                try:
                    cfg_tree = json.loads(line)
                    self.cfg_counter.count(cfg_tree)
                except:
                    print('Error: compute_cfg_frequency_in_train_file(): line: {0}'.format(line))

    def compute_word_frequency_in_cfg(self):
        self.word_freq = defaultdict(int)

        # Terminal word can be assigned to multiple non-Terminals
        for (sym, word), count in self.cfg_counter.unary.iteritems():
            self.word_freq[word] += count

    def is_rare_word(self, word):
        return self.word_freq[word.lower()] < self.rare_count_threshold

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
            assert (type(tree[1]) is not list), "expected string: {0}".format(tree[1])
            if self.is_rare_word(word):
                tree[1] = '_RARE_'

    # x -> y z
    # x,y,z are non-terminals
    def compute_binary_parameter(self, x, y, z):
        key = (x, y, z)
        if key not in self.cfg_counter.binary:
            return 0.0
        else:
            return self.cfg_counter.binary[key]*1.0/self.cfg_counter.nonterm[x]

    # x -> w
    # x: non terminal
    # w: terminal
    def compute_unary_parameter(self, x, w):
        key = (x, w)
        if key not in self.cfg_counter.unary:
            return 0.0
        else:
            return self.cfg_counter.unary[key]*1.0/self.cfg_counter.nonterm[x]

    def compute_parse_tree_using_cky_algorithm(self, sentence):
        tokens = sentence.strip().split(' ')
        n = len(tokens)

        # replace the rare token with rare symbol
        for i in range(n):
            if self.is_rare_word(tokens[i]):
                tokens[i] = '_RARE_'.lower()
            else:
                tokens[i] = tokens[i].lower()

        pi = {}  # store the matrix in dict form
        bp = {}

        # initialize the dict
        for i in range(n):
            pi[i] = {}
            bp[i] = {}

            for j in range(i, n):
                pi[i][j] = {}
                bp[i][j] = {}

                if i == j:
                    # initialize pi for x->w for each of the word
                    for x in self.cfg_counter.nonterm:
                        pi[i][i][x] = self.compute_unary_parameter(x, tokens[i])
                        bp[i][i][x] = None
                else:
                    for x in self.cfg_counter.nonterm:
                        pi[i][j][x] = 0.0
                        bp[i][j][x] = None

        # Now for x -> y z where x,y,z are non-terminals
        for l in range(1, n):
            for i in range(n-l):
                j = i + l
                # Here we only consider the (x,y,z) tuple which we have seen in training data
                for (x, y, z) in self.cfg_counter.binary.keys():
                    q_x_to_yz = self.compute_binary_parameter(x, y, z)

                    if q_x_to_yz <= pi[i][j][x]:
                        # current x -> y z  can't give better probability than already computed max prob with
                        # non-terminal x spanning words i..j inclusive
                        continue

                    max_arg_s, max_val_s_i_j = self.compute_best_split(pi, i, j, y, z)

                    # now check if the current value of pi[i][j][x] is better than the value computed earlier
                    val = q_x_to_yz * max_val_s_i_j
                    if pi[i][j][x] <= val:
                        pi[i][j][x] = val
                        bp[i][j][x] = [max_arg_s, y, z]

                """
                # In the following we try all the (x,y,z) combinations
                # This is a slower process
                for x in self.cfg_counter.nonterm:
                    max_pi_i_j_x = 0.0
                    # [s,y,z]
                    arg_max_pi = [None, None, None]

                    for y in self.cfg_counter.nonterm:
                        for z in self.cfg_counter.nonterm:
                            q_x_to_yz = self.compute_binary_parameter(x, y, z)

                            max_val_s_i_j = 0.0
                            max_arg_s = None
                            for s in range(i, j):
                                val = pi[i][s][y] * pi[s+1][j][z]
                                if max_val_s_i_j < val:
                                    max_val_s_i_j = val
                                    max_arg_s = s

                            val = q_x_to_yz * max_val_s_i_j
                            if max_pi_i_j_x < val:
                                max_pi_i_j_x = val
                                arg_max_pi = [max_arg_s, y, z]

                    pi[i][j][x] = max_pi_i_j_x
                    bp[i][j][x] = arg_max_pi
                """

        assert (pi[0][n-1]['S'] > 0), "pi[0][{0}]['S'] is zero".format(n-1)
        # split the sentence into tokens again. In the beginning of the function we had replaced the rare tokens with rare symbol
        tokens = sentence.strip().split(' ')
        best_parse_tree = self.create_parse_tree_from_backpointers(tokens, bp, 0, n-1, 'S')
        best_prob = pi[0][n-1]['S']

        return best_parse_tree, best_prob

    # For the given x -> y z for the words spanning i,..,j inclusive, compute the best split
    # Select s which maximizes pi(i,s,y)*pi(s+1,j,z)
    @staticmethod
    def compute_best_split(pi, i, j, y, z):
        max_val_s_i_j = 0.0
        max_arg_s = None
        for s in range(i, j):
            val = pi[i][s][y] * pi[s + 1][j][z]
            if max_val_s_i_j <= val:
                max_val_s_i_j = val
                max_arg_s = s

        return max_arg_s, max_val_s_i_j

    def create_parse_tree_from_backpointers(self, tokens, bp, i, j, x):
        if i == j:
            return [x, tokens[i]]

        assert (bp[i][j][x] is not None), "bp[{0}][{1}][{2}] is None".format(i, j, x)
        split_index, y, z = bp[i][j][x]

        parse_tree_left_child = self.create_parse_tree_from_backpointers(tokens, bp, i, split_index, y)
        parse_tree_right_child = self.create_parse_tree_from_backpointers(tokens, bp, split_index+1, j, z)

        parse_tree = [x, parse_tree_left_child, parse_tree_right_child]

        return parse_tree

    def compute_parse_tree_for_test_sentences(self, test_sentence_file, test_key_file):
        with open(test_key_file, 'w') as wfd:
            with open(test_sentence_file, 'r') as rfd:
                sent_i = 0
                for sentence in rfd:
                    sent_i += 1
                    if sent_i % 20 == 19:
                        print('sent_i: {0}'.format(sent_i))

                    try:
                        tree, prob = self.compute_parse_tree_using_cky_algorithm(sentence)
                        # convert into json
                        tree_json = json.dumps(tree)
                        wfd.write('{0}\n'.format(tree_json))
                    except:
                        err = sys.exc_info()[0]
                        print('Error: compute_parse_tree_for_test_sentences(): sent_i: {0} :: sentence: {1} :: \
                        error: {2}\n'.format(sent_i, sentence, err))


def test_parse(xml_filename):
    # tree = ET.parse('C:/Users/310246390/Desktop/trial.xml')
    tree = ET.parse(xml_filename)
    root = tree.getroot()

    for abstractText in root.iter('AbstractText'):
        for sentence in abstractText.iter('sentence'):
            for sentence_root in sentence:
                pcfg_obj = PCFG()
                tree_ptb_array = pcfg_obj.create_parse_tree(sentence_root)
                tree_cnf_array = deepcopy(tree_ptb_array)
                pcfg_obj.convert_PTB_to_CNF(tree_cnf_array)
                # convert single quotes to double quotes
                # required due to failure in json load of tree in count_cfg_freq.py
                tree_cnf_json = json.dumps(tree_cnf_array)
                print('{0}: {1}'.format(sentence.attrib['id'], tree_cnf_json))
                gh = 0


"""
http://stackoverflow.com/questions/1835787/what-is-the-range-of-values-a-float-can-have-in-python
Look at answers of steveha and tzot:
    Actually, you can probably get numbers smaller than 1e-308 via denormals, but there is a significant performance hit
    to this. I found that Python is able to handle 1e-324 but underflows on 1e-325 and returns 0.0 as the value
"""
def test_precision():
    val = 1.0
    mult = 0.1
    for i in range(1, 500):
        val *= mult
        print('i: {0} :: val: {1}'.format(i, val))
        if val == 0.0:
            print('break at i: {0}'.format(i))
            break

if __name__ == '__main__':
    # filename = 'C:/Users/310246390/Desktop/trial2.xml'
    # filename = 'C:/KA/data/NLP/GENIA/GENIA_treebank_v1/1516825.xml'
    # test_parse(filename)

    # test_precision()

    pcfg_obj = PCFG()
    folder_treebank = 'C:/KA/data/NLP/GENIA/GENIA_treebank_v1'
    file_train_key = './data/parse_train.dat'
    file_train_xml = './data/parse_train.xml'
    file_train_with_rare_key = './data/parse_train_with_rare.dat'
    file_input_dev = './data/parse_dev.dat'  # This contains list of sentences
    file_key_dev = './data/parse_dev.key'
    file_out_dev = './data/parse_dev.out'
    n_file_train = 1000
    pcfg_obj.create_train_and_test_parse_tree(folder_treebank, file_train_key, file_input_dev, file_key_dev, n_file_train)

    pcfg_obj.compute_cfg_frequency_in_train_file(file_train_key)
    pcfg_obj.compute_word_frequency_in_cfg()
    pcfg_obj.create_train_with_rare(file_train_key, file_train_with_rare_key)
    # Now re-compute cfg frequencies
    pcfg_obj.compute_cfg_frequency_in_train_file(file_train_with_rare_key)
    pcfg_obj.compute_parse_tree_for_test_sentences(file_input_dev, file_out_dev)

"""
TBD: use sum of log probabilities instead of multiplication of probabilities to avoid float under-flow
     In psuedocode of Yoshimasa Tsuruoka's paper: Iterative CKY parsing for Probabilistic Context-Free Grammars, they
     have used sum of log probabilities

To transform any parse tree into Chomsky Normal Form:
    http://www.nltk.org/_modules/nltk/treetransforms.html

http://stackoverflow.com/questions/4162642/python-single-vs-double-quotes-in-json (cowboybkit's answer)
http://stackoverflow.com/questions/8744113/python-list-by-value-not-by-reference (joaquin's answer)
"""