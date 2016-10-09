from collections import defaultdict
from itertools import izip
import random
import json
import ast
import io


class Alignment:
    def __init__(self, foreign_corpus, native_corpus):
        self.expected_count_dict = defaultdict(float)
        self.q_param = dict()
        self.t_param = dict()
        self.foreign_corpus = foreign_corpus
        self.native_corpus = native_corpus
        self.null_word = u'NULL'

    def initialize_parameters(self, ibm_model, distribution='uniform', word_delimiter=' '):
        """Initialization of parameters for the EM algorithm"""
        if ibm_model == 2:
            assert len(self.t_param) > 0, "First load the t_param created by IBM model 1"

        with io.open(self.foreign_corpus, 'r', encoding='utf-8') as fd_foreign, io.open(self.native_corpus, 'r', encoding='utf-8') as fd_native:
            count_line = 0
            for line_foreign, line_native in izip(fd_foreign, fd_native):
                count_line += 1
                if count_line % 1000 == 999:
                    print("\tcount_line[initialize_parameters]: {0}".format(count_line))

                line_foreign = line_foreign.strip()
                line_native = line_native.strip()

                if line_native == '' or line_foreign == '':
                    continue

                tokens_foreign = line_foreign.split(word_delimiter)
                tokens_native = line_native.split(word_delimiter)

                mk = len(tokens_foreign)
                lk = len(tokens_native)

                prob_uniform = 1.0 / (lk + 1)
                for i in range(1, mk+1):
                    word_foreign = tokens_foreign[i-1]

                    for j in range(0, lk+1):
                        if j == 0:
                            word_native = self.null_word
                        else:
                            word_native = tokens_native[j-1]

                        if ibm_model == 2:
                            # t_param has been initialized by running IBM model 1
                            if distribution == 'uniform':
                                self.q_param[(i, lk, mk, j)] = prob_uniform
                            else:
                                self.q_param[(i, lk, mk, j)] = random.random()
                        elif ibm_model == 1:
                            # For IBM model 1, at this stage we are just collecting foreign/native word pairs
                            if word_native not in self.t_param:
                                self.t_param[word_native] = dict()
                            if word_foreign not in self.t_param[word_native]:
                                self.t_param[word_native][word_foreign] = 0.0  # assigning a dummy value

                """
                for i in range(0, lk+1):
                    if i == 0:
                        word_native = self.null_word
                    else:
                        word_native = tokens_native[i-1]

                    if word_native not in self.t_param:
                        self.t_param[word_native] = dict()

                    for j in range(1, mk+1):
                        word_foreign = tokens_foreign[j-1]

                        if word_foreign not in self.t_param[word_native]:
                            self.t_param[word_native][word_foreign] = 0.0
                """

            if ibm_model == 1:
                # Now assign parameter value based on 'distribution'
                if distribution == 'uniform':
                    for word_native in self.t_param:
                        prob_uniform = 1.0 / len(self.t_param[word_native])
                        for word_foreign in self.t_param[word_native]:
                            self.t_param[word_native][word_foreign] = prob_uniform
                elif distribution == 'random':
                    for word_native in self.t_param:
                        for word_foreign in self.t_param[word_native]:
                            self.t_param[word_native][word_foreign] = random.random()
                else:
                    assert False, "incorrect distribution passed"

    def compute_parameters(self, ibm_model, n_iterations):
        self.initialize_parameters(ibm_model)

        for iter_i in range(n_iterations):
            print('iteration #{0}'.format(iter_i+1))
            self.compute_parameters_for_iteration(ibm_model)
            self.recalculate_parameters(ibm_model)
            # self.print_parameters()

        # Now save the parameters in a file
        # self.write_parameters_in_text(t_param_file, q_param_file)

    # compute parameters for an iteration
    def compute_parameters_for_iteration(self, ibm_model, word_delimiter=' '):
        # reset counts to zero
        self.expected_count_dict = defaultdict(float)
        with io.open(self.foreign_corpus, 'r', encoding='utf-8') as fd_foreign, io.open(self.native_corpus, 'r', encoding='utf-8') as fd_native:
            # iterate through each pair of sentences
            count_line = 0
            for line_foreign, line_native in izip(fd_foreign, fd_native):
                count_line += 1
                if count_line % 100 == 99:
                    print("\tcount_line: {0}".format(count_line))

                line_foreign = line_foreign.strip()
                line_native = line_native.strip()
                # print('{0}'.format(line_foreign))
                # print('{0}'.format(line_native))

                if line_native == '' or line_foreign == '':
                    print('\ttranslation missing for line #{0}'.format(count_line))
                    continue

                tokens_foreign = line_foreign.split(word_delimiter)
                tokens_native = line_native.split(word_delimiter)

                mk = len(tokens_foreign)
                lk = len(tokens_native)

                # notation as per lecture notes: 0th index represents null, 1st word's index is 1
                for i in range(1, mk+1):
                    word_foreign = tokens_foreign[i-1]

                    delta_denom = 0.0
                    for j in range(0, lk+1):
                        if j == 0:
                            word_native = self.null_word
                        else:
                            word_native = tokens_native[j-1]

                        if ibm_model == 2:
                            assert ((i, lk, mk, j) in self.q_param), "({0},{1},{2},{3}) missing in q_param".format(i, lk, mk, j)
                        """
                        # initialize t_param, q_param as we see it for first time
                        if (i, lk, mk, j) not in self.q_param:
                            self.q_param[(i, lk, mk, j)] = random.random()

                        if (word_native, word_foreign) not in self.t_param:
                            self.t_param[(word_native, word_foreign)] = random.random()
                        """

                        assert word_native in self.t_param, u"native word: {0} wasn't initialized in t_param".format(word_native)
                        assert word_foreign in self.t_param[word_native], u"foreign word: {0} wasn't initialized in t_param[{1}]".format(word_foreign, word_native)

                        if ibm_model == 1:
                            delta_denom += self.t_param[word_native][word_foreign]
                        else:
                            delta_denom += self.q_param[(i, lk, mk, j)] * self.t_param[word_native][word_foreign]

                    for j in range(0, lk+1):
                        if j == 0:
                            word_native = self.null_word
                        else:
                            word_native = tokens_native[j-1]

                        if ibm_model == 1:
                            delta_k_i_j = self.t_param[word_native][word_foreign] / delta_denom
                        else:
                            delta_k_i_j = self.q_param[(i, lk, mk, j)] * self.t_param[word_native][word_foreign] / delta_denom

                        self.expected_count_dict[word_native] += delta_k_i_j
                        self.expected_count_dict[(word_native, word_foreign)] += delta_k_i_j
                        self.expected_count_dict[(i, lk, mk, j)] += delta_k_i_j
                        self.expected_count_dict[(i, lk, mk)] += delta_k_i_j

                """
                if count_line > 20:
                    break
                """

    def recalculate_parameters(self, ibm_model):
        """t,q parameters are calculated after each iteration"""
        """
        for t_tuple in self.t_param:
            word_native = t_tuple[0]
            self.t_param[t_tuple] = self.expected_count_dict[t_tuple]/self.expected_count_dict[word_native]
        """
        for word_native in self.t_param:
            for word_foreign in self.t_param[word_native]:
                try:
                    self.t_param[word_native][word_foreign] = self.expected_count_dict[(word_native, word_foreign)]/self.expected_count_dict[word_native]
                except ZeroDivisionError:
                    print('\texpected count is zero for native word: {0}'.format(word_native.encode('utf-8')))

        if ibm_model == 2:
            for q_tuple in self.q_param:
                i = q_tuple[0]
                lk = q_tuple[1]
                mk = q_tuple[2]
                j = q_tuple[3]
                self.q_param[q_tuple] = self.expected_count_dict[q_tuple]/self.expected_count_dict[(i, lk, mk)]

    def compute_alignment(self, ibm_model, foreign_dev, native_dev, dev_output, word_delimiter=' '):
        with open(dev_output, 'w') as fd_output:
            with io.open(foreign_dev, 'r', encoding='utf-8') as fd_foreign, io.open(native_dev, 'r', encoding='utf-8') as fd_native:
                # iterate over each pair of sentences
                count_line = 0
                for line_foreign, line_native in izip(fd_foreign, fd_native):
                    count_line += 1
                    if count_line % 100 == 99:
                        print("\tcount_line: {0}".format(count_line))

                    line_foreign = line_foreign.strip()
                    line_native = line_native.strip()

                    if line_native == '' or line_foreign == '':
                        print('\ttranslation missing for line #{0}'.format(count_line))
                        continue

                    tokens_foreign = line_foreign.split(word_delimiter)
                    tokens_native = line_native.split(word_delimiter)

                    mk = len(tokens_foreign)
                    lk = len(tokens_native)

                    # notation as per lecture notes: 0th index represents null, 1st word's index is 1
                    for i in range(1, mk + 1):
                        word_foreign = tokens_foreign[i - 1]

                        # align the foreign word to the English word which has the highest t(f|e) probability
                        max_alignment_prob_j = 0
                        max_alignment_prob = 0

                        if self.null_word in self.t_param:
                            if word_foreign in self.t_param[self.null_word]:
                                if ibm_model == 1:
                                    max_alignment_prob = self.t_param[self.null_word][word_foreign]
                                else:
                                    max_alignment_prob = self.q_param[(i, lk, mk, 0)] * self.t_param[self.null_word][word_foreign]

                        for j in range(1, lk + 1):
                            word_native = tokens_native[j - 1]
                            if word_native in self.t_param and word_foreign in self.t_param[word_native]:
                                if ibm_model == 1:
                                    cur_alignment_prob = self.t_param[word_native][word_foreign]
                                else:
                                    cur_alignment_prob = self.q_param[(i, lk, mk, j)] * self.t_param[word_native][word_foreign]

                                if max_alignment_prob < cur_alignment_prob:
                                    max_alignment_prob_j = j
                                    max_alignment_prob = cur_alignment_prob

                        fd_output.write('{0} {1} {2}\n'.format(count_line, max_alignment_prob_j, i))

    def print_parameters(self):
        print('\tt_params')
        for t_tuple in self.t_param:
            print('\t\t{0} : {1}'.format(t_tuple, self.t_param[t_tuple]))

        print('\tq_params')
        for q_tuple in self.q_param:
            print('\t\t{0} : {1}'.format(q_tuple, self.q_param[q_tuple]))

    def write_parameters_in_text(self, t_param_file, q_param_file):
        with open(t_param_file, 'w') as fd:
            for t_tuple in self.t_param:
                fd.write('{0} : {1}\n'.format(t_tuple, self.t_param[t_tuple]))

        with open(q_param_file, 'w') as fd:
            for q_tuple in self.q_param:
                fd.write('{0} : {1}\n'.format(q_tuple, self.q_param[q_tuple]))

    def write_parameters(self, param_file):
        # self.print_t_param(self.null_word)  # TBD: remove later
        with io.open(param_file, mode='w', encoding='utf-8') as fd:
            # convert the tuple key into string as required by json
            # t_param_dict = {str(k): self.t_param[k] for k in self.t_param}
            q_param_dict = {str(k): self.q_param[k] for k in self.q_param}

            param_json = {'t_param': self.t_param, 'q_param': q_param_dict}
            # json.dump(param_json, fd, ensure_ascii=False)
            fd.write(unicode(json.dumps(param_json, ensure_ascii=False)))

    def print_t_param(self, word_native):
        """For debugging purpose"""
        print self.t_param[word_native]

    def load_parameters(self, param_file):
        """Load parameters saved in json"""
        with io.open(param_file, 'r', encoding='utf-8') as fd:
            param_dict = json.load(fd)
            self.t_param = param_dict['t_param']
            q_param_dict = param_dict['q_param']
            # convert the str key into tuple
            # self.t_param = {ast.literal_eval(k): t_param_dict[k] for k in t_param_dict}
            self.q_param = {ast.literal_eval(k): q_param_dict[k] for k in q_param_dict}
            print('parameters loaded from json file')

if __name__ == '__main__':
    alignment_obj = Alignment('./data/europarl/corpus.es', './data/europarl/corpus.en')
    ibm_model = 2
    train_flag = False

    if train_flag:
        n_iterations = 5
        if ibm_model == 2:
            # First load the t_param computed by IBM model 1
            alignment_obj.load_parameters('./data/output_europarl/param_ibm_model_1.json')

        alignment_obj.compute_parameters(ibm_model, n_iterations)

        if ibm_model == 1:
            alignment_obj.write_parameters('./data/output_europarl/param_ibm_model_1.json')
        else:
            alignment_obj.write_parameters('./data/output_europarl/param_ibm_model_2.json')
    else:
        if ibm_model == 1:
            alignment_obj.load_parameters('./data/output_europarl/param_ibm_model_1.json')
        else:
            alignment_obj.load_parameters('./data/output_europarl/param_ibm_model_2.json')

        alignment_obj.compute_alignment(ibm_model, './data/europarl/dev.es', './data/europarl/dev.en', './data/output_europarl/dev.out')

"""
Note: Implemented in python 2.7
TBD:
    1)  Instead of fixed number of iterations, we can stop based on convergence.
        Perhaps convergence can be checked using log-likelihood function L(t,q) mentioned in page 17 of Prof's note (ibm12.pdf)
        Theorem mentioned in page 18 shows that this function is non-decreasing.

    2)  Follow the approach of initializing t parameters from IBM model 1. Explained in page 20-22 of ibm12.pdf

http://stackoverflow.com/questions/11295171/read-two-textfile-line-by-line-simultaneously-python
http://stackoverflow.com/questions/12337583/saving-dictionary-whose-keys-are-tuples-with-json-python

unicode handling in python 2.7
    https://www.azavea.com/blog/2014/03/24/solving-unicode-problems-in-python-2-7/
    http://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file-in-python (Antony Hatchkins's answer)
"""
