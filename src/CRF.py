# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
import nltk
import string
import numpy as np
import gensim
# from decimal import Decimal
import re
from sklearn.metrics import f1_score
import time


class CRF():
    def __init__(self):
        self.train_size, self.test_size = -1, -1
        self.n_param = 0
        self.num_tags = 3
        self.sentences, self.labels = [], []
        self.start_symbol, self.end_symbol = 'bos', 'eos'
        self.label_index = {'S': 0, 'NKP': 1, 'KP': 2}
        self.index_label = {0: 'S', 1: 'NKP', 2: 'KP'}
        self.start_state, self.end_state = 'S', 'S'
        self.start_state_id = 0
        self.prior_feature_expectation = np.zeros(self.n_param)
        self.theta = np.zeros(self.n_param)
        self.sigma2 = 100.0  # np.inf  # no Gaussian prior
        self.feature_index, self.index_feature = {}, {}
        self.Y_train = []
        self.X_train1 = []
        self.X_train11 = []
        self.X_train2 = []
        self.X_test = []
        self.Y_test = []
        self.Y_test2 = []
        self.X_test1 = []
        self.X_test11 = []

        self.X_test2 = []
        self.sentences2 = []
        self.Y_labels = []

    def remove_sym(self, seq):
        c1 = []
        for w in seq:
            w1 = w.encode("utf-8")
            w2 = re.sub(r"[``'?.!/;:!@#$()-+'']", '', w1)
            c1.append(w2)
        return c1

    def doc_pre(self, seq):
        files = seq.replace('\n', '').replace('\t', ' ')
        tagged_words = nltk.word_tokenize(files)
        punct = set(string.punctuation)
        content = [phrase.lower() for phrase in tagged_words if phrase.lower not in punct]
        content3 = filter(lambda ch: ch not in " ``'?.!/;:!@#$()-+[]'', ", content)
        content33 = self.remove_sym(content3)
        return content33

    def tfidf_seq(self, cant, s, do):
        a1 = {}
        for i in range(len(cant)):
            a1[cant[i][0]] = cant[i][1]
            C = []
            for i in range(len(s)):
                w = s[i]
                idx = do[w]
                if idx in a1.keys():
                    itf = a1[idx]
                else:
                    itf = 0
                C.append(itf)
        return C

    def is_intitle(self, seq, title):
        C = []
        for i in range(len(seq)):
            w = seq[i]
            if w in title:
                ite = 1
            else:
                ite = 0
            C.append(ite)
        return C

    def load_trainingdata(self, filename):

        ###### training sentence
        #        corpus_root = r"C:\\Users\\xiaoleilu2\\Desktop\\2TR\\Hulth2003\\Train1"
        corpus_root = filename
        file_pattern = r'.*\.abstr'
        textlists = nltk.corpus.BracketParseCorpusReader(corpus_root, file_pattern)
        filenames = textlists.fileids()

        CT = []
        for file in filenames:
            raw = textlists.raw(file)
            fulltxt = raw.split('\r\n')
            text = fulltxt[1]
            content = self.doc_pre(text)
            content4 = [ph for ph in content if len(ph) > 1]
            s = ['bos']
            e = ['eos']
            s.extend(content4)
            s.extend(e)
            CT.append(s)
            self.sentences.append(s)

        self.train_size = len(self.sentences)

        #        corpus_root2 = r"C:\\Users\\xiaoleilu2\\Desktop\\2TR\\Hulth2003\\Train1"
        corpus_root2 = filename
        file_pattern2 = r'.*\.uncontr'
        textlists2 = nltk.corpus.BracketParseCorpusReader(corpus_root2, file_pattern2)
        filenames2 = textlists2.fileids()

        CK = []
        for file in filenames2:
            raw = textlists.raw(file)
            fulltxt = raw.split('\r\n')
            text = fulltxt[0]
            kpt = self.doc_pre(text)
            CK.append(kpt)

        Label = []
        for i in range(self.train_size):
            seq = CT[i]
            KP = CK[i]
            label = [0] * len(seq)
            for j in range(len(seq)):
                if seq[j] == 'bos' or seq[j] == 'eos':
                    label[j] = 'S'
                elif seq[j] in KP:
                    label[j] = 'KP'
                else:
                    label[j] = 'NKP'
            Label.append(label)
            self.Y_train.append(label)

        for i in range(len(self.Y_train)):
            y = self.Y_train[i]
            y1 = list()
            for j in range(len(y)):
                y1.append(self.label_index[y[j]])
            self.Y_labels.append(y1)

        #### feature function
        feature_id = 0
        ########## transition funciton
        for tag1 in range(self.num_tags):
            for tag2 in range(self.num_tags):
                feature = ('T', tag1, tag2)
                self.index_feature[feature_id] = feature
                self.feature_index[feature] = feature_id
                feature_id += 1

        ##########  tfidf state feature
        dictionary = gensim.corpora.Dictionary(CT)
        do = dictionary.token2id
        corpus = [dictionary.doc2bow(boc_text) for boc_text in CT]
        tfidf = gensim.models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

        TF = []
        for i in range(self.train_size):
            cant = corpus_tfidf[i]
            s = CT[i]
            seq_t = self.tfidf_seq(cant, s, do)
            TF.append(seq_t)
            self.X_train1.append(seq_t)
            ###avg
            atf = sum(seq_t) / len(seq_t)
            for i in range(len(seq_t)):
                if seq_t[i] <= atf:
                    seq_t[i] = 0
                else:
                    seq_t[i] = 1
            self.X_train11.append(seq_t)

        unift = [0, 1]
        for tf in unift:
            for tag in range(self.num_tags):
                feature = ('TF', tf, tag)
                self.index_feature[feature_id] = feature
                self.feature_index[feature] = feature_id
                feature_id += 1

        ######### is in title state feature
        OTitle = []
        for file in filenames:
            raw = textlists.raw(file)
            fulltxt = raw.split('\r\n')
            text = fulltxt[1]
            content = self.doc_pre(text)
            content4 = [ph for ph in content if len(ph) > 1]
            s = ['bos']
            e = ['eos']
            s.extend(content4)
            s.extend(e)
            ##title
            title = fulltxt[0]
            tc = self.doc_pre(title)
            ### is in title
            C = self.is_intitle(s, tc)
            OTitle.append(C)
            self.X_train2.append(C)

        OT = [0, 1]
        for ot in OT:
            for tag in range(self.num_tags):
                feature = ('OT', ot, tag)
                self.index_feature[feature_id] = feature
                self.feature_index[feature] = feature_id
                feature_id += 1

        self.n_param = len(self.index_feature)
        self.theta = np.ones(self.n_param) / self.n_param
        self.get_prior_feature_expectation()

    def load_testingdata(self, filename):

        ###### training sentence
        #        corpus_root = r"C:\\Users\\xiaoleilu2\\Desktop\\2TR\\Hulth2003\\Test1"
        corpus_root = filename
        file_pattern = r'.*\.abstr'
        textlists = nltk.corpus.BracketParseCorpusReader(corpus_root, file_pattern)
        filenames = textlists.fileids()

        CT = []
        for file in filenames:
            raw = textlists.raw(file)
            fulltxt = raw.split('\r\n')
            text = fulltxt[1]
            content = self.doc_pre(text)
            content4 = [ph for ph in content if len(ph) > 1]
            s = ['bos']
            e = ['eos']
            s.extend(content4)
            s.extend(e)
            CT.append(s)
            self.sentences2.append(s)

        self.test_size = len(self.sentences2)

        corpus_root2 = filename
        file_pattern2 = r'.*\.uncontr'
        textlists2 = nltk.corpus.BracketParseCorpusReader(corpus_root2, file_pattern2)
        filenames2 = textlists2.fileids()

        CK = []
        for file in filenames2:
            raw = textlists.raw(file)
            fulltxt = raw.split('\r\n')
            text = fulltxt[0]
            kpt = self.doc_pre(text)
            CK.append(kpt)

        Label = []
        for i in range(self.test_size):
            seq = CT[i]
            KP = CK[i]
            label = [0] * len(seq)
            for j in range(len(seq)):
                if seq[j] == 'bos' or seq[j] == 'eos':
                    label[j] = 'S'
                elif seq[j] in KP:
                    label[j] = 'KP'
                else:
                    label[j] = 'NKP'
            Label.append(label)
            self.Y_test.append(label)

        for i in range(len(self.Y_test)):
            y = self.Y_test[i]
            y1 = list()
            for j in range(len(y)):
                y1.append(self.label_index[y[j]])
            self.Y_test2.append(y1)

        ##########  tfidf  feature
        dictionary = gensim.corpora.Dictionary(CT)
        do = dictionary.token2id
        corpus = [dictionary.doc2bow(boc_text) for boc_text in CT]
        tfidf = gensim.models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

        TF = []
        for i in range(self.test_size):
            cant = corpus_tfidf[i]
            s = CT[i]
            seq_t = self.tfidf_seq(cant, s, do)
            TF.append(seq_t)
            self.X_test1.append(seq_t)
            ###avg
            atf = sum(seq_t) / len(seq_t)
            for i in range(len(seq_t)):
                if seq_t[i] <= atf:
                    seq_t[i] = 0
                else:
                    seq_t[i] = 1
            self.X_test11.append(seq_t)

        ######### is in title state feature
        OTitle = []
        for file in filenames:
            raw = textlists.raw(file)
            fulltxt = raw.split('\r\n')
            text = fulltxt[1]
            content = self.doc_pre(text)
            content4 = [ph for ph in content if len(ph) > 1]
            s = ['bos']
            e = ['eos']
            s.extend(content4)
            s.extend(e)
            ##title
            title = fulltxt[0]
            tc = self.doc_pre(title)
            ### is in title
            C = self.is_intitle(s, tc)
            OTitle.append(C)
            self.X_test2.append(C)

    def get_prior_feature_expectation(self):
        self.prior_feature_expectation = np.zeros(self.n_param)
        for i in range(self.train_size):
            x, y = self.sentences[i], self.Y_labels[i]
            len_i = len(y) - 2
            f1 = self.X_train11[i]
            f2 = self.X_train2[i]
            for t in xrange(1, len_i + 1):
                tag1, tag2 = y[t], y[t - 1]
                t_feature = [self.feature_index[('T', tag2, tag1)]]
                tf1 = f1[t]
                tf_feature = [self.feature_index[('TF', tf1, tag1)]]
                ot2 = f2[t]
                ot_feature = [self.feature_index[('OT', ot2, tag1)]]
                active_features = t_feature + tf_feature + ot_feature
                self.prior_feature_expectation[active_features] += 1.0
        return self.prior_feature_expectation

    def log_potential_at(self, tf, ot, yt, yt1):
        tag1, tag2 = yt, yt1
        t_feature = [self.feature_index[('T', tag2, tag1)]]
        tf_feature = [self.feature_index[('TF', tf, tag1)]]
        ot_feature = [self.feature_index[('OT', ot, tag1)]]

        active_features = t_feature + tf_feature + ot_feature
        return self.theta[active_features].sum()

    def cal_potential(self, f1, f2, labels_seq):
        res = 0.0
        len_y = len(labels_seq) - 2
        for i in range(1, len_y + 1):
            lc, lp = labels_seq[i], labels_seq[i - 1]
            tf1 = f1[i]
            ot2 = f2[i]
            res += self.log_potential_at(tf1, ot2, lc, lp)
        return res

    def forward_backward(self, f1, f2, y, log_potential=None, return_alpha=False, return_y_star=False):

        def log_sum_exp(arr):
            max_value = np.max(arr)
            return max_value + np.log(np.sum(np.exp(arr - max_value)))

        seq_l = len(y) - 2

        if log_potential is None:
            log_potential = np.zeros((seq_l + 1, self.num_tags, self.num_tags))
            for i in range(seq_l + 1):
                tf1 = f1[i]
                ot2 = f2[i]
                for tag1 in range(self.num_tags):
                    for tag2 in range(self.num_tags):
                        log_potential[i, tag1, tag2] = self.log_potential_at(tf1, ot2, tag1, tag2)
        if return_y_star:
            log_delta = np.zeros((seq_l + 1, self.num_tags))
            pre = np.zeros((seq_l + 1, self.num_tags), dtype='int')

            log_delta[1] = log_potential[1, :, self.start_state_id]
            pre[1] = self.start_state_id  # should be redundant
            for t in xrange(2, seq_l + 1):
                for j in xrange(self.num_tags):
                    best = np.argmax(log_potential[t, j, :] + log_delta[t - 1])
                    pre[t, j] = best
                    log_delta[t, j] = log_potential[t, j, best] + log_delta[t - 1, best]

            y_star = [self.start_state_id] * (seq_l + 1)
            y_star[seq_l] = np.argmax(log_delta[seq_l])
            for t in xrange(seq_l - 1, 0, -1):
                y_star[t] = pre[t + 1, y_star[t + 1]]
            y_star = y_star[1:]
        else:
            y_star = None

        if return_alpha:
            # Forward propagation
            log_alpha = np.zeros((seq_l + 1, self.num_tags))
            log_alpha[1] = log_potential[1, :, self.start_state_id]
            for t in xrange(2, seq_l + 1):
                for j in xrange(self.num_tags):
                    log_alpha[t, j] = log_sum_exp(log_potential[t, j, :] + log_alpha[t - 1])
        else:
            log_alpha = None

        # Backward propagation
        log_beta = np.zeros((seq_l + 1, self.num_tags))
        log_beta[seq_l] = 0.0
        for t in xrange(seq_l - 1, 0, -1):
            for i in xrange(self.num_tags):
                log_beta[t, i] = log_sum_exp(log_potential[t + 1, :, i] + log_beta[t + 1])
        log_beta0 = log_sum_exp(log_potential[1, :, self.start_state_id] + log_beta[1])
        log_z = log_beta0

        ####### likelihood prob
        log_pot = self.cal_potential(f1, f2, y)
        cll = log_pot - log_z
        return log_alpha, log_beta, log_z, cll, y_star

    def PLL(self):
        res = 0.0
        for i in xrange(self.train_size):
            y = self.Y_labels[i]
            f1 = self.X_train11[i]
            f2 = self.X_train2[i]
            _, _, _, cll, _ = self.forward_backward(f1, f2, y)
            res += cll
        res -= np.dot(self.theta, self.theta) / (2.0 * self.sigma2)
        return res

    def model_exp(self, i):

        y = self.Y_labels[i]
        f1 = self.X_train11[i]
        f2 = self.X_train2[i]
        len_i = len(y) - 2
        log_potential = np.zeros((len_i + 1, self.num_tags, self.num_tags))
        for t in range(len_i + 1):
            tf1 = f1[t]
            ot2 = f2[t]
            for yt in range(self.num_tags):
                for yt1 in range(self.num_tags):
                    log_potential[t, yt, yt1] = self.log_potential_at(tf1, ot2, yt, yt1)
        log_alpha, log_beta, log_z, cll, _ = self.forward_backward(f1, f2, y, log_potential, return_alpha=True)

        log_p1 = np.zeros((len_i + 1, self.num_tags, self.num_tags))
        for t in range(1, len_i + 1):
            for y1 in range(self.num_tags):
                for y2 in range(self.num_tags):
                    if t == 1 and y1 != self.start_state_id:
                        continue
                    log_p1[t, y1, y2] = log_alpha[t - 1, y1] + log_potential[t, y2, y1] + log_beta[t, y2] - log_z

        log_p = np.exp(log_p1)
        for yt1 in xrange(self.num_tags):
            if yt1 == self.start_state_id:
                continue
            for yt in xrange(self.num_tags):
                log_p[1, yt1, yt] = 0.0
        del log_alpha, log_beta

        model_expectation_for_example_seq = np.zeros(self.theta.shape)

        '''
        for i in range(1,len_i+1):
            tf1 = f1[i]
            ot2 = f2[i]
            for tag2 in range(self.num_tags):
                for tag1 in range(self.num_tags):
                    t_feature = [self.feature_index[('T',tag2,tag1)]]
                    tf_feature = [self.feature_index[('TF', tf1,tag1)]]
                    ot_feature = [self.feature_index[('OT'),ot2,tag1]]
                    active_features = t_feature + tf_feature + ot_feature
                    model_expectation_for_example_seq[active_features] += log_p[t,tag2,tag1]
        '''

        for t in xrange(1, len_i + 1):
            tag1, tag2 = y[t], y[t - 1]
            t_feature = [self.feature_index[('T', tag2, tag1)]]
            tf1 = f1[t]
            tf_feature = [self.feature_index[('TF', tf1, tag1)]]
            ot2 = f2[t]
            ot_feature = [self.feature_index[('OT', ot2, tag1)]]
            active_features = t_feature + tf_feature + ot_feature
            model_expectation_for_example_seq[active_features] += log_p[t, tag2, tag1]

        for i in range(len(model_expectation_for_example_seq)):
            round(model_expectation_for_example_seq[i], 5)

        return model_expectation_for_example_seq

    def PLL_gra(self):
        gradient = self.prior_feature_expectation - self.theta / self.sigma2
        for i in range(self.train_size):
            gradient -= self.model_exp(i)
        for i in range(len(gradient)):
            gradient[i] = round(gradient[i], 5)
        return gradient

    def ncll(self, theta):
        self.theta = theta
        return -self.PLL()

    def ncll_prime(self, theta):
        self.theta = theta
        return -self.PLL_gra()

    def fit(self):
        res = minimize(self.ncll, self.theta, method='BFGS',
                       jac=self.ncll_prime, options={'disp': True, 'maxiter': 100})
        if res.success:
            self.theta = res.x
        else:
            print 'Failed to optimize CLL'

    def predict(self):
        res = []
        for i in xrange(len(self.sentences2)):
            x = self.sentences2[i]
            y = [self.start_state_id] * len(x)  # dummy y
            f1 = self.X_test11[i]
            f2 = self.X_test2[i]
            _, _, _, _, y_star = self.forward_backward(f1, f2, y, return_y_star=True)
            res.append(map(lambda idx: self.index_label[idx], y_star))
        # print 'res:', res
        return res



