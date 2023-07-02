# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 11:11:58 2018

@author: admin
"""

import numpy as np
from scipy.optimize import minimize

import nltk
import string

import gensim
# from decimal import Decimal
import re


class SMCRF2():
    def __init__(self):
        self.train_size, self.test_size = -1, -1
        self.n_param = 0
        self.num_tags1 = 3
        self.num_tags2 = 12
        self.sentences, self.labels = [], []
        self.start_symbol, self.end_symbol = '<BOS>', '<EOS>'
        self.label_index = {'S': 0, 'NKP': 1, 'KP': 2}
        self.index_label = {0: 'S', 1: 'NKP', 2: 'KP'}
        self.start_state, self.end_state = 'S', 'S'
        self.start_state_id = 0
        self.prior_feature_expectation = np.zeros(self.n_param)
        self.theta = np.zeros(self.n_param)
        self.sigma2 = 100.0  # np.inf  # no Gaussian prior
        self.feature_index, self.index_feature = {}, {}
        #        self.X_train,self.Y_train = [],[]
        self.label_index2 = {(0,): 0, (1,): 1, (2,): 2, (0, 0): 3, (1, 1): 4, (2, 2): 5, (0, 1): 6, (1, 2): 7,
                             (2, 1): 8, (2, 0): 9, (1, 0): 10, (0, 2): 11}

        self.L = 2
        self.Y_labels = []
        self.Y_train = []
        self.X_train1 = []
        self.X_train2 = []

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
        corpus_root = filename
        ### file_pattern should be adjusted across different sets
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
        for tag1 in range(self.num_tags2):
            for tag2 in range(self.num_tags2):
                feature = ('T', tag1, tag2)
                self.index_feature[feature_id] = feature
                self.feature_index[feature] = feature_id
                feature_id += 1

                ###############
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

        ### segment
        segseq = []
        alltff = []
        for i in range(len(TF)):
            sen = TF[i]
            sen1 = []
            for t in range(len(sen)):
                sent = {}
                for d in range(1, self.L + 1):
                    if t - d + 1 >= 0:
                        s = t - d + 1
                        e = d
                        tf1 = sen[t - d + 1:t + 1]
                        tf_d = sum(tf1)
                        sent[d] = tf_d
                        alltff.append(tf_d)
                sen1.append(sent)
            segseq.append(sen1)
            self.X_train1.append(sen1)

        unift = set(alltff)
        for tf in unift:
            for tag in range(self.num_tags2):
                feature = ('TF', tf, tag)
                self.index_feature[feature_id] = feature
                self.feature_index[feature] = feature_id
                feature_id += 1

        ##########
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

        segseq2 = []
        for i in range(len(OTitle)):
            sen = OTitle[i]
            sen1 = []
            for t in range(len(sen)):
                sent = {}
                for d in range(1, self.L + 1):
                    if t - d + 1 >= 0:
                        s = t - d + 1
                        e = d
                        tf1 = sen[t - d + 1:t + 1]
                        tf_d = sum(tf1)
                        sent[d] = tf_d
                sen1.append(sent)
            segseq2.append(sen1)
            self.X_train2.append(sen1)

        OT = [0, 1, 2]
        for ot in OT:
            for tag in range(self.num_tags2):
                feature = ('OT', ot, tag)
                self.index_feature[feature_id] = feature
                self.feature_index[feature] = feature_id
                feature_id += 1

        self.n_param = len(self.index_feature)
        self.theta = np.ones(self.n_param) / self.n_param
        self.get_prior_feature_expectation2()

    def feature_at(self, k, f1, f2, t, d, yt, yt1):
        #### yt current  yt1 previous
        feature = self.index_feature[k]
        tf1 = f1[t]
        ot1 = f2[t]
        tf11 = tf1[d]
        ot11 = ot1[d]

        if feature[0] == 'T':
            _, tag1, tag2 = feature
            if tag1 == yt1 and tag2 == yt:
                return 1.0
            else:
                return 0.0
        elif feature[0] == 'OT':
            _, fea, tag = feature
            if fea == ot11 and tag == yt:
                return 1.0
            else:
                return 0.0
        else:
            _, fea, tag = feature
            if fea == tf11 and tag == yt:
                return 1.0
            else:
                return 0.0

    def get_prior_feature_expectation(self):
        self.prior_feature_expectation = np.zeros(self.n_param)
        for i in range(self.train_size):
            y = self.Y_labels[i]
            f1 = self.X_train1[i]
            f2 = self.X_train2[i]
            len_i = len(y) - 2
            for k in xrange(self.n_param):
                for t in xrange(1, len_i + 1):
                    for d in range(1, self.L + 1):
                        e = t
                        s = t - d + 1
                        yc = y[s:e + 1]
                        yt = tuple(yc)
                        yp = y[s - 1]
                        ytc = self.label_index2[yt]
                        self.prior_feature_expectation[k] += self.feature_at(k, f1, f2, t, d, ytc, yp)
        return self.prior_feature_expectation

    def get_prior_feature_expectation2(self):
        self.prior_feature_expectation = np.zeros(self.n_param)
        for i in range(self.train_size):
            y = self.Y_labels[i]
            tf = self.X_train1[i]
            ot = self.X_train2[i]
            len_i = len(y) - 2
            for t in xrange(1, len_i + 1):
                for d in range(1, self.L + 1):
                    e = t
                    s = t - d + 1
                    if t - d + 1 > 0:
                        yc = y[s:e + 1]
                        yt = tuple(yc)
                        yp = y[s - 1]
                        tf1 = tf[t]
                        tf11 = tf1[d]
                        ot1 = ot[t]
                        ot11 = ot1[d]
                        ytc = self.label_index2[yt]
                        t_feature = [self.feature_index[('T', yp, ytc)]]
                        ot_feature = [self.feature_index[('OT', ot11, ytc)]]
                        tf_feature = [self.feature_index[('TF', tf11, ytc)]]
                        active_features = t_feature + ot_feature + tf_feature
                        self.prior_feature_expectation[active_features] += 1.0

        return self.prior_feature_expectation

    def log_potential_at(self, tf1, ot1, t, d, yt, yt1):
        # active feature
        global active_features
        if t - d + 1 >= 0:
            tf11 = tf1[d]
            ot11 = ot1[d]
            t_feature = [self.feature_index[('T', yt1, yt)]]
            ot_feature = [self.feature_index[('OT', ot11, yt)]]
            tf_feature = [self.feature_index[('TF', tf11, yt)]]
            active_features = t_feature + ot_feature + tf_feature
        return self.theta[active_features].sum()

    def cal_potential(self, tf1, ot1, labels_seq):
        res = 0.0
        y = labels_seq
        len_y = len(labels_seq) - 2
        for t in range(1, len_y + 1):
            tf11 = tf1[t]
            ot11 = ot1[t]
            for d in range(1, self.L + 1):
                e = t
                s = t - d + 1
                yc = y[s:e + 1]
                yt = tuple(yc)
                yt1 = y[s - 1]
                yt = self.label_index2[yt]
                res += self.log_potential_at(tf11, ot11, t, d, yt, yt1)
        return res

    def log_sum_exp(self, arr):
        max_value = np.max(arr)
        return max_value + np.log(np.sum(np.exp(arr - max_value)))

    def forward_backward(self, tf1, ot1, labels_seq, log_potential=None):
        seq_l = len(labels_seq) - 2
        y = labels_seq
        if log_potential is None:
            log_potential = np.zeros((seq_l + 1, self.L, self.num_tags2, self.num_tags2))
            for t in range(seq_l + 1):
                tf11 = tf1[t]
                ot11 = ot1[t]
                for d in range(1, self.L + 1):
                    for tag1 in range(self.num_tags2):
                        for tag2 in range(self.num_tags2):
                            log_potential[t, d - 1, tag1, tag2] = self.log_potential_at(tf11, ot11, t, d, tag1, tag2)

        log_alpha = np.zeros((seq_l + 1, self.num_tags2))
        #        st = self.label_index[self.start_state]
        log_alpha[1] = log_potential[1, 0, :, self.start_state_id]
        aa = np.zeros((seq_l + 1, self.num_tags2, self.L))

        for i in range(2, seq_l + 1):
            for d in range(1, self.L + 1):
                for j in range(self.num_tags2):
                    if i - d < 0:
                        break
                    aa[i, j, d - 1] = self.log_sum_exp(log_potential[i, d - 1, j, :] + log_alpha[i - d])
            for j in range(self.num_tags2):
                log_alpha[i, j] = self.log_sum_exp(aa[i, j, :])

        # Backward propagation
        log_beta = np.zeros((seq_l + 1, self.num_tags2))
        log_beta[seq_l] = 0.0
        bb = np.zeros((seq_l + 1, self.num_tags2, self.L))

        for t in range(seq_l - 1, 0, -1):
            for d in range(1, self.L + 1):
                for i in range(self.num_tags2):
                    if t + d - 1 > seq_l - 1:
                        break
                    bb[t, i, d - 1] = self.log_sum_exp(log_potential[t + 1, d - 1, :, i] + log_beta[t + d])
            for j in range(self.num_tags2):
                log_beta[t, j] = self.log_sum_exp(bb[t, j, :])

        log_z = self.log_sum_exp(log_potential[1, 0, :, self.start_state_id] + log_beta[1])

        '''         
        log_alpha = np.zeros((seq_l+1, self.num_tags2))

        log_alpha[1] = log_potential[1, :,self.start_state_id ]
        for i in range(2, seq_l+1):
          for j in range(self.num_tags2):
              log_alpha[i, j] = self.log_sum_exp(log_potential[i, j, :] + log_alpha[i-1])
      # Backward propagation
        log_beta = np.zeros((seq_l+1,self.num_tags2))
        log_beta[seq_l] = 0.0
        for t in range(seq_l-1, 0, -1):
            for i in range(self.num_tags2):
                log_beta[t, i] = self.log_sum_exp(log_potential[t+1, :, i] + log_beta[t+1])
        log_z = self.log_sum_exp(log_potential[1, :, self.start_state_id] + log_beta[1])
        '''
        ####### likelihood prob
        log_pot = self.cal_potential(tf1, ot1, y)
        cll = log_pot - log_z
        return log_alpha, log_beta, log_z, cll

    def PLL(self):
        res = 0.0
        for i in range(self.train_size):
            y = self.Y_labels[i]
            tf1 = self.X_train1[i]
            ot1 = self.X_train2[i]
            _, _, _, cll = self.forward_backward(tf1, ot1, y)
            res += cll
        res -= np.dot(self.theta, self.theta) / (2.0 * self.sigma2)
        return res

    def model_exp(self, i):
        y = self.Y_labels[i]
        len_i = len(y) - 2
        tf1 = self.X_train1[i]
        ot1 = self.X_train2[i]
        log_potential = np.zeros((len_i + 1, self.L, self.num_tags2, self.num_tags2))
        for t in range(len_i + 1):
            tf11 = tf1[t]
            ot11 = ot1[t]
            for d in range(1, self.L + 1):
                for tag1 in range(self.num_tags2):
                    for tag2 in range(self.num_tags2):
                        log_potential[t, d - 1, tag1, tag2] = self.log_potential_at(tf11, ot11, t, d, tag1, tag2)
        log_alpha, log_beta, log_z, cll = self.forward_backward(tf1, ot1, y, log_potential)
        # compute marginal prob
        log_p = np.zeros((len_i + 1, self.L, self.num_tags2))
        for t in range(1, len_i + 1):
            for tag1 in range(self.num_tags2):
                for tag2 in range(self.num_tags2):
                    for d in range(1, self.L + 1):
                        if t - d >= 0 and tag1 != self.start_state_id:
                            continue
                        log_p[t, d - 1, tag2] = log_alpha[t - d, tag1] + log_potential[t, d - 1, tag2, tag1] + log_beta[
                            t, tag2] - log_z

        log_p = np.exp(log_p)
        for yt1 in xrange(self.num_tags2):
            if yt1 == self.start_state_id:
                continue
            for d in range(1, self.L + 1):
                log_p[1, d - 1, yt1] = 0.0
        del log_alpha, log_beta

        model_expectation_for_example_seq = np.zeros(self.theta.shape)

        for t in range(1, len_i + 1):
            for d in range(1, self.L + 1):
                if t - d + 1 >= 0:
                    e = t
                    s = t - d + 1
                    yc = y[s:e + 1]
                    yt = tuple(yc)
                    yt1 = y[s - 1]
                    tf11 = tf1[t]
                    tf12 = tf11[d]
                    ot11 = ot1[t]
                    ot12 = ot11[d]
                    yt = self.label_index2[yt]
                    t_feature = [self.feature_index[('T', yt1, yt)]]
                    ot_feature = [self.feature_index[('OT', ot12, yt)]]
                    tf_feature = [self.feature_index[('TF', tf12, yt)]]
                    active_features = t_feature + ot_feature + tf_feature
                    model_expectation_for_example_seq[active_features] += log_p[t - d + 1, d - 1, yt]
        return model_expectation_for_example_seq

    def PLL_gra(self):
        gradient = self.prior_feature_expectation - self.theta / self.sigma2
        for i in range(self.train_size):
            gradient -= self.model_exp(i)
        return gradient

    def ncll(self, theta):
        self.theta = theta
        return -self.PLL()

    def ncll_prime(self, theta):
        self.theta = theta
        return -self.PLL_gra()

    def fit(self):
        res = minimize(self.ncll, self.theta, method='CG',
                       jac=self.ncll_prime, options={'disp': True, 'maxiter': 5})
        if res.success:
            self.theta = res.x
        else:
            print 'Failed to optimize CLL'


