import random

import itertools
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.seq2seq import BasicDecoder

from basic.read_data_nqa import DataSet
from my.tensorflow import get_initializer
from my.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d
from my.tensorflow.rnn import bidirectional_dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, AttentionCell
from tensorflow.python.layers import core as layers_core
from my.tensorflow import flatten

def get_multi_gpu_models(config):
    models = []
    for gpu_idx in range(config.num_gpus):
        with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
            if gpu_idx > 0:
                tf.get_variable_scope().reuse_variables()
            model = Model(config, scope, rep=gpu_idx == 0)
            models.append(model)
    return models


class Model(object):
    def __init__(self, config, scope, rep=True):
        self.scope = scope
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Define forward inputs here
        N, M, JX, JQ, VW, VC, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size

        # Batch major format

        # Context
        self.x = tf.placeholder('int32', [N, None, None], name='x') # going to be [N, M, JX] i.e. [batch size, max sentences, max words]
        self.cx = tf.placeholder('int32', [N, None, None, W], name='cx')
        self.x_mask = tf.placeholder('bool', [N, None, None], name='x_mask')

        # Query
        self.q = tf.placeholder('int32', [N, None], name='q') # going to be [N, JQ] i.e. [batch size, max words]
        self.cq = tf.placeholder('int32', [N, None, W], name='cq')
        self.q_mask = tf.placeholder('bool', [N, None], name='q_mask')

        # Outputs
        self.y = tf.placeholder('bool', [N, None, None], name='y')
        self.y2 = tf.placeholder('bool', [N, None, None], name='y2')

        # Misc
        self.wy = tf.placeholder('bool', [N, None, None], name='wy')
        self.is_train = tf.placeholder('bool', [], name='is_train')
        self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name='new_emb_mat')
        self.na = tf.placeholder('bool', [N], name='na')

        # New placeholders for our case
        self.decoder_inputs = tf.placeholder('int32', [N, None], name='decoder_inputs') # [batch_size, max words]
        self.target_sequence_length = tf.placeholder(shape=(N,), dtype=tf.int32, name='target_sequence_length') # batch_size
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets') # [batch_size, max_decoder_time]
        self.target_weights = tf.placeholder(shape=(None, None), dtype=tf.float32, name='target_weights') # [batch_size, max_decoder_time] # Same as loss mask

        # Define misc
        self.tensor_dict = {} # seems to be for keeping track of intermediate values during forward pass -- not super important maybe?

        # Forward outputs / loss inputs
        self.logits = None
        self.yp = None
        self.var_list = None
        self.na_prob = None
        self.decoder_logits_train = None

        # Loss outputs
        self.loss = None

        self._build_forward()
        self._build_loss()
        # TODO: Delete this?
        self.var_ema = None
        if rep:
            self._build_var_ema()
        if config.mode == 'train':
            self._build_ema()

        self.summary = tf.summary.merge_all()
        self.summary = tf.summary.merge(tf.get_collection("summaries", scope=self.scope))

    def _build_forward(self):
        config = self.config
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
            config.max_word_size
        JX = tf.shape(self.x)[2]  # words
        JQ = tf.shape(self.q)[1] # words
        M = tf.shape(self.x)[1]
        dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size

        print ("dhruv is here",N, self.x.get_shape(), JX, self.q.get_shape(), VW, VC, d, W,dc, dw, dco)
        with tf.variable_scope("emb"):
            if config.use_char_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')

                with tf.variable_scope("char"):
                    Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)  # [N, M, JX, W, dc]
                    Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)  # [N, JQ, W, dc]
                    Acx = tf.reshape(Acx, [-1, JX, W, dc])
                    Acq = tf.reshape(Acq, [-1, JQ, W, dc])

                    filter_sizes = list(map(int, config.out_channel_dims.split(',')))
                    heights = list(map(int, config.filter_heights.split(',')))
                    assert sum(filter_sizes) == dco, (filter_sizes, dco)
                    with tf.variable_scope("conv"):
                        xx = multi_conv1d(Acx, filter_sizes, heights, "VALID",  self.is_train, config.keep_prob, scope="xx")
                        if config.share_cnn_weights:
                            tf.get_variable_scope().reuse_variables()
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="xx")
                        else:
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="qq")
                        xx = tf.reshape(xx, [-1, M, JX, dco])
                        qq = tf.reshape(qq, [-1, JQ, dco])

            if config.use_word_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    if config.mode == 'train':
                        word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, dw], initializer=get_initializer(config.emb_mat))
                    else:
                        word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, dw], dtype='float')
                    if config.use_glove_for_unk:
                        word_emb_mat = tf.concat(axis=0, values=[word_emb_mat, self.new_emb_mat])

                with tf.name_scope("word"):
                    Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, M, JX, d] i.e. [batch size, max sentences, max words, embedding size]
                    Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [N, JQ, d] i.e. [batch size, max words, embedding size]
                    self.tensor_dict['x'] = Ax
                    self.tensor_dict['q'] = Aq
                if config.use_char_emb:
                    xx = tf.concat(axis=3, values=[xx, Ax])  # [N, M, JX, di]
                    qq = tf.concat(axis=2, values=[qq, Aq])  # [N, JQ, di]
                else:
                    xx = Ax
                    qq = Aq

        # highway network
        if config.highway:
            with tf.variable_scope("highway"):
                xx = highway_network(xx, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                qq = highway_network(qq, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)

        self.tensor_dict['xx'] = xx
        self.tensor_dict['qq'] = qq

        cell_fw = BasicLSTMCell(d, state_is_tuple=True)
        cell_bw = BasicLSTMCell(d, state_is_tuple=True)
        d_cell_fw = SwitchableDropoutWrapper(cell_fw, self.is_train, input_keep_prob=config.input_keep_prob)
        d_cell_bw = SwitchableDropoutWrapper(cell_bw, self.is_train, input_keep_prob=config.input_keep_prob)
        cell2_fw = BasicLSTMCell(d, state_is_tuple=True)
        cell2_bw = BasicLSTMCell(d, state_is_tuple=True)
        d_cell2_fw = SwitchableDropoutWrapper(cell2_fw, self.is_train, input_keep_prob=config.input_keep_prob)
        d_cell2_bw = SwitchableDropoutWrapper(cell2_bw, self.is_train, input_keep_prob=config.input_keep_prob)
        cell3_fw = BasicLSTMCell(d, state_is_tuple=True)
        cell3_bw = BasicLSTMCell(d, state_is_tuple=True)
        d_cell3_fw = SwitchableDropoutWrapper(cell3_fw, self.is_train, input_keep_prob=config.input_keep_prob)
        d_cell3_bw = SwitchableDropoutWrapper(cell3_bw, self.is_train, input_keep_prob=config.input_keep_prob)
        cell4_fw = BasicLSTMCell(d, state_is_tuple=True)
        cell4_bw = BasicLSTMCell(d, state_is_tuple=True)
        d_cell4_fw = SwitchableDropoutWrapper(cell4_fw, self.is_train, input_keep_prob=config.input_keep_prob)
        d_cell4_bw = SwitchableDropoutWrapper(cell4_bw, self.is_train, input_keep_prob=config.input_keep_prob)
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]

        with tf.variable_scope("prepro"):
            (fw_u, bw_u), ((_, fw_u_f), (_, bw_u_f)) = bidirectional_dynamic_rnn(d_cell_fw, d_cell_bw, qq, q_len, dtype='float', scope='u1')  # [N, J, d], [N, d]
            u = tf.concat(axis=2, values=[fw_u, bw_u])
            if config.share_lstm_weights:
                tf.get_variable_scope().reuse_variables()
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, xx, x_len, dtype='float', scope='u1')  # [N, M, JX, 2d]
                h = tf.concat(axis=3, values=[fw_h, bw_h])  # [N, M, JX, 2d]
            else:
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, xx, x_len, dtype='float', scope='h1')  # [N, M, JX, 2d]
                h = tf.concat(axis=3, values=[fw_h, bw_h])  # [N, M, JX, 2d]
            self.tensor_dict['u'] = u
            self.tensor_dict['h'] = h

        with tf.variable_scope("main"):
            if config.dynamic_att: # not true
                p0 = h
                u = tf.reshape(tf.tile(tf.expand_dims(u, 1), [1, M, 1, 1]), [N * M, JQ, 2 * d])
                q_mask = tf.reshape(tf.tile(tf.expand_dims(self.q_mask, 1), [1, M, 1]), [N * M, JQ])
                first_cell_fw = AttentionCell(cell2_fw, u, mask=q_mask, mapper='sim',
                                              input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
                first_cell_bw = AttentionCell(cell2_bw, u, mask=q_mask, mapper='sim',
                                              input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
                second_cell_fw = AttentionCell(cell3_fw, u, mask=q_mask, mapper='sim',
                                            input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
                second_cell_bw = AttentionCell(cell3_bw, u, mask=q_mask, mapper='sim',
                                               input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
            else:
                p0 = attention_layer(config, self.is_train, h, u, h_mask=self.x_mask, u_mask=self.q_mask, scope="p0", tensor_dict=self.tensor_dict) # p0 seems to be G in paper
                first_cell_fw = d_cell2_fw
                second_cell_fw = d_cell3_fw
                first_cell_bw = d_cell2_bw
                second_cell_bw = d_cell3_bw

            (fw_g0, bw_g0), _ = bidirectional_dynamic_rnn(first_cell_fw, first_cell_bw, p0, x_len, dtype='float', scope='g0')  # [N, M, JX, 2d]
            g0 = tf.concat(axis=3, values=[fw_g0, bw_g0])

            # (fw_g1, _), (my_fw_final_state, _) = bidirectional_dynamic_rnn(second_cell_fw, second_cell_bw, g0, x_len, dtype='float', scope='g1')  # [N, M, JX, 2d]
            # g1 = tf.concat(axis=3, values=[fw_g1, bw_g1]) # g1 seems to be M in paper
            # FIXME: Concatenate my_fw_final_state and my_bw_final_state

            flat_g0 = flatten(g0, 2)  # [-1, J, d]
            flat_x_len = None if x_len is None else tf.cast(flatten(x_len, 0), 'int64')

            _, my_fw_final_state = tf.nn.dynamic_rnn(
                cell = second_cell_fw,
                inputs = flat_g0,
                sequence_length = flat_x_len,
                initial_state=None,
                dtype='float',
                parallel_iterations=None,
                swap_memory=False,
                time_major=False,
                scope='my_fw_final_state'
            ) # FIXME: Replace with bi-rnn

            # print(tf.shape(my_fw_final_state)) # Tensor("model_0/main/Shape_14:0", shape=(3,), dtype=int32, device=/device:GPU:0)
            # print(my_fw_final_state) # LSTMStateTuple(c=<tf.Tensor 'model_0/main/g1/fw/fw/while/Exit_2:0' shape=(?, 100) dtype=float32>, h=<tf.Tensor 'model_0/main/g1/fw/fw/while/Exit_3:0' shape=(?, 100) dtype=float32>)

            # For now go with my_fw_final_state only

            # Decoder embedding

            # Embedding decoder/matrix

            tgt_vocab_size = VW # hparam # FIXME: Obtain embeddings differently?
            print(tgt_vocab_size)
            tgt_embedding_size = dw # hparam
            print(tgt_embedding_size)

            embedding_decoder = word_emb_mat
            # tf.Variable(
            #     tf.constant(0.0, shape=[tgt_vocab_size, tgt_embedding_size]), trainable=False, name="embedding_decoder"
            # )

            # Look up embedding
            decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, self.decoder_inputs) # [batch_size, max words, embedding_size]
            print(decoder_emb_inp)

            def decode(helper, scope, reuse=None, maximum_iterations=None):
                with tf.variable_scope(scope, reuse=reuse):
                    decoder_cell = BasicLSTMCell(d, state_is_tuple=True) # hparam
                    projection_layer = layers_core.Dense(
                        tgt_vocab_size, use_bias=False) # hparam

                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        decoder_cell, helper, my_fw_final_state,
                        output_layer=projection_layer) # decoder

                    final_outputs, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder, output_time_major=False, impute_finished=True) # dynamic decoding # TODO: check tutorial for more

                    return final_outputs

            # Decoder
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp, self.target_sequence_length, time_major=False) # helper

            final_outputs= decode(helper=training_helper, scope="HAHA", reuse=None)

            decoder_logits_train = final_outputs.rnn_output

            print(decoder_logits_train)
            self.decoder_logits_train = decoder_logits_train

    def _build_loss(self):
        config = self.config
        JX = tf.shape(self.x)[2]
        M = tf.shape(self.x)[1]
        JQ = tf.shape(self.q)[1]
        N = config.batch_size

        # Loss
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_targets, logits=self.decoder_logits_train)
        train_loss = (tf.reduce_sum(crossent * self.target_weights) / tf.cast(N, dtype=tf.float32))

        tf.add_to_collection('losses', train_loss)

        self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss') # loss for the entire batch ?
        tf.summary.scalar(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss) # FIXME: Needed?

    def _build_ema(self):
        self.ema = tf.train.ExponentialMovingAverage(self.config.decay)
        ema = self.ema
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + tf.get_collection("ema/vector", scope=self.scope)
        ema_op = ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def _build_var_ema(self):
        self.var_ema = tf.train.ExponentialMovingAverage(self.config.var_decay)
        ema = self.var_ema
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def get_feed_dict(self, batch, is_train, supervised=True):
        assert isinstance(batch, DataSet)
        config = self.config
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
        feed_dict = {}

        if config.len_opt:
            """
            Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
            First test without len_opt and make sure no OOM, and use len_opt
            """
            if sum(len(sent) for para in batch.data['x'] for sent in para) == 0:
                new_JX = 1
            else:
                new_JX = max(len(sent) for para in batch.data['x'] for sent in para)
            JX = min(JX, new_JX)

            if sum(len(ques) for ques in batch.data['q']) == 0:
                new_JQ = 1
            else:
                new_JQ = max(len(ques) for ques in batch.data['q'])
            JQ = min(JQ, new_JQ)

        if config.cpu_opt:
            if sum(len(para) for para in batch.data['x']) == 0:
                new_M = 1
            else:
                new_M = max(len(para) for para in batch.data['x'])
            M = min(M, new_M)

        x = np.zeros([N, M, JX], dtype='int32')
        cx = np.zeros([N, M, JX, W], dtype='int32')
        x_mask = np.zeros([N, M, JX], dtype='bool')
        q = np.zeros([N, JQ], dtype='int32')
        cq = np.zeros([N, JQ, W], dtype='int32')
        q_mask = np.zeros([N, JQ], dtype='bool')
        answer = np.zeros([N, JX], dtype='int32')
        answer_eos = np.zeros([N, JX], dtype='int32')

        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.cx] = cx
        feed_dict[self.q] = q
        feed_dict[self.cq] = cq
        feed_dict[self.q_mask] = q_mask
        feed_dict[self.is_train] = is_train
        feed_dict[self.decoder_inputs]=answer
        feed_dict[self.decoder_targets]=answer_eos
        feed_dict[self.target_sequence_length]=batch.data['ans_len']

        if config.use_glove_for_unk:
            feed_dict[self.new_emb_mat] = batch.shared['new_emb_mat']

        X = batch.data['x']
        CX = batch.data['cx']

        if supervised:
            y = np.zeros([N, M, JX], dtype='bool')
            y2 = np.zeros([N, M, JX], dtype='bool')
            wy = np.zeros([N, M, JX], dtype='bool')
            na = np.zeros([N], dtype='bool')
            feed_dict[self.y] = y
            feed_dict[self.y2] = y2
            feed_dict[self.wy] = wy
            feed_dict[self.na] = na

            for i, (xi, cxi, yi, nai) in enumerate(zip(X, CX, batch.data['y'], batch.data['na'])):
                if nai:
                    na[i] = nai
                    continue
                start_idx, stop_idx = random.choice(yi)
                j, k = start_idx
                j2, k2 = stop_idx
                if config.single:
                    X[i] = [xi[j]]
                    CX[i] = [cxi[j]]
                    j, j2 = 0, 0
                if config.squash:
                    offset = sum(map(len, xi[:j]))
                    j, k = 0, k + offset
                    offset = sum(map(len, xi[:j2]))
                    j2, k2 = 0, k2 + offset
                y[i, j, k] = True
                y2[i, j2, k2-1] = True
                if j == j2:
                    wy[i, j, k:k2] = True
                else:
                    wy[i, j, k:len(batch.data['x'][i][j])] = True
                    wy[i, j2, :k2] = True

        def _get_word(word):
            d = batch.shared['word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:
                    return d[each]
            if config.use_glove_for_unk:
                d2 = batch.shared['new_word2idx']
                for each in (word, word.lower(), word.capitalize(), word.upper()):
                    if each in d2:
                        return d2[each] + len(d)
            return 1

        def _get_char(char):
            d = batch.shared['char2idx']
            if char in d:
                return d[char]
            return 1

        for i, xi in enumerate(X):
            if self.config.squash:
                xi = [list(itertools.chain(*xi))]
            for j, xij in enumerate(xi):
                if j == config.max_num_sents:
                    break
                for k, xijk in enumerate(xij):
                    if k == config.max_sent_size:
                        break
                    each = _get_word(xijk)
                    assert isinstance(each, int), each
                    x[i, j, k] = each
                    x_mask[i, j, k] = True

        for i, cxi in enumerate(CX):
            if self.config.squash:
                cxi = [list(itertools.chain(*cxi))]
            for j, cxij in enumerate(cxi):
                if j == config.max_num_sents:
                    break
                for k, cxijk in enumerate(cxij):
                    if k == config.max_sent_size:
                        break
                    for l, cxijkl in enumerate(cxijk):
                        if l == config.max_word_size:
                            break
                        cx[i, j, k, l] = _get_char(cxijkl)

        for i, qi in enumerate(batch.data['q']):
            for j, qij in enumerate(qi):
                q[i, j] = _get_word(qij)
                q_mask[i, j] = True

        for i, cqi in enumerate(batch.data['cq']):
            for j, cqij in enumerate(cqi):
                for k, cqijk in enumerate(cqij):
                    cq[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break

        for i, ansi in enumerate(batch.data['answerss']):
            for j, ansij in enumerate(ansi[0]):#answer with sos
                if j == config.max_sent_size:
                    break
                    each = _get_word(ansij)
                    assert isinstance(each, int), each
                    answer[i,j] = each
            for k, ansik in enumerate(ansi[1]):#answer with eos
                if j == config.max_sent_size:
                    break
                    each = _get_word(ansik)
                    assert isinstance(each, int), each
                    answer_eos[i,k] = each
        

        if supervised:
            assert np.sum(~(x_mask | ~wy)) == 0

        return feed_dict


def bi_attention(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "bi_attention"):
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        h_aug = tf.tile(tf.expand_dims(h, 3), [1, 1, 1, JQ, 1])
        u_aug = tf.tile(tf.expand_dims(tf.expand_dims(u, 1), 1), [1, M, JX, 1, 1])
        if h_mask is None:
            hu_mask = None
        else:
            h_mask_aug = tf.tile(tf.expand_dims(h_mask, 3), [1, 1, 1, JQ])
            u_mask_aug = tf.tile(tf.expand_dims(tf.expand_dims(u_mask, 1), 1), [1, M, JX, 1])
            hu_mask = h_mask_aug & u_mask_aug

        u_logits = get_logits([h_aug, u_aug], None, True, wd=config.wd, mask=hu_mask,
                              is_train=is_train, func=config.logit_func, scope='u_logits')  # [N, M, JX, JQ]
        u_a = softsel(u_aug, u_logits)  # [N, M, JX, d]
        h_a = softsel(h, tf.reduce_max(u_logits, 3))  # [N, M, d]
        h_a = tf.tile(tf.expand_dims(h_a, 2), [1, 1, JX, 1])

        if tensor_dict is not None:
            a_u = tf.nn.softmax(u_logits)  # [N, M, JX, JQ]
            a_h = tf.nn.softmax(tf.reduce_max(u_logits, 3))
            tensor_dict['a_u'] = a_u
            tensor_dict['a_h'] = a_h
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
            for var in variables:
                tensor_dict[var.name] = var

        return u_a, h_a


def attention_layer(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "attention_layer"):
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        if config.q2c_att or config.c2q_att:
            u_a, h_a = bi_attention(config, is_train, h, u, h_mask=h_mask, u_mask=u_mask, tensor_dict=tensor_dict)
        if not config.c2q_att:
            u_a = tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_mean(u, 1), 1), 1), [1, M, JX, 1])
        if config.q2c_att:
            p0 = tf.concat(axis=3, values=[h, u_a, h * u_a, h * h_a])
        else:
            p0 = tf.concat(axis=3, values=[h, u_a, h * u_a])
        return p0
