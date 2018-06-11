import sys
import time
import math
import numpy as np
import tensorflow as tf
import keras.backend as K
import gensim
import jieba

from keras.backend.tensorflow_backend import set_session
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Dropout, LSTM, Embedding, Lambda, Flatten, RepeatVector, Permute, Reshape, TimeDistributed, Bidirectional, multiply, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from gensim.models import word2vec

import evaluation
import model_utils
import data_utils




def train():
    ## argument
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    predict_path = sys.argv[3]
    model_name = sys.argv[4]
    char_embed_path = sys.argv[5]
    word_embed_path = sys.argv[6]
    pos_embed_path = sys.argv[7]
    dict_path = sys.argv[8]

    train_rate = 0.9
    max_char_ctx_len = 1160
    max_word_ctx_len = 680
    
    char_ctx_len = 1160
    char_qus_len = 240
    
    word_ctx_len = 400
    word_qus_len = 40
    
    word_char_len = 5

    char_embed_size = 128
    word_embed_size = 128
    pos_embed_size = 32
    hidden_size = 64
    model_size = 64

    max_epochs = 50
    batch_size = 8

    lr = 0.001
    drop_rate = 0.5
    recur_drop_rate = 0.0
    patience = 20

    ## load data
    print("load data")
    st = time.time()
    train_raw_data = data_utils.load_json_data(train_path)
    test_raw_data = data_utils.load_json_data(test_path)
#    # load pos data
#    train_gen_pos_data = data_utils.load_json_data(train_pos_path)
#    test_gen_pos_data = data_utils.load_json_data(test_pos_path)
    # load embedding
    char_embedding = word2vec.Word2Vec.load(char_embed_path)
    word_embedding = word2vec.Word2Vec.load(word_embed_path)
    pos_embedding = word2vec.Word2Vec.load(pos_embed_path)
    et = time.time()
    print("cost time:", et - st)

    ## process data
    print("process data")
    st = time.time()
    train_data = data_utils.make_train_data(train_raw_data)  # data format: (id, context, question, answer_start, answer_end)
    test_data = data_utils.make_test_data(test_raw_data)  # data format: (id, context, question)
    train_context = [data[1] for data in train_data]
    train_question = [data[2] for data in train_data]
    train_char_answer_start = [data[3] for data in train_data]
    train_char_answer_end = [data[4] for data in train_data]
#    train_context_poss = [data['context'] for data in train_gen_pos_data['data']]
#    train_question_poss = [data['question'] for data in train_gen_pos_data['data']]
    test_id = [data[0] for data in test_data]
    test_context = [data[1] for data in test_data]
    test_question  =[data[2] for data in test_data]
#    test_context_poss = [data['context'] for data in test_gen_pos_data['data']]
#    test_question_poss = [data['question'] for data in test_gen_pos_data['data']]
    del train_data
    del test_data
    et = time.time()
    print("cost time:", et - st)

    ## tokenize data
    print("tokenize data")
    st = time.time()
    train_context_chars = data_utils.tokenize_to_chars(train_context)
    train_question_chars = data_utils.tokenize_to_chars(train_question)
    test_context_chars = data_utils.tokenize_to_chars(test_context)
    test_question_chars = data_utils.tokenize_to_chars(test_question)
    train_context_words = data_utils.tokenize_to_words(train_context, init_dict=True, dict_path=dict_path)
    train_question_words = data_utils.tokenize_to_words(train_question, init_dict=True, dict_path=dict_path)    
    test_context_words = data_utils.tokenize_to_words(test_context, init_dict=True, dict_path=dict_path)
    test_question_words = data_utils.tokenize_to_words(test_question, init_dict=True, dict_path=dict_path)    
    train_context_poss = data_utils.tokenize_to_poss(train_context, init_dict=True, dict_path=dict_path)
    train_question_poss = data_utils.tokenize_to_poss(train_question, init_dict=True, dict_path=dict_path)
    test_context_poss = data_utils.tokenize_to_poss(test_context, init_dict=True, dict_path=dict_path)
    test_question_poss = data_utils.tokenize_to_poss(test_question, init_dict=True, dict_path=dict_path)
    et = time.time()
    print("cost time:", et - st)
    
    ## build vocabulary
    print("build vocabulary")
    st = time.time()
    chars = train_context_chars + train_question_chars + test_context_chars + test_question_chars
    words = train_context_words + train_question_words + test_context_words + test_question_words
    poss = train_context_poss + train_question_poss + test_context_poss + test_question_poss
    char_vocab, rev_char_vocab = data_utils.build_vocabulary_with_embedding(chars, char_embedding)
    word_vocab, rev_word_vocab = data_utils.build_vocabulary_with_embedding(words, word_embedding)
    pos_vocab, rev_pos_vocab = data_utils.build_vocabulary_with_embedding(poss, pos_embedding)
    word_pos_vocab = data_utils.build_vocabulary_with_kv_pairs(words, poss)
    char_vocab_size = len(char_vocab)
    word_vocab_size = len(word_vocab)
    pos_vocab_size = len(pos_vocab)
    word_pos_vocab_size = len(word_pos_vocab)
    et = time.time()
    print("char vocab size:", char_vocab_size)
    print("word vocab size:", word_vocab_size)
    print("pos vocab size:", pos_vocab_size)
    print("word pos vocab size:", word_pos_vocab_size)
    print("cost time:", et - st)

    ## select data
    # select the data which sequence lengths satisfy length constraints
    print("select data")
    st = time.time()
    select_indices = data_utils.select_data_by_lengths(train_context_words, train_question_words, word_ctx_len, word_qus_len)
    train_context_chars = [train_context_chars[i] for i in select_indices]
    train_context_words = [train_context_words[i] for i in select_indices]
    train_context_poss = [train_context_poss[i] for i in select_indices]
    train_question_chars = [train_question_chars[i] for i in select_indices]
    train_question_words = [train_question_words[i] for i in select_indices]
    train_question_poss = [train_question_poss[i] for i in select_indices]
    train_char_answer_start = [train_char_answer_start[i] for i in select_indices]
    train_char_answer_end = [train_char_answer_end[i] for i in select_indices]
    et = time.time()
    print("cost time:", et - st)

    ## set answer
    # it should be done after tokenize sentences to words
    print("set answer")
    st = time.time()
    train_word_answer_start, train_word_answer_end = data_utils.set_word_answer(train_context_words, train_char_answer_start, train_char_answer_end, word_ctx_len)
    train_answer_start, train_answer_end = train_word_answer_start, train_word_answer_end
    et = time.time()
    print("cost time:", et - st)    

    ## pad data
    print("pad data")
    st = time.time()
    # clip words to chars
    # it should be done after build vocab (add PAD)
    train_context_clip_chars = data_utils.clip_words_to_chars(train_context_words, word_char_len)
    train_question_clip_chars = data_utils.clip_words_to_chars(train_question_words, word_char_len)
    test_context_clip_chars = data_utils.clip_words_to_chars(test_context_words, word_char_len)
    test_question_clip_chars = data_utils.clip_words_to_chars(test_question_words, word_char_len)
#    print("Debug: tarin_context_clip_chars[0]:")
#    print(train_context_clip_chars[0])
#    print("Debug: train_question_clip_chars[0]:")
#    print(train_question_clip_chars[0])

    # padding
    train_context_pad_chars = data_utils.pad_sequences(train_context_clip_chars, word_ctx_len * word_char_len)
    train_question_pad_chars = data_utils.pad_sequences(train_question_clip_chars, word_qus_len * word_char_len)
    train_context_pad_words = data_utils.pad_sequences(train_context_words, word_ctx_len)
    train_question_pad_words = data_utils.pad_sequences(train_question_words, word_qus_len)
    train_context_pad_poss = data_utils.pad_sequences(train_context_poss, word_ctx_len)
    train_question_pad_poss = data_utils.pad_sequences(train_question_poss, word_qus_len)
    test_context_pad_chars = data_utils.pad_sequences(test_context_clip_chars, word_ctx_len * word_char_len)
    test_question_pad_chars = data_utils.pad_sequences(test_question_clip_chars, word_qus_len * word_char_len)
    test_context_pad_words = data_utils.pad_sequences(test_context_words, word_ctx_len)
    test_question_pad_words = data_utils.pad_sequences(test_question_words, word_qus_len)
    test_context_pad_poss = data_utils.pad_sequences(test_context_poss, word_ctx_len)
    test_question_pad_poss = data_utils.pad_sequences(test_question_poss, word_qus_len)
    et = time.time()
    print("cost time:", et - st)
    ## make arrays
    print("make arrays")
    st = time.time()
    # map vocab to index
#    print("Debug: train_context_pad_words[0]:")
#    print(train_context_pad_words[0])
#    print("Debug: train_question_pad_words[0]:")
#    print(train_question_pad_words[0])
    train_context_char_indices = data_utils.map_vocabulary_index(train_context_pad_chars, char_vocab)
    train_question_char_indices = data_utils.map_vocabulary_index(train_question_pad_chars, char_vocab)
    train_context_word_indices = data_utils.map_vocabulary_index(train_context_pad_words, word_vocab)
    train_question_word_indices = data_utils.map_vocabulary_index(train_question_pad_words, word_vocab)
    train_context_pos_indices = data_utils.map_vocabulary_index(train_context_pad_poss, pos_vocab)
    train_question_pos_indices = data_utils.map_vocabulary_index(train_question_pad_poss, pos_vocab)
    test_context_char_indices = data_utils.map_vocabulary_index(test_context_pad_chars, char_vocab)
    test_question_char_indices = data_utils.map_vocabulary_index(test_question_pad_chars, char_vocab)
    test_context_word_indices = data_utils.map_vocabulary_index(test_context_pad_words, word_vocab)
    test_question_word_indices = data_utils.map_vocabulary_index(test_question_pad_words, word_vocab)
    test_context_pos_indices = data_utils.map_vocabulary_index(test_context_pad_poss, pos_vocab)
    test_question_pos_indices = data_utils.map_vocabulary_index(test_question_pad_poss, pos_vocab)
    # make one-hot label
    train_answer_start_onehot = data_utils.one_hot_encoding(train_answer_start, word_ctx_len)
    train_answer_end_onehot = data_utils.one_hot_encoding(train_answer_end, word_ctx_len)
    # to array
    # X1: context chars; X2: context words; X3: context poss;  
    # X4: question chars; X5: question words; X6: question poss; 
    # Y1: answer_start, Y2: answer_end
    train_X1 = np.array(train_context_char_indices, dtype=np.int32)
    train_X2 = np.array(train_context_word_indices, dtype=np.int32)
    train_X3 = np.array(train_context_pos_indices, dtype=np.int32)
    train_X4 = np.array(train_question_char_indices, dtype=np.int32)
    train_X5 = np.array(train_question_word_indices, dtype=np.int32)
    train_X6 = np.array(train_question_pos_indices, dtype=np.int32)
    train_Y1 = np.array(train_answer_start_onehot, dtype=np.int32)
    train_Y2 = np.array(train_answer_end_onehot, dtype=np.int32)
    train_word_ans1 = np.array(train_answer_start, dtype=np.int32)
    train_word_ans2 = np.array(train_answer_end, dtype=np.int32)
    train_ans1 = np.array(train_char_answer_start, dtype=np.int32)
    train_ans2 = np.array(train_char_answer_end, dtype=np.int32)
    test_X1 = np.array(test_context_char_indices, dtype=np.int32)
    test_X2 = np.array(test_context_word_indices, dtype=np.int32)
    test_X3 = np.array(test_context_pos_indices, dtype=np.int32)
    test_X4 = np.array(test_question_char_indices, dtype=np.int32)
    test_X5 = np.array(test_question_word_indices, dtype=np.int32)
    test_X6 = np.array(test_question_pos_indices, dtype=np.int32)
    # make embedding weight matrix
    word_embed_matrix = data_utils.make_embedding_matrix(word_embedding, word_vocab, word_embed_size)
    char_embed_matrix = data_utils.make_embedding_matrix(char_embedding, char_vocab, char_embed_size)
    pos_embed_matrix = data_utils.make_embedding_matrix(pos_embedding, pos_vocab, pos_embed_size)
    
    # delete data for releasing memory
    del train_context, train_question, test_context, test_question
    del train_context_chars, train_question_chars, test_context_chars, test_question_chars
#    del train_context_words, train_question_words, test_context_words, test_question_words
    del train_context_clip_chars, train_question_clip_chars, test_context_clip_chars, test_question_clip_chars
    del train_context_char_indices, train_question_char_indices, test_context_char_indices, test_question_char_indices
    del train_context_word_indices, train_question_word_indices, test_context_word_indices, test_question_word_indices
    del train_context_pos_indices, train_question_pos_indices, test_context_pos_indices, test_question_pos_indices
    del train_word_answer_start, train_word_answer_end, train_char_answer_start, train_char_answer_end
    del train_answer_start_onehot, train_answer_end_onehot
    et = time.time()
    print("train shape:", train_X1.shape, train_X2.shape, train_X3.shape, train_X4.shape, train_X5.shape, train_X6.shape, train_Y1.shape, train_Y2.shape)
    print("test shape:", test_X1.shape, test_X2.shape, test_X3.shape, test_X4.shape, test_X5.shape, test_X6.shape)
    print("cost time:", et - st)


    ## XXX build model
    print("build model")
    st = time.time()
    # input layers
    # X1: context chars; X2: context words; X3: context poss; 
    # X4: question chars; X5: question words; X6: question poss; 
    # Y1: answer_start; Y2: answer_end
    var_x1_input = Input(shape=(word_ctx_len*word_char_len,), dtype=np.int32)
    var_x2_input = Input(shape=(word_ctx_len,), dtype=np.int32)
    var_x3_input = Input(shape=(word_ctx_len,), dtype=np.int32)
    var_x4_input = Input(shape=(word_qus_len*word_char_len,), dtype=np.int32)
    var_x5_input = Input(shape=(word_qus_len,), dtype=np.int32)
    var_x6_input = Input(shape=(word_qus_len,), dtype=np.int32)

    # embedding layers
    var_x1_embed = Embedding(input_dim=char_vocab_size, output_dim=char_embed_size, weights=[char_embed_matrix], input_length=word_ctx_len*word_char_len, trainable=False)(var_x1_input)  # shape: (None, ctx_length * word_length, char_embed_size)
    var_x2_embed = Embedding(input_dim=word_vocab_size, output_dim=word_embed_size, weights=[word_embed_matrix], input_length=word_ctx_len, trainable=False)(var_x2_input)  # shape: (None, ctx_length, word_embed_size)
    var_x3_embed = Embedding(input_dim=pos_vocab_size, output_dim=pos_embed_size, weights=[pos_embed_matrix], input_length=word_ctx_len, trainable=False)(var_x3_input)  # shape: (None, ctx_length, pos_embed_size)
    var_x4_embed = Embedding(input_dim=char_vocab_size, output_dim=char_embed_size, weights=[char_embed_matrix], input_length=word_qus_len*word_char_len, trainable=False)(var_x4_input)  # shape: (None, qus_length * word_length, char_embed_size)
    var_x5_embed = Embedding(input_dim=word_vocab_size, output_dim=word_embed_size, weights=[word_embed_matrix], input_length=word_qus_len, trainable=False)(var_x5_input)  # shape: (None, qus_length, word_embed_size)
    var_x6_embed = Embedding(input_dim=pos_vocab_size, output_dim=pos_embed_size, weights=[pos_embed_matrix], input_length=word_qus_len, trainable=False)(var_x6_input)  # shape: (None, qus_length, pos_embed_size)
    
    var_x1_embed = Reshape([word_ctx_len, word_char_len * char_embed_size])(var_x1_embed)  # shape: (None, ctx_length, word_length * char_embed_size)
    var_x4_embed = Reshape([word_qus_len, word_char_len * char_embed_size])(var_x4_embed)  # shape: (None, qus_length, word_length * char_embed_size)
    var_char_embed_layer = Dense(units=word_embed_size)
    var_x1_embed = TimeDistributed(var_char_embed_layer, input_shape=(word_ctx_len, word_char_len * char_embed_size))(var_x1_embed)  # shape: (None, ctx_length, word_embed_size)
    var_x1_embed = Activation('relu')(var_x1_embed)
#    var_x1_embed = Dropout(rate=drop_rate)(var_x1_embed)
    var_x4_embed = TimeDistributed(var_char_embed_layer, input_shape=(word_qus_len, word_char_len * char_embed_size))(var_x4_embed)  # shape: (None, qus_length, word_embed_size)
    var_x4_embed = Activation('relu')(var_x4_embed)
#    var_x4_embed = Dropout(rate=drop_rate)(var_x4_embed)
    
    #XXX concatenate word embedding and pos embedding directly
    var_ctx_embed = concatenate([var_x1_embed, var_x2_embed, var_x3_embed], axis=2)  # shape: (None, ctx_length, word_embed_size * 2 + pos_embed_size)
    var_qus_embed = concatenate([var_x4_embed, var_x5_embed, var_x6_embed], axis=2)  # shape: (None, qus_length, word_embed_size * 2 + pos_embed_size)
    var_ctx_embed = Dropout(rate=drop_rate)(var_ctx_embed)
    var_qus_embed = Dropout(rate=drop_rate)(var_qus_embed)
    
    var_ctx_lstm = Bidirectional(LSTM(units=hidden_size, recurrent_dropout=recur_drop_rate, return_sequences=True))(var_ctx_embed)  # shape: (None, ctx_length, hidden_size * 2)
    var_qus_lstm = Bidirectional(LSTM(units=hidden_size, recurrent_dropout=recur_drop_rate, return_sequences=True))(var_qus_embed)  # shape: (None, qus_length, hidden_size * 2)
    # dropout ?
#    var_ctx_lstm = Dropout(rate=drop_rate)(var_ctx_lstm)
#    var_qus_lstm = Dropout(rate=drop_rate)(var_qus_lstm)

    # attention layers
    var_ctx_flatten = Flatten()(var_ctx_lstm)  # shape: (None, ctx_length * hidden_size * 2) 
    var_qus_flatten = Flatten()(var_qus_lstm)  # shape: (None, qus_length * hidden_size * 2)
    var_ctx_repeat = RepeatVector(word_qus_len)(var_ctx_flatten)  # shape: (None, qus_length, ctx_length * hidden_size * 2)
    var_qus_repeat = RepeatVector(word_ctx_len)(var_qus_flatten)  # shape: (None, ctx_length, qus_length * hidden_size * 2)
    var_ctx_repeat = Reshape([word_qus_len, word_ctx_len, hidden_size * 2])(var_ctx_repeat)  # shape: (None, qus_length, ctx_length, hidden_size * 2)
    var_qus_repeat = Reshape([word_ctx_len, word_qus_len, hidden_size * 2])(var_qus_repeat)  # shape: (None, ctx_length, qus_length, hidden_size * 2)
    var_ctx_repeat = Permute([2,1,3])(var_ctx_repeat)  # shape: (None, ctx_length, qus_length, hidden_size * 2)
    var_mul_repeat = multiply([var_ctx_repeat, var_qus_repeat]) # shape: (None, ctx_length, qus_length, hidden_size * 2)
    
    var_sim_repeat = concatenate([var_ctx_repeat, var_qus_repeat, var_mul_repeat], axis=3) # shape: (None, ctx_length, qus_length, hidden_size * 6)
    var_sim_sequence = Reshape([word_ctx_len * word_qus_len, hidden_size * 6])(var_sim_repeat)  # shape: (None, ctx_length * qus_length, hidden_size * 6)
    # dropout ?
#    var_sim_sequence = Dropout(rate=drop_rate)(var_sim_sequence)
    var_similarity = TimeDistributed(Dense(units=1), input_shape=(word_ctx_len*word_qus_len, hidden_size * 6))(var_sim_sequence)  # shape: (None, ctx_length * qus_length, 1)
    var_similarity = Reshape([word_ctx_len, word_qus_len])(var_similarity)  # shape: (None, ctx_length, qus_length)
    var_similarity = Activation('relu')(var_similarity)
    # dropout ?
#    var_similarity = Dropout(rate=drop_rate)(var_similarity)

    var_c2qatt_weight = TimeDistributed(Activation('softmax'), input_shape=(word_ctx_len, word_qus_len))(var_similarity)  # shape: (None, ctx_length, qus_length)
    var_c2qatt_ctx = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 1]))([var_c2qatt_weight, var_qus_lstm])  # shape: (None, ctx_length, hidden_size * 2)

    var_q2catt_weight = Lambda(lambda x: K.max(x, axis=2))(var_similarity)  # shape: (None, ctx_length)
    var_q2catt_weight = RepeatVector(hidden_size * 2)(var_q2catt_weight)  # shape: (None, hidden_size * 2, ctx_length)
    var_q2catt_weight = Permute([2,1])(var_q2catt_weight)  # shape: (None, ctx_length, hidden_size * 2)
    var_q2catt_ctx = multiply([var_q2catt_weight, var_ctx_lstm])  # shape: (None, ctx_length, hidden_size * 2)

    var_c2qctx_attmul = multiply([var_ctx_lstm, var_c2qatt_ctx])  # shape: (None, ctx_length, hidden_size * 2)
    var_q2cctx_attmul = multiply([var_ctx_lstm, var_q2catt_ctx])  # shape: (None, ctx_length, hidden_size * 2)
    var_attention = concatenate([var_ctx_lstm, var_c2qatt_ctx, var_c2qctx_attmul, var_q2cctx_attmul], axis=2)  # shape: (None, ctx_length, hidden_size * 8)
    var_attention = Activation('relu')(var_attention)
#    # dropout ?
#    var_attention = Dropout(rate=drop_rate)(var_attention)

    # model layers
    var_model1_lstm = Bidirectional(LSTM(units=model_size, recurrent_dropout=recur_drop_rate, return_sequences=True))(var_attention)  # shape: (None, ctx_length, model_size * 2)
    var_model1_att = concatenate([var_attention, var_model1_lstm], axis=2)  # shape: (None, ctx_length, hidden_size * 8 + model_size * 2)
    # dropout ?
#    var_model1_att = Dropout(rate=drop_rate)(var_model1_att)
    
    var_model2_lstm = Bidirectional(LSTM(units=model_size, recurrent_dropout=recur_drop_rate, return_sequences=True))(var_model1_lstm)  # shape: (None, ctx_length, model_size * 2)
    var_model2_att = concatenate([var_attention, var_model2_lstm], axis=2)  # shape: (None, ctx_length, hidden_size * 8 + model_size * 2)
    # dropout ?
#    var_model2_att = Dropout(rate=drop_rate)(var_model2_att)
    
    # output layers
    var_pointer1_weight = TimeDistributed(Dense(units=1), input_shape=(word_ctx_len, hidden_size*8+model_size*2))(var_model1_att)  # shape: (None, ctx_length, 1)
    var_pointer1_weight = Flatten()(var_pointer1_weight)  # shape: (None, ctx_length)
    var_pointer1 = Activation('softmax')(var_pointer1_weight)  # shape: (None, ctx_length)
    
    var_pointer2_weight = TimeDistributed(Dense(units=1), input_shape=(word_ctx_len, hidden_size*8+model_size*2))(var_model2_att)  # shape: (None, ctx_length, 1)
    var_pointer2_weight = Flatten()(var_pointer2_weight)  # shape: (None, ctx_length)
    var_pointer2 = Activation('softmax')(var_pointer2_weight)  # shape: (None, ctx_length)

    model = Model(inputs=[var_x1_input, var_x2_input, var_x3_input, var_x4_input, var_x5_input, var_x6_input],
            outputs=[var_pointer1, var_pointer2])
    
    adam = Adam(lr=lr)

#    # Set loss functions ?
#    def two_pointers_crossentropy(y_true, y_pred):
#        p1_true, p1_pred = y_true[0], y_pred[0]
#        p2_true, p2_pred = y_true[:,1], y_pred[1]
#        p1_loss = categorical_crops
    # XXX use multiple loss    
    model.compile(optimizer=adam,
            loss=['categorical_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            metrics=['accuracy'])
    et = time.time()
    print("cost time:", et - st)

    ## save vocabulary
    print("save vocabulary")
    st = time.time()
    data_utils.write_json_data('model_%s_char_vocab.json' %model_name, char_vocab)
    data_utils.write_json_data('model_%s_word_vocab.json' %model_name, word_vocab)
    data_utils.write_json_data('model_%s_pos_vocab.json' %model_name, pos_vocab)
    data_utils.write_json_data('model_%s_word_pos_vocab.json' %model_name, word_pos_vocab)
    et = time.time()
    print("cost time:", et - st)

    ## train model
    print("train model")
    st = time.time()
    history = model.fit(
            x=[train_X1, train_X2, train_X3, train_X4, train_X5, train_X6],
            y=[train_Y1, train_Y2],
            batch_size=batch_size,
            epochs=max_epochs,
            shuffle=True,
            validation_split=1 - train_rate,
            callbacks=[
                ModelCheckpoint('model_%s.h5' %model_name, monitor='val_loss', save_best_only=True),
                EarlyStopping(monitor='val_loss', patience=patience)])
    et = time.time()
    print("cost time:", et - st)

    ## save history
    print("save history")
    st = time.time()
    data_utils.write_history('model_%s_history.pkl' %model_name, history)
    et = time.time()
    print("cost time:", et - st)

    ## evaluate
    print("evaluate")
    st = time.time()
    model = load_model('model_%s.h5' %model_name, custom_objects={'tf': tf})
    # compute predict
    print("predict")
    st = time.time()
    train_Y1_hat, train_Y2_hat = model.predict([train_X1, train_X2, train_X3, train_X4, train_X5, train_X6], batch_size=batch_size)
    et = time.time()
    print("cost time:", et - st)
    train_Y1_word_pred, train_Y2_word_pred = model_utils.constraint_predict(train_Y1_hat, train_Y2_hat)
    train_Y1_pred, train_Y2_pred = data_utils.set_char_answer(train_context_words, train_Y1_word_pred, train_Y2_word_pred)
    train_Y1_pred = np.array(train_Y1_pred, dtype=np.int32)
    train_Y2_pred = np.array(train_Y2_pred, dtype=np.int32)
    # evaluate predict with setting answer (word answer)
    train_acc1, train_acc2, train_accuracy = evaluation.compute_accuracy(train_word_ans1, train_Y1_word_pred, train_word_ans2, train_Y2_word_pred)
    train_prec, train_rec, train_f1 = evaluation.compute_scores(train_word_ans1, train_Y1_word_pred, train_word_ans2, train_Y2_word_pred, word_ctx_len)
    print("word-level train accuracy:", train_acc1, train_acc2, train_accuracy)
    print("word-level train prec rec:", train_prec, train_rec)
    print("word-level train f1:", train_f1)
    # evaluate predict with real answer (char answer)
    train_acc1, train_acc2, train_accuracy = evaluation.compute_accuracy(train_ans1, train_Y1_pred, train_ans2, train_Y2_pred)
    train_prec, train_rec, train_f1 = evaluation.compute_scores(train_ans1, train_Y1_pred, train_ans2, train_Y2_pred, max_char_ctx_len)
    print("char-level train accuracy:", train_acc1, train_acc2, train_accuracy)
    print("char-level train prec rec:", train_prec, train_rec)
    print("char-level train f1:", train_f1)
    et = time.time()
    print("cost time:", et - st)

    ## test
    print("test")
    st = time.time()
    test_Y1_hat, test_Y2_hat = model.predict([test_X1, test_X2, test_X3, test_X4, test_X5, test_X6], batch_size=batch_size)
    # compute predict
    test_Y1_word_pred, test_Y2_word_pred = model_utils.constraint_predict(test_Y1_hat, test_Y2_hat)
    test_Y1_pred, test_Y2_pred = data_utils.set_char_answer(test_context_words, test_Y1_word_pred, test_Y2_word_pred)
    test_Y1_pred = np.array(test_Y1_pred, dtype=np.int32)
    test_Y2_pred = np.array(test_Y2_pred, dtype=np.int32)
    data_utils.write_predict(predict_path, test_id, test_Y1_pred, test_Y2_pred)
    et = time.time()
    print("cost time:", et - st)





if __name__ == '__main__':
    train()
