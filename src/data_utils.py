import json
import pickle
import jieba
import numpy as np
import jieba.posseg


# load json raw data or json vocab
def load_json_data(file_name):
    with open(file_name, 'r') as f:
        data_dict = json.load(f)
    return data_dict


def load_predict(file_name):
    Ids = []
    preds = []
    with open(file_name, 'r') as f:
        f.readline()
        for line in f:
            Id, pred_str = line.strip().split(',')
            pred = [int(p) for p in pred_str.split()]
            Ids.append(Id)
            preds.append(pred)
    return Ids, preds


# make train data: list of (id, context, question, answer_start, answer_end) tuples
# define: answer_start: the index of start char; answer_end: the index of end char
def make_train_data(raw_data_dict):
    train_tuple_list = []
    for data in raw_data_dict['data']:
        for paragraph in data['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                qa_id = qa['id']
                question = qa['question']
                for answer in qa['answers']:
                    text = answer['text']
                    answer_start = answer['answer_start']
                    text_len = len(text)
                    answer_end = answer_start + text_len - 1
                    # add train tuple
                    train_tuple = [None, None, None, None, None]
                    train_tuple[0] = qa_id
                    train_tuple[1] = context
                    train_tuple[2] = question
                    train_tuple[3] = answer_start
                    train_tuple[4] = answer_end
                    train_tuple_list.append(train_tuple)
    return train_tuple_list



# make test data: list of (id, context, question) tuples
def make_test_data(raw_data_dict):
    test_tuple_list = []
    for data in raw_data_dict['data']:
        for paragraph in data['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                qa_id = qa['id']
                question = qa['question']
                # add test tuple
                test_tuple = [None, None, None]
                test_tuple[0] = qa_id
                test_tuple[1] = context
                test_tuple[2] = question
                test_tuple_list.append(test_tuple)
    return test_tuple_list





#XXX use jieba to cut the sentences
def tokenize_to_words(sentences, init_dict=False, dict_path=None):
    token_list = []
    if init_dict is True:
        jieba.set_dictionary(dict_path)
    for sentence in sentences:
        seq_list = jieba.cut(sentence)
        token = [seq for seq in seq_list]
        token_list.append(token)
    return token_list



# split the sentence into single chinese character
def tokenize_to_chars(sentences):
    token_list = []
    for sentence in sentences:
        token = [char for char in sentence]
        token_list.append(token)
    return token_list



def tokenize_to_poss(sentences, init_dict=False, dict_path=None):
    token_list = []
    if init_dict is True:
        jieba.set_dictionary(dict_path)
    for sentence in sentences:
        seq_list = jieba.posseg.cut(sentence)
        token = [seq for word, seq in seq_list]
        token_list.append(token)
    return token_list



def tokenize_with_vocab():
    pass




def count_lengths(tokens_list):
    word_lengths = []
    char_lengths = []
    for tokens in tokens_list:
        char_len = []
        for token in tokens:
            char_len.append(len(token))
        word_lengths.append(len(char_len))
        char_lengths.append(char_len)
    return word_lengths, char_lengths



def build_vocabulary_with_embedding(tokens_list, embedding):
    vocab = {}
    rev_vocab = []
    index = 0
    for tokens in tokens_list:
        for token in tokens:
            if token in embedding and token not in vocab:
                vocab[token] = index
                rev_vocab.append(token)
                index += 1
    # add special vocab
    #XXX only add PAD by now
    vocab['$PAD$'] = index
    rev_vocab.append('$PAD$')
#    print("vocabulary size:", len(rev_vocab))
    return vocab, rev_vocab


# build a vocabulary which keys are in keys_list and values are in values_list
def build_vocabulary_with_kv_pairs(keys_list, values_list):
    vocab = {}
    if len(keys_list) != len(values_list):
        print("Debug: kv list length mismatch:", len(keys_list), len(values_list))
    for keys, values in zip(keys_list, values_list):
        if len(keys) != len(values):
            print("Debug: kv pairs length mismatch:", len(keys), len(values))
        for k, v in zip(keys, values):
            vocab[k] = v
    return vocab




def map_vocabulary_index(tokens_list, vocab):
    token_idxs_list = []
    for tokens in tokens_list:
        idxs = []
        for token in tokens:
#            print("Debug: token:", token)
            if token in vocab:
                idx = vocab[token]
            else:
                idx = vocab['$PAD$']
            idxs.append(idx)
        token_idxs_list.append(idxs)
    return token_idxs_list






# label the indices of words in a sentence
def make_location_index(word_idxs, vocab):
    loc_idxs_list = []
    pad_idx = vocab['$PAD$']
    for w_idxs in word_idxs:
        l_idxs = []
        for l, w_idx in enumerate(w_idxs):
            if w_idx == pad_idx:
                l_idx = 0
            else:
                l_idx = l+1
            l_idxs.append(l_idx)
        loc_idxs_list.append(l_idxs)
    return loc_idxs_list



# select data which lengths satisfy the constraints and return the indices
def select_data_by_lengths(ctx_word_seqs, qus_word_seqs, max_ctx_len, max_qus_len):
    # use both max_qus_len and max_qus_len by now
    indices = []
    for i in range(len(ctx_word_seqs)):
        ctx_seq_len = len(ctx_word_seqs[i])
        qus_seq_len = len(qus_word_seqs[i])
        if ctx_seq_len <= max_ctx_len and qus_seq_len <= max_qus_len:
            indices.append(i)
    return indices



# given char-based pointers of answers, set the word-based pointers of answers
# XXX replace the answer index larger than word number with word number - 1
def set_word_answer(words_list, char_start_list, char_end_list, max_word_num):
    word_start_list = []
    word_end_list = []
    for words, char_start, char_end in zip(words_list, char_start_list, char_end_list):
        word_start = 0
        word_end = 0
        char_count = 0
        found_start = False
        found_end = False
        for idx, word in enumerate(words):
            word_len = len(word)
            char_count += word_len
            if not found_start and char_count > char_start:
                word_start = idx
                found_start = True
            if not found_end and char_count > char_end:
                word_end = idx
                found_end = True
            if found_end:
                break
        word_start = min(max_word_num - 1, word_start)
        word_end = min(max_word_num - 1, word_end)
        word_start_list.append(word_start)
        word_end_list.append(word_end)
    return word_start_list, word_end_list



# given word-based pointers of answers, set the char-based pointers of answers
def set_char_answer(words_list, word_start_list, word_end_list):
    char_start_list = []
    char_end_list = []
    for words, word_start, word_end in zip(words_list, word_start_list, word_end_list):
        char_start = len(''.join(words[:word_start]))
        char_end = len(''.join(words[:word_end+1])) - 1
        char_start_list.append(char_start)
        char_end_list.append(char_end)
    return char_start_list, char_end_list



def make_sequence_windows(seq_list, ans_start_list, ans_end_list, window, stride, is_train_data):
    windows_list = []
    windows_indices = []
    windows_ans_start_list = []
    windows_ans_end_list = []
    windows_num = 0
    # process training data
    if is_train_data:
        for seq, ans_start, ans_end in zip(seq_list, ans_start_list, ans_end_list):
            seq_len = len(seq)
            # short sequnece as a window
            if seq_len <= window:
                windows_list.append(seq)
                windows_ans_start_list.append(ans_start)
                windows_ans_end_list.append(ans_end)
                windows_indices.append([windows_num])
                windows_num += 1
            # slide window to split long sequence
            else:
                indices = []
                for index in range(0,seq_len,stride):
                    # get window sequence and reset answer
                    wd_seq = seq[index:index + window]
                    wd_ans_start = max(0, ans_start - index)
                    wd_ans_end = max(0, ans_end - index)
                    wd_ans_start = min(wd_ans_start, window - 1)
                    wd_ans_end = min(wd_ans_end, window - 1)
                    windows_list.append(wd_seq)
                    windows_ans_start_list.append(wd_ans_start)
                    windows_ans_end_list.append(wd_ans_end)
                    indices.append(windows_num)
                    windows_num += 1
                    if index + window >= seq_len:
                        break
                windows_indices.append(indices)
    # process testing data
    else:
        for seq in seq_list:
            seq_len = len(seq)
            # short sequnece as a window
            if seq_len <= window:
                windows_list.append(seq)
                windows_indices.append([windows_num])
                windows_num += 1
            # slide window to split long sequence
            else:
                indices = []
                for index in range(0,seq_len,stride):
                    # get window sequence and reset answer
                    wd_seq = seq[index:index + window]
                    windows_list.append(wd_seq)
                    indices.append(windows_num)
                    windows_num += 1
                    if index + window >= seq_len:
                        break
                windows_indices.append(indices)
    return windows_list, windows_ans_start_list, windows_ans_end_list, windows_indices



def align_sequence_to_windows(seq_list, window_indices):
    assert len(seq_list) == len(window_indices)
    align_seq_list = []
    for seq, indices in zip(seq_list, window_indices):
        for index in indices:
            align_seq_list.append(seq)
    return align_seq_list



# use PAD to align the words of a sentence to the chars of the sentence
def align_words_to_chars(words_list):
    align_words_list = []
    for words in words_list:
        align_words = []
        for word in words:
            word_len = len(word)
            align_words.append(word)
            # add PAD
            for i in range(1, word_len):
                align_words.append('$PAD$')
        align_words_list.append(align_words)
    return align_words_list




# clip the chars of a word to fit the max length of chars
def clip_words_to_chars(words_list, max_char_len):
    clip_chars_list = []
    for words in words_list:
        clip_chars = []
        for word in words:
            word_len = len(word)
            if word_len <= max_char_len:
                for i in range(word_len):
                    clip_chars.append(word[i])
                for i in range(word_len, max_char_len):
                    clip_chars.append('$PAD$')
            else:
                for i in range(max_char_len):
                    clip_chars.append(word[i])
        clip_chars_list.append(clip_chars)
    return clip_chars_list




# add PAD to sequences in order to fit the max sequence length
def pad_sequences(sequences, max_seq_len):
    pad_sequences = []
    for sequence in sequences:
        pad_sequence = []
        seq_len = len(sequence)
        if seq_len <= max_seq_len:
            for i in range(seq_len):
                pad_sequence.append(sequence[i])
            # add PAD
            for i in range(seq_len, max_seq_len):
                pad_sequence.append('$PAD$')
        else:
            for i in range(max_seq_len):
                pad_sequence.append(sequence[i])
        pad_sequences.append(pad_sequence)
    return pad_sequences



def one_hot_encoding(index_list, index_num):
    onehot_index_list = []
    for index in index_list:
        onehot_index = [0 for i in range(index_num)]
        onehot_index[index] = 1
        onehot_index_list.append(onehot_index)
    return onehot_index_list



# make a look-up matrix from vocab to word embedding
# if one vocab not in embedding, set it to zero
def make_embedding_matrix(embedding, vocab, embed_size):
    embedding_matrix = np.zeros((len(vocab), embed_size))
    for word, idx in vocab.items():
        if word in embedding:
            embedding_vector = embedding[word]
            embedding_matrix[idx] = embedding_vector.copy()
    return embedding_matrix





def write_history(file_name, history):
    with open(file_name, 'wb') as f:
        pickle.dump(history.history, f)



def write_json_data(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f)



def write_predict(file_name, id_list, pred1, pred2):
    assert len(id_list) == pred1.shape[0]
    assert len(id_list) == pred2.shape[0]
    with open(file_name, 'w') as f:
        print("id,answer", file=f)
        for i in range(len(id_list)):
            answer_string = ' '.join(str(j) for j in range(pred1[i], pred2[i]+1))
            print("%s,%s" %(id_list[i], answer_string), file=f)
