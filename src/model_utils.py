import random
import numpy as np


# add constraint to prediction
# constraint 1: pred1 should be smaller than pred2
# constraint 2: pred length should be smaller than max_length
def constraint_predict(pred_prob1, pred_prob2, max_length=None):
    assert pred_prob1.shape == pred_prob2.shape
    pred1 = np.argmax(pred_prob1, axis=1)
    pred2 = []
    num = pred_prob1.shape[0]
    for i in range(num):
        # find the index with highest probability among indices bigger than pred1
        idx = np.argmax(pred_prob2[i, pred1[i]:])
        # limit the length of predict to min(idx, max_length)
        if max_length != None:
            if idx <= max_length:
                pred2.append(idx+pred1[i])
            else:
                pred2.append(max_length+pred1[i])
        else:
            pred2.append(idx+pred1[i])
    pred2 = np.array(pred2, dtype=np.int32)
    return pred1, pred2




# merge and constraint predictions of different windows to the same question
# constraint 1: pred1 should be smaller than pred2
def merge_constraint_predict(window_indices, pred_prob1, pred_prob2, window_size):
    merge_pred1 = []
    merge_pred2 = []
    
    # constraint predict: found the window prediction
    window_pred1 = np.argmax(pred_prob1, axis=1)
    window_pred2 = []
    window_num = pred_prob1.shape[0]    
    for i in range(window_num):
        # find the index with highest probability among indices bigger than pred1
        idx = np.argmax(pred_prob2[i, window_pred1[i]:])
        window_pred2.append(idx+window_pred1[i])

    for indices in window_indices:
        sel_wd_pred1 = None
        sel_wd_pred2 = None
        max_prob = -np.inf
        for index in indices:
            # ignore the prediction in the boundary
            if window_pred1[index] >= window_size - 2 or window_pred2[index] <= 1:
                continue
            # select the window prediction with the max average probability of predict1 and predict2
            prob = (pred_prob1[index, window_pred1[index]] + pred_prob2[index, window_pred2[index]]) / 2.
            if max_prob < prob:
                sel_wd_pred1 = window_pred1[index]
                sel_wd_pred2 = window_pred2[index]
                max_prob = prob
        # random select if no predict satisfy the condition
        if sel_wd_pred1 == None or sel_wd_pred2 == None:
            index = random.choice(indices)
            sel_wd_pred1 = window_pred1[index]
            sel_wd_pred2 = window_pred2[index]
        merge_pred1.append(sel_wd_pred1)
        merge_pred2.append(sel_wd_pred2)
    merge_pred1 = np.array(merge_pred1, dtype=np.int32)
    merge_pred2 = np.array(merge_pred2, dtype=np.int32)
    return merge_pred1, merge_pred2

