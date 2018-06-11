import numpy as np




def compute_accuracy(true1, pred1, true2, pred2):
    assert true1.shape == pred1.shape
    assert true2.shape == pred2.shape
    acc1 = np.mean(1*(true1 == pred1))
    acc2 = np.mean(1*(true2 == pred2))
    acc = (acc1 + acc2) / 2.
    return acc1, acc2, acc



# comupte the precision, recall, f1-score
def compute_scores(true1, pred1, true2, pred2, max_length):
    assert true1.shape == pred1.shape
    assert true2.shape == pred2.shape
    all_set = set(range(max_length))    
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    num = true1.shape[0]
    for i in range(num):
        true_set = set(range(true1[i], true2[i]+1))
        pred_set = set(range(pred1[i], pred2[i]+1))
        tp = true_set & pred_set
        fp = pred_set & (all_set - true_set)
        fn = true_set & (all_set - pred_set)
        prec = len(tp) / (len(tp) + len(fp))
        rec = len(tp) / (len(tp) + len(fn))
        if prec != 0.0 or rec != 0.0:
            f1 = 2*prec*rec / (prec + rec)
        else:
            f1 = 0.0
        precision += prec
        recall += rec
        f1_score += f1
    precision /= num
    recall /= num
    f1_score /= num
    return precision, recall, f1_score
