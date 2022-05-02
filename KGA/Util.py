import multiprocessing
import time
import gc

import numpy as np
from tensorflow.python.keras import backend as K
import tensorflow as tf

import Levenshtein

def l1(a, b):
    return K.sum(K.abs(a - b), axis=-1, keepdims=True)

def l2(a, b):
    return K.sum(K.square(a - b), axis=-1, keepdims=True)


def eval_alignment_by_sim_mat(embed1, embed2, top_k, nums_threads, csls=0, accurate=False,output = True):
    t = time.time()
    sim_mat = sim_handler(embed1, embed2, csls, nums_threads)
    ref_num = sim_mat.shape[0]
    t_num = [0 for k in top_k]
    t_mean = 0
    t_mrr = 0
    t_prec_set = set()
    tasks = div_list(np.array(range(ref_num)), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_rank_by_sim_mat, (task, sim_mat[task, :], top_k, accurate)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)
        t_prec_set |= prec_set
    assert len(t_prec_set) == ref_num
    acc = np.array(t_num) / ref_num * 100
    for i in range(len(acc)):
        acc[i] = round(acc[i], 2)
    t_mean /= ref_num
    t_mrr /= ref_num
    if output:
        if accurate:
            print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc, t_mean,
                                                                                                       t_mrr,
                                                                                                       time.time() - t))
        else:
            print("hits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    hits1 = acc[0]
    del sim_mat
    gc.collect()
    return t_prec_set, hits1

def eval_alignment_by_sim_mat_withName(embed1, embed2,eName1,eName2, top_k, nums_threads, csls=0, accurate=False,output = True):
    t = time.time()
    sim_mat = sim_handler(embed1, embed2, csls, nums_threads)
    sim_mat_name = simNmae_handler(eName1, eName2)
    #
    sim_mat_com = sim_mat + sim_mat_name
    sim_mat = sim_mat_com / np.linalg.norm(sim_mat_com, axis=-1, keepdims=True)

    # sim_mat_name = simNmae_handler(eName1, eName2)
    # sim_mat = sim_mat_name / np.linalg.norm(sim_mat_name, axis=-1, keepdims=True)

    ref_num = sim_mat.shape[0]
    t_num = [0 for k in top_k]
    t_mean = 0
    t_mrr = 0
    t_prec_set = set()

    # tasks = div_list(np.array(range(ref_num)), nums_threads)
    # pool = multiprocessing.Pool(processes=len(tasks))
    # reses = list()
    # for task in tasks:
    #     reses.append(pool.apply_async(cal_rank_by_sim_mat, (task, sim_mat[task, :], top_k, accurate)))
    # pool.close()
    # pool.join()
    #
    # for res in reses:
    #     mean, mrr, num, prec_set = res.get()
    #     t_mean += mean
    #     t_mrr += mrr
    #     t_num += np.array(num)
    #     t_prec_set |= prec_set

    reses = cal_rank_by_sim_mat(np.array(range(ref_num)), sim_mat, top_k, accurate)
    mean, mrr, num, prec_set = reses[0], reses[1], reses[2], reses[3]
    t_mean += mean
    t_mrr += mrr
    t_num += np.array(num)
    t_prec_set |= prec_set

    assert len(t_prec_set) == ref_num
    acc = np.array(t_num) / ref_num * 100
    for i in range(len(acc)):
        acc[i] = round(acc[i], 2)
    t_mean /= ref_num
    t_mrr /= ref_num
    if output:
        if accurate:
            print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc, t_mean,
                                                                                                       t_mrr,
                                                                                                       time.time() - t))
        else:
            print("hits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    hits1 = acc[0]
    del sim_mat
    gc.collect()
    return t_prec_set, hits1


def sim_handler(embed1, embed2, k, nums_threads):
    sim_mat = np.matmul(embed1, embed2.T)
    if k <= 0:
        print("k = 0")
        return sim_mat
    csls1 = CSLS_sim(sim_mat, k, nums_threads)
    csls2 = CSLS_sim(sim_mat.T, k, nums_threads)
    # for i in range(sim_mat.shape[0]):
    #     for j in range(sim_mat.shape[1]):
    #         sim_mat[i][j] = 2 * sim_mat[i][j] - csls1[i] - csls2[j]
    # return sim_mat
    csls_sim_mat = 2 * sim_mat.T - csls1
    csls_sim_mat = csls_sim_mat.T - csls2
    del sim_mat
    gc.collect()
    return csls_sim_mat

def CSLS_sim(sim_mat1, k, nums_threads):
    # sorted_mat = -np.partition(-sim_mat1, k, axis=1) # -np.sort(-sim_mat1)
    # nearest_k = sorted_mat[:, 0:k]
    # sim_values = np.mean(nearest_k, axis=1)

    tasks = div_list(np.array(range(sim_mat1.shape[0])), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_csls_sim, (sim_mat1[task, :], k)))
    pool.close()
    pool.join()
    sim_values = None
    for res in reses:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat1.shape[0]
    return sim_values

def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]
    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return

def cal_csls_sim(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    sim_values = np.mean(nearest_k, axis=1)
    return sim_values

def cal_rank_by_sim_mat(task, sim, top_k, accurate):
    mean = 0
    mrr = 0
    num = [0 for k in top_k]
    prec_set = set()
    for i in range(len(task)):
        ref = task[i]
        if accurate:
            rank = (-sim[i, :]).argsort()
        else:
            rank = np.argpartition(-sim[i, :], np.array(top_k) - 1)
        prec_set.add((ref, rank[0]))
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, mrr, num, prec_set


def simNmae_handler(eName1, eName2):
    length = len(eName1)
    sim_mat = np.zeros((length, length))
    for i in range(length):
        name1 = eName1[i]
        for j in range(length):
            name2 = eName2[j]
            l = Levenshtein.distance(name1, name2)
            sim_mat[i][j] = 1/(l + 1)

    sim_mat = sim_mat / np.linalg.norm(sim_mat, axis=-1, keepdims=True)
    return sim_mat


def tf_unique_2d(x):
    x_shape = tf.shape(x)  # (3,2)
    x1 = tf.tile(x, [1, x_shape[0]])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
    x2 = tf.tile(x, [x_shape[0], 1])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]

    x1_2 = tf.reshape(x1, [x_shape[0] * x_shape[0], x_shape[1]])
    x2_2 = tf.reshape(x2, [x_shape[0] * x_shape[0], x_shape[1]])
    cond = tf.reduce_all(tf.equal(x1_2, x2_2), axis=1)
    cond = tf.reshape(cond, [x_shape[0], x_shape[0]])  # reshaping cond to match x1_2 & x2_2
    cond_shape = tf.shape(cond)
    cond_cast = tf.cast(cond, tf.int32)  # convertin condition boolean to int
    cond_zeros = tf.zeros(cond_shape, tf.int32)  # replicating condition tensor into all 0's

    # CREATING RANGE TENSOR
    r = tf.range(x_shape[0])
    r = tf.add(tf.tile(r, [x_shape[0]]), 1)
    r = tf.reshape(r, [x_shape[0], x_shape[0]])

    # converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so when we take min it wont get selected & in end we will only take values <max(indx).
    f1 = tf.multiply(tf.ones(cond_shape, tf.int32), x_shape[0] + 1)
    f2 = tf.ones(cond_shape, tf.int32)
    cond_cast2 = tf.where(tf.equal(cond_cast, cond_zeros), f1, f2)  # if false make it max_index+1 else keep it 1

    # multiply range with new int boolean mask
    r_cond_mul = tf.multiply(r, cond_cast2)
    r_cond_mul2 = tf.reduce_min(r_cond_mul, axis=1)
    r_cond_mul3, unique_idx = tf.unique(r_cond_mul2)
    r_cond_mul4 = tf.subtract(r_cond_mul3, 1)

    # get actual values from unique indexes
    op = tf.gather(x, r_cond_mul4)

    return op, unique_idx
