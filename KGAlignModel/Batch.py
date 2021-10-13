import multiprocessing as mp
import time

import math
import numpy as np
import random

from modules.load.kg import KG
from modules.load.kgs import KGs
from modules.utils.util import task_divide

from itertools import combinations


def generate_relation_triple_batch_queue(triple_list1, triple_list2, triple_set1, triple_set2,
                                         entity_list1, entity_list2, batch_size,
                                         steps, out_queue, neighbor1, neighbor2, kg_neg_triple_multi):
    for step in steps:
        pos_batch, neg_batch = generate_relation_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                                              entity_list1, entity_list2, batch_size,
                                                              step, neighbor1, neighbor2, kg_neg_triple_multi)
        out_queue.put((pos_batch, neg_batch))
    exit(0)

def generate_relation_triple_batch(kg, args):
    batchList = list()
    triples_num = kg.relation_triples_num
    triple_steps = int(math.ceil(triples_num / args.batch_size))
    steps_tasks = task_divide(list(range(triple_steps)), args.batch_threads_num)
    triple_list = kg.relation_triples_list
    triple_set = kg.relation_triples_set
    entity_list = kg.entities_list
    batch_size = args.batch_size
    for steps_task in steps_tasks:
        for step in steps_task:
            batch_size = int(len(triple_list) / (len(triple_list)) * batch_size)
            pos_batch = generate_pos_triples(triple_list, batch_size, step)
            neg_batch = generate_neg_triples_fast(pos_batch, triple_set, entity_list, args.kg_neg_triple_multi, neighbor=None)
            batchList.append((pos_batch, neg_batch))
    return batchList

def generate_entityPair_batch(kg:KG, args):
    batchList = list()
    entityNum = kg.entities_num
    batch_size = args.kg_batch_size
    for i in range(0, entityNum, batch_size):
        e_batch = kg.entities_list[i:i+batch_size]
        batchList.append(e_batch)
    return batchList


def generate_pos_triples(triples, batch_size, step, is_fixed_size=False):
    start = step * batch_size
    end = start + batch_size
    if end > len(triples):
        end = len(triples)
    pos_batch = triples[start: end]
    # pos_batch = random.sample(triples, batch_size)
    if is_fixed_size and len(pos_batch) < batch_size:
        pos_batch += triples[:batch_size - len(pos_batch)]
    return pos_batch


def generate_neg_triples_fast(pos_batch, all_triples_set, entities_list, kg_neg_triple_multi, neighbor=None, max_try=10):
    if neighbor is None:
        neighbor = dict()
    neg_batch = list()
    for head, relation, tail in pos_batch:
        neg_triples = list()
        nums_to_sample = kg_neg_triple_multi
        head_candidates = neighbor.get(head, entities_list)
        tail_candidates = neighbor.get(tail, entities_list)
        for i in range(max_try):
            corrupt_head_prob = np.random.binomial(1, 0.5)
            if corrupt_head_prob:
                neg_heads = random.sample(head_candidates, nums_to_sample)
                i_neg_triples = {(h2, relation, tail) for h2 in neg_heads}
            else:
                neg_tails = random.sample(tail_candidates, nums_to_sample)
                i_neg_triples = {(head, relation, t2) for t2 in neg_tails}
            if i == max_try - 1:
                neg_triples += list(i_neg_triples)
                break
            else:
                i_neg_triples = list(i_neg_triples - all_triples_set)
                neg_triples += i_neg_triples
            if len(neg_triples) == kg_neg_triple_multi:
                break
            else:
                nums_to_sample = kg_neg_triple_multi - len(neg_triples)
        assert len(neg_triples) == kg_neg_triple_multi
        neg_batch.extend(neg_triples)
    assert len(neg_batch) == kg_neg_triple_multi * len(pos_batch)
    return neg_batch


def generate_align_batch(kgs, args, neighbors1=None, neighbors2=None):
    batch_size = args.batch_size
    if batch_size > len(kgs.train_entities1):
        batch_size = len(kgs.train_entities1)
    index = np.random.choice(len(kgs.train_entities1), batch_size)
    train_links = np.array(kgs.train_links)
    pos_links = train_links[index,]
    neg_links = list()
    if neighbors1 is None:
        neg_ent1 = list()
        neg_ent2 = list()
        for i in range(args.align_neg_triple_multi):
            neg_ent1.extend(random.sample(kgs.train_entities1, batch_size))
            neg_ent2.extend(random.sample(kgs.train_entities2, batch_size))
        neg_links.extend([(neg_ent1[i], neg_ent2[i]) for i in range(len(neg_ent1))])
    else:
        for i in range(args.batch_size):
            e1 = pos_links[i, 0]
            candidates = random.sample(neighbors1.get(e1), args.align_neg_triple_multi)
            neg_links.extend([(e1, candidate) for candidate in candidates])
            e2 = pos_links[i, 1]
            candidates = random.sample(neighbors2.get(e2), args.align_neg_triple_multi)
            neg_links.extend([(candidate, e2) for candidate in candidates])
    neg_links = set(neg_links) - set(kgs.train_links)
    neg_links = np.array(list(neg_links))
    return pos_links, neg_links

def generate_ent_batch(kgs: KGs, args):
    batch_size = args.kg_batch_size
    kg1 = kgs.kg1
    kg2 = kgs.kg2
    if batch_size > kg1.entities_num:
        batch_size = kg1.entities_num
    if batch_size > kg2.entities_num:
        batch_size = kg2.entities_num

    index1 = np.random.choice(kg1.entities_num, batch_size)
    index2 = np.random.choice(kg2.entities_num, batch_size)

    twoIndex1 = list(combinations(list(index1), 2))
    twoIndex2 = list(combinations(list(index2), 2))
    return twoIndex1, twoIndex2