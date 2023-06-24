import os
import pickle

import scipy.sparse as sp
import numpy as np
import pandas as pd

from moduleUtil import Bert_encoder


def load_data(dataPath: str, train_ratio=0.3, dev_ratio=0.7, test_ratio=0):
    id_ent1, ent_id1, id_entName1 = loadDict(os.path.join(dataPath, 'entN_ids_1'))
    id_ent2, ent_id2, id_entName2 = loadDict(os.path.join(dataPath, 'entN_ids_2'))
    id_ent, ent_id, id_entName = {**id_ent1, **id_ent2}, {**ent_id1, **ent_id2}, {**id_entName1, **id_entName2}
    ent_data = dict()
    ent_data["id_ent"], ent_data["ent_id"], ent_data["id_entName"] = id_ent, ent_id, id_entName

    id_attr1, attr_id1, id_attrName1 = loadDict(os.path.join(dataPath, 'attr_ids_1'))
    id_attr2, attr_id2, id_attrName2 = loadDict(os.path.join(dataPath, 'attr_ids_2'))
    id_attr, attr_id, id_attrName = {**id_attr1, **id_attr2}, {**attr_id1, **attr_id2}, {**id_attrName1, **id_attrName2}
    attr_data = dict()
    attr_data["id_attr"], attr_data["attr_id"], attr_data["id_attrName"] = id_attr, attr_id, id_attrName

    id_rel1, rel_id1, id_relName1 = loadDict(os.path.join(dataPath, 'rel_ids_1'))
    id_rel2, rel_id2, id_relName2 = loadDict(os.path.join(dataPath, 'rel_ids_2'))
    id_rel, rel_id, id_relName = {**id_rel1, **id_rel2}, {**rel_id1, **rel_id2}, {**id_relName1, **id_relName2}
    # 添加自循环关系
    num_rel = len(id_rel)
    id_rel[num_rel] = "self"
    rel_id["self"] = num_rel
    id_relName[len(id_rel)] = "self"
    rel_data = dict()
    rel_data["id_rel"], rel_data["rel_id"], rel_data["id_relName"] = id_rel, rel_id, id_relName

    ent_attr_triples1 = read_attribute_triples(os.path.join(dataPath, 'ent_attr_1'))
    ent_attr_triples2 = read_attribute_triples(os.path.join(dataPath, 'ent_attr_2'))
    ent_attr_triples = ent_attr_triples1.union(ent_attr_triples2)

    ent_ent_triples1 = read_relation_triples(os.path.join(dataPath, 'triples_1'))
    ent_ent_triples2 = read_relation_triples(os.path.join(dataPath, 'triples_2'))
    ent_ent_triples = ent_ent_triples1.union(ent_ent_triples2)

    attr_matrix = get_attr_matrix(ent_attr_triples)

    isAtriples, NerTriples = spilt_triples(ent_ent_triples, rel_id)

    all_matrix = get_matrix(ent_ent_triples, id_ent, rel_id)
    pc_matrix = get_matrix(isAtriples, id_ent, rel_id)
    n_matrix = get_matrix(NerTriples, id_ent, rel_id)

    data = dict()
    data["ent_data"] = ent_data
    data["attr_data"] = attr_data
    data["rel_data"] = rel_data
    data["attr_matrix"] = attr_matrix

    data["all_matrix"] = all_matrix
    data["pc_matrix"] = pc_matrix
    data["n_matrix"] = n_matrix

    alignment_pair = read_links(os.path.join(dataPath, 'ref_ent_ids'), id_ent)
    np.random.shuffle(alignment_pair)
    length = len(alignment_pair)
    splitl = [0, int(length * train_ratio), int(length * train_ratio + length * dev_ratio),
              int(length * train_ratio + length * dev_ratio + length * test_ratio)]
    train_pair, dev_pair, test_pair = alignment_pair[0:splitl[1]], alignment_pair[splitl[1]: splitl[2]], alignment_pair[splitl[3]:]

    saved_AV_path = os.path.join(dataPath, 'predata', "AVH0.pkl")
    AVH = None
    if os.path.exists(saved_AV_path):
        print('load saved AVH data from', saved_AV_path)
        AVH = pickle.load(open(saved_AV_path, 'rb'))
    else:
        AVH = buildAVH(id_entName, id_attrName, ent_attr_triples, 0, isContainAttr=True)
        print('save AVH data to', saved_AV_path)
        pickle.dump(AVH, open(saved_AV_path, 'wb'),protocol = 4)

    return train_pair, dev_pair, test_pair, data, AVH


def loadDict(file_name):
    print("read attribute triples:", file_name)
    ide, eid, idName = dict(), dict(), dict()
    for line in open(file_name, 'r', encoding='utf-8'):
        line = line.replace("\n", "")
        id, e = line.split("\t")
        eName = e.split("/")[-1]
        ide[int(id)] = e
        eid[e] = int(id)
        idName[int(id)] = eName

    return ide, eid, idName


def read_attribute_triples(file_path):
    print("read attribute triples:", file_path)
    triples = set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip().strip('\n').split('\t')
        if len(params) < 3:
            continue
        head = params[0]
        attr = params[1]
        value = params[2].split("^^")[0]
        if len(params) > 3:
            for p in params[3:]:
                value = value + ' ' + p.strip()
        value = value.strip().rstrip('.').strip()
        triples.add((head, attr, value))

    return triples


def read_relation_triples(file_path):
    print("read relation triples:", file_path)
    triples = set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 3
        h = params[0].strip()
        r = params[1].strip()
        t = params[2].strip()
        triples.add((h, r, t))

    return triples


def get_attr_matrix(ent_attr_triples):
    ent_attr_index = []
    ent_attr_normValue = []
    entity_attr_value = []

    num_attr_of_ent = dict()
    for triple in ent_attr_triples:
        entityid = int(triple[0])
        attrid = int(triple[1])
        value = triple[2].split("^^")[0].replace("\"", "")

        ent_attr_index.append([entityid, attrid])
        ent_attr_normValue.append(1)
        entity_attr_value.append(value)

        num_attr = num_attr_of_ent.get(entityid, 0)
        num_attr_of_ent[entityid] = num_attr + 1

    for i in range(len(ent_attr_normValue)):
        entityid = ent_attr_index[i][0]
        ent_attr_normValue[i] /= num_attr_of_ent[entityid]

    data = {"ent_id": np.array(ent_attr_index)[:,0], "attr_id": np.array(ent_attr_index)[:,1], "normValue": ent_attr_normValue, "literal": entity_attr_value}
    attr_matrix = pd.DataFrame(data=data).sort_values(by="ent_id").values

    return attr_matrix


def get_ent_matrix(ent_ent_triples, id_ent):
    ent_ent = dict()

    num_entities = len(id_ent)

    ent_ent_matrix_bi = sp.lil_matrix((num_entities, num_entities))
    ent_ent_matrix_in = sp.lil_matrix((num_entities, num_entities))
    ent_ent_matrix_out = sp.lil_matrix((num_entities, num_entities))

    ent_ent_index_addself_bi = []
    ent_ent_index_addself_in = []
    ent_ent_index_addself_out = []
    ent_ent_normValue_addself_bi = []
    ent_ent_normValue_addself_in = []
    ent_ent_normValue_addself_out = []

    ent_ent_index_bi = []
    ent_ent_index_in = []
    ent_ent_index_out = []

    num_ner_of_ent_bi = dict()
    num_ner_of_ent_in = dict()
    num_ner_of_ent_out = dict()
    for triple in ent_ent_triples:
        h_id = int(triple[0])
        t_id = int(triple[2])

        ent_ent_index_out.append([h_id, t_id])
        ent_ent_index_in.append([t_id, h_id])

        ent_ent_matrix_out[h_id, t_id] = 1
        ent_ent_matrix_in[t_id, h_id] = 1

        ent_ent_matrix_bi[h_id, t_id] = 1
        ent_ent_matrix_bi[t_id, h_id] = 1

        ent_ent_index_addself_in.append([h_id, t_id])
        ent_ent_normValue_addself_in.append([1])
        num_ner = num_ner_of_ent_in.get(h_id, 0)
        num_ner_of_ent_in[h_id] = num_ner + 1

        ent_ent_index_addself_out.append([t_id, h_id])
        ent_ent_normValue_addself_out.append([1])
        num_ner = num_ner_of_ent_out.get(t_id, 0)
        num_ner_of_ent_out[t_id] = num_ner + 1

        ent_ent_index_addself_bi.append([h_id, t_id])
        ent_ent_normValue_addself_bi.append([1])
        num_ner = num_ner_of_ent_bi.get(h_id, 0)
        num_ner_of_ent_bi[h_id] = num_ner + 1
        ent_ent_index_addself_bi.append([t_id, h_id])
        ent_ent_normValue_addself_bi.append([1])
        num_ner = num_ner_of_ent_bi.get(t_id, 0)
        num_ner_of_ent_bi[t_id] = num_ner + 1

    for i in range(num_entities):
        ent_ent_matrix_out[i, i] = 1
        ent_ent_matrix_in[i, i] = 1
        ent_ent_matrix_bi[i, i] = 1

        ent_ent_index_addself_in.append([i, i])
        ent_ent_normValue_addself_in.append([1])
        num_ner = num_ner_of_ent_in.get(i, 0)
        num_ner_of_ent_in[i] = num_ner + 1

        ent_ent_index_addself_out.append([i, i])
        ent_ent_normValue_addself_out.append([1])
        num_ner = num_ner_of_ent_out.get(i, 0)
        num_ner_of_ent_out[i] = num_ner + 1

        ent_ent_index_addself_bi.append([i, i])
        ent_ent_normValue_addself_bi.append([1])
        num_ner = num_ner_of_ent_bi.get(i, 0)
        num_ner_of_ent_bi[i] = num_ner + 1

    for i in range(len(ent_ent_normValue_addself_bi)):
        entityid = ent_ent_index_addself_bi[i][0]
        ent_ent_normValue_addself_bi[i][0] /= num_ner_of_ent_bi[entityid]

    for i in range(len(ent_ent_normValue_addself_in)):
        entityid = ent_ent_index_addself_in[i][0]
        ent_ent_normValue_addself_in[i][0] /= num_ner_of_ent_in[entityid]

    for i in range(len(ent_ent_normValue_addself_out)):
        entityid = ent_ent_index_addself_out[i][0]
        ent_ent_normValue_addself_out[i][0] /= num_ner_of_ent_out[entityid]

    ent_ent_addself_bi = np.array(sorted(np.concatenate((np.array(ent_ent_index_addself_bi), np.array(ent_ent_normValue_addself_bi)),
                   axis=1), key=lambda x:(x[0], x[1])))
    ent_ent_addself_in = np.array(sorted(np.concatenate((np.array(ent_ent_index_addself_in), np.array(ent_ent_normValue_addself_in)),
                   axis=1), key=lambda x:(x[0], x[1])))
    ent_ent_addself_out = np.array(sorted(np.concatenate((np.array(ent_ent_index_addself_out), np.array(ent_ent_normValue_addself_out)),
                   axis=1), key=lambda x:(x[0], x[1])))


    ent_ent_matrix_in = normalize_adj(ent_ent_matrix_in)
    ent_ent_matrix_out = normalize_adj(ent_ent_matrix_out)
    ent_ent_matrix_bi = normalize_adj(ent_ent_matrix_bi)
    ent_ent_index_bi = ent_ent_index_in + ent_ent_index_out

    ent_ent["ent_ent_matrix_bi"] = ent_ent_matrix_bi
    ent_ent["ent_ent_matrix_in"] = ent_ent_matrix_in
    ent_ent["ent_ent_matrix_out"] = ent_ent_matrix_out

    ent_ent["ent_ent_index_bi"] = np.array(sorted(ent_ent_index_bi, key=lambda x: x[0]))
    ent_ent["ent_ent_index_in"] = np.array(sorted(ent_ent_index_in, key=lambda x: x[0]))
    ent_ent["ent_ent_index_out"] = np.array(sorted(ent_ent_index_out, key=lambda x: x[0]))

    ent_ent["ent_ent_addself_bi"] = ent_ent_addself_bi
    ent_ent["ent_ent_addself_in"] = ent_ent_addself_in
    ent_ent["ent_ent_addself_out"] = ent_ent_addself_out

    return ent_ent


def get_rel_matrix(ent_ent_triples, id_ent, id_rel):
    ent_rel = dict()

    num_entities = len(id_ent)

    ent_rel_index_addself_bi = []
    ent_rel_index_addself_in = []
    ent_rel_index_addself_out = []
    ent_rel_normValue_addself_bi = []
    ent_rel_normValue_addself_in = []
    ent_rel_normValue_addself_out = []

    ent_rel_index_bi = []
    ent_rel_index_in = []
    ent_rel_index_out = []

    num_rel_of_ent_bi = dict()
    num_rel_of_ent_in = dict()
    num_rel_of_ent_out = dict()
    for triple in ent_ent_triples:
        h_id = int(triple[0])
        r_id = int(triple[1])
        t_id = int(triple[2])

        ent_rel_index_out.append([h_id, r_id])
        ent_rel_index_in.append([t_id, r_id])

        ent_rel_index_addself_out.append([h_id, r_id])
        ent_rel_normValue_addself_out.append([1])
        num_rel = num_rel_of_ent_out.get(h_id, 0)
        num_rel_of_ent_out[h_id] = num_rel + 1

        ent_rel_index_addself_in.append([t_id, r_id])
        ent_rel_normValue_addself_in.append([1])
        num_rel = num_rel_of_ent_in.get(t_id, 0)
        num_rel_of_ent_in[t_id] = num_rel + 1

        ent_rel_index_addself_bi.append([h_id, r_id])
        ent_rel_normValue_addself_bi.append([1])
        num_rel = num_rel_of_ent_bi.get(h_id, 0)
        num_rel_of_ent_bi[h_id] = num_rel + 1
        ent_rel_index_addself_bi.append([t_id, r_id])
        ent_rel_normValue_addself_bi.append([1])
        num_rel = num_rel_of_ent_bi.get(t_id, 0)
        num_rel_of_ent_bi[t_id] = num_rel + 1

    self_rel_id = id_rel.get("self")
    for i in range(num_entities):
        ent_rel_index_addself_in.append([i, self_rel_id])
        ent_rel_normValue_addself_in.append([1])
        num_rel = num_rel_of_ent_in.get(i, 0)
        num_rel_of_ent_in[i] = num_rel + 1

        ent_rel_index_addself_out.append([i, self_rel_id])
        ent_rel_normValue_addself_out.append([1])
        num_rel = num_rel_of_ent_out.get(i, 0)
        num_rel_of_ent_out[i] = num_rel + 1

        ent_rel_index_addself_bi.append([i, self_rel_id])
        ent_rel_normValue_addself_bi.append([1])
        num_rel = num_rel_of_ent_bi.get(i, 0)
        num_rel_of_ent_bi[i] = num_rel + 1

    for i in range(len(ent_rel_normValue_addself_bi)):
        entityid = ent_rel_index_addself_bi[i][0]
        ent_rel_normValue_addself_bi[i][0] /= num_rel_of_ent_bi[entityid]

    for i in range(len(ent_rel_normValue_addself_in)):
        entityid = ent_rel_index_addself_in[i][0]
        ent_rel_normValue_addself_bi[i][0] /= num_rel_of_ent_in[entityid]

    for i in range(len(ent_rel_normValue_addself_out)):
        entityid = ent_rel_index_addself_out[i][0]
        ent_rel_normValue_addself_out[i][0] /= num_rel_of_ent_out[entityid]

    ent_rel_addself_bi = np.concatenate((np.array(ent_rel_index_addself_bi), np.array(ent_rel_normValue_addself_bi)),
                   axis=1)
    ent_rel_addself_in = np.concatenate((np.array(ent_rel_index_addself_in), np.array(ent_rel_normValue_addself_in)),
                   axis=1)
    ent_rel_addself_out = np.concatenate((np.array(ent_rel_index_addself_out), np.array(ent_rel_normValue_addself_out)),
                   axis=1)

    ent_rel_index_bi = ent_rel_index_in + ent_rel_index_out

    ent_rel["ent_rel_index_bi"] = np.array(sorted(ent_rel_index_bi, key=lambda x: x[0]))
    ent_rel["ent_rel_index_in"] = np.array(sorted(ent_rel_index_in, key=lambda x: x[0]))
    ent_rel["ent_rel_index_out"] = np.array(sorted(ent_rel_index_out, key=lambda x: x[0]))

    ent_rel["ent_rel_addself_bi"] = ent_rel_addself_bi
    ent_rel["ent_rel_addself_in"] = ent_rel_addself_in
    ent_rel["ent_rel_addself_out"] = ent_rel_addself_out

    return ent_rel

def get_matrix(ent_ent_triples, id_ent, rel_id):
    matrix = dict()

    num_entities = len(id_ent)

    ent_ent_matrix_bi = sp.lil_matrix((num_entities, num_entities))
    ent_ent_matrix_in = sp.lil_matrix((num_entities, num_entities))
    ent_ent_matrix_out = sp.lil_matrix((num_entities, num_entities))

    ent_ent_index_addself_bi = []
    ent_ent_index_addself_in = []
    ent_ent_index_addself_out = []
    ent_ent_normValue_addself_bi = []
    ent_ent_normValue_addself_in = []
    ent_ent_normValue_addself_out = []

    ent_ent_index_bi = []
    ent_ent_index_in = []
    ent_ent_index_out = []

    ent_rel_index_addself_bi = []
    ent_rel_index_addself_in = []
    ent_rel_index_addself_out = []
    ent_rel_normValue_addself_bi = []
    ent_rel_normValue_addself_in = []
    ent_rel_normValue_addself_out = []

    ent_rel_index_bi = []
    ent_rel_index_in = []
    ent_rel_index_out = []

    num_ner_of_ent_bi = dict()
    num_ner_of_ent_in = dict()
    num_ner_of_ent_out = dict()
    for triple in ent_ent_triples:
        h_id = int(triple[0])
        t_id = int(triple[2])

        ent_ent_index_out.append([h_id, t_id])
        ent_ent_index_in.append([t_id, h_id])

        ent_ent_matrix_out[h_id, t_id] = 1
        ent_ent_matrix_in[t_id, h_id] = 1

        ent_ent_matrix_bi[h_id, t_id] = 1
        ent_ent_matrix_bi[t_id, h_id] = 1

        ent_ent_index_addself_in.append([h_id, t_id])
        ent_ent_normValue_addself_in.append([1])
        num_ner = num_ner_of_ent_in.get(h_id, 0)
        num_ner_of_ent_in[h_id] = num_ner + 1

        ent_ent_index_addself_out.append([t_id, h_id])
        ent_ent_normValue_addself_out.append([1])
        num_ner = num_ner_of_ent_out.get(t_id, 0)
        num_ner_of_ent_out[t_id] = num_ner + 1

        ent_ent_index_addself_bi.append([h_id, t_id])
        ent_ent_normValue_addself_bi.append([1])
        num_ner = num_ner_of_ent_bi.get(h_id, 0)
        num_ner_of_ent_bi[h_id] = num_ner + 1
        ent_ent_index_addself_bi.append([t_id, h_id])
        ent_ent_normValue_addself_bi.append([1])
        num_ner = num_ner_of_ent_bi.get(t_id, 0)
        num_ner_of_ent_bi[t_id] = num_ner + 1

    for i in range(num_entities):
        ent_ent_matrix_out[i, i] = 1
        ent_ent_matrix_in[i, i] = 1
        ent_ent_matrix_bi[i, i] = 1

        ent_ent_index_addself_in.append([i, i])
        ent_ent_normValue_addself_in.append([1])
        num_ner = num_ner_of_ent_in.get(i, 0)
        num_ner_of_ent_in[i] = num_ner + 1

        ent_ent_index_addself_out.append([i, i])
        ent_ent_normValue_addself_out.append([1])
        num_ner = num_ner_of_ent_out.get(i, 0)
        num_ner_of_ent_out[i] = num_ner + 1

        ent_ent_index_addself_bi.append([i, i])
        ent_ent_normValue_addself_bi.append([1])
        num_ner = num_ner_of_ent_bi.get(i, 0)
        num_ner_of_ent_bi[i] = num_ner + 1

    for i in range(len(ent_ent_normValue_addself_bi)):
        entityid = ent_ent_index_addself_bi[i][0]
        ent_ent_normValue_addself_bi[i][0] /= num_ner_of_ent_bi[entityid]

    for i in range(len(ent_ent_normValue_addself_in)):
        entityid = ent_ent_index_addself_in[i][0]
        ent_ent_normValue_addself_in[i][0] /= num_ner_of_ent_in[entityid]

    for i in range(len(ent_ent_normValue_addself_out)):
        entityid = ent_ent_index_addself_out[i][0]
        ent_ent_normValue_addself_out[i][0] /= num_ner_of_ent_out[entityid]


    num_rel_of_ent_bi = dict()
    num_rel_of_ent_in = dict()
    num_rel_of_ent_out = dict()
    for triple in ent_ent_triples:
        h_id = int(triple[0])
        r_id = int(triple[1])
        t_id = int(triple[2])

        ent_rel_index_out.append([h_id, r_id])
        ent_rel_index_in.append([t_id, r_id+len(rel_id)])

        ent_rel_index_addself_out.append([h_id, r_id])
        ent_rel_normValue_addself_out.append([1])
        num_rel = num_rel_of_ent_out.get(h_id, 0)
        num_rel_of_ent_out[h_id] = num_rel + 1

        ent_rel_index_addself_in.append([t_id, r_id])
        ent_rel_normValue_addself_in.append([1])
        num_rel = num_rel_of_ent_in.get(t_id, 0)
        num_rel_of_ent_in[t_id] = num_rel + 1

        ent_rel_index_addself_bi.append([h_id, r_id])
        ent_rel_normValue_addself_bi.append([1])
        num_rel = num_rel_of_ent_bi.get(h_id, 0)
        num_rel_of_ent_bi[h_id] = num_rel + 1
        ent_rel_index_addself_bi.append([t_id, r_id+len(rel_id)])
        ent_rel_normValue_addself_bi.append([1])
        num_rel = num_rel_of_ent_bi.get(t_id, 0)
        num_rel_of_ent_bi[t_id] = num_rel + 1

    self_rel_id = rel_id.get("self")
    for i in range(num_entities):
        ent_rel_index_addself_in.append([i, self_rel_id])
        ent_rel_normValue_addself_in.append([1])
        num_rel = num_rel_of_ent_in.get(i, 0)
        num_rel_of_ent_in[i] = num_rel + 1

        ent_rel_index_addself_out.append([i, self_rel_id])
        ent_rel_normValue_addself_out.append([1])
        num_rel = num_rel_of_ent_out.get(i, 0)
        num_rel_of_ent_out[i] = num_rel + 1

        ent_rel_index_addself_bi.append([i, self_rel_id])
        ent_rel_normValue_addself_bi.append([1])
        num_rel = num_rel_of_ent_bi.get(i, 0)
        num_rel_of_ent_bi[i] = num_rel + 1

    for i in range(len(ent_rel_normValue_addself_bi)):
        entityid = ent_rel_index_addself_bi[i][0]
        ent_rel_normValue_addself_bi[i][0] /= num_rel_of_ent_bi[entityid]

    for i in range(len(ent_rel_normValue_addself_in)):
        entityid = ent_rel_index_addself_in[i][0]
        ent_rel_normValue_addself_bi[i][0] /= num_rel_of_ent_in[entityid]

    for i in range(len(ent_rel_normValue_addself_out)):
        entityid = ent_rel_index_addself_out[i][0]
        ent_rel_normValue_addself_out[i][0] /= num_rel_of_ent_out[entityid]

    addself_bi = np.array(sorted(np.concatenate((np.array(ent_ent_index_addself_bi), np.array(ent_ent_normValue_addself_bi),
                                 np.array(ent_rel_index_addself_bi), np.array(ent_rel_normValue_addself_bi)), axis=1),
                                 key=lambda x:(x[0], x[1], x[4])))
    addself_in = np.array(sorted(np.concatenate((np.array(ent_ent_index_addself_in), np.array(ent_ent_normValue_addself_in),
                                 np.array(ent_rel_index_addself_in), np.array(ent_rel_normValue_addself_in)), axis=1),
                                 key=lambda x:(x[0], x[1], x[4])))
    addself_out = np.array(sorted(np.concatenate((np.array(ent_ent_index_addself_out), np.array(ent_ent_normValue_addself_out),
                                 np.array(ent_rel_index_addself_out), np.array(ent_rel_normValue_addself_out)), axis=1),
                                 key=lambda x:(x[0], x[1], x[4])))


    ent_ent_index_bi = ent_ent_index_in + ent_ent_index_out
    ent_rel_index_bi = ent_rel_index_in + ent_rel_index_out
    index_bi = np.array(sorted(np.concatenate((np.array(ent_ent_index_bi), np.array(ent_rel_index_bi)), axis=1),
                                 key=lambda x:(x[0], x[1], x[3])))


    ent_ent_matrix_in = normalize_adj(ent_ent_matrix_in)
    ent_ent_matrix_out = normalize_adj(ent_ent_matrix_out)
    ent_ent_matrix_bi = normalize_adj(ent_ent_matrix_bi)


    matrix["ent_ent_matrix_bi"] = ent_ent_matrix_bi
    matrix["ent_ent_matrix_in"] = ent_ent_matrix_in
    matrix["ent_ent_matrix_out"] = ent_ent_matrix_out

    matrix["index_bi"] = index_bi

    matrix["addself_bi"] = addself_bi
    matrix["addself_in"] = addself_in
    matrix["addself_out"] = addself_out

    return matrix


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T


def spilt_triples(ent_ent_triples: set, rel_id: dict):
    typeIDSet = set()
    typeIDSet.add(rel_id.get("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"))
    typeIDSet.add(rel_id.get("http://www.w3.org/2000/01/rdf-schema#subClassOf"))
    typeIDSet.add(rel_id.get("http://dbpedia.org/ontology/type"))
    typeIDSet.add(rel_id.get("http://www.wikidata.org/entity/P249"))
    typeIDSet.add(rel_id.get("http://www.wikidata.org/entity/P31"))
    typeIDSet.add(rel_id.get("http://dbkwik.webdatacommons.org/memory-alpha.wikia.com/property/type"))
    typeIDSet.add(rel_id.get("http://dbkwik.webdatacommons.org/memory-beta.wikia.com/property/type"))
    typeIDSet.add(rel_id.get("label"))
    typeIDSet.remove(None)

    isAtriples = set()
    NerTriples = set()
    for triple in ent_ent_triples:
        rel = int(triple[1])
        if rel in typeIDSet:
            isAtriples.add(triple)
        else:
            NerTriples.add(triple)

    return isAtriples, NerTriples


def read_links(file_path, id_ent):
    print("read links:", file_path)
    links = list()
    refs = list()
    reft = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        e1 = int(params[0].strip())
        e2 = int(params[1].strip())
        refs.append(e1)
        reft.append(e2)
        if e1 in id_ent.keys() and e2 in id_ent.keys():
            links.append((e1, e2))
    assert len(refs) == len(reft)
    return links


def buildAVH(id_entName, id_attrName, ent_attr_triples, maxAttrNum, isContainAttr = True):
    liter_encoder = Bert_encoder()

    entityNum = len(id_entName)
    EH = np.zeros((entityNum, 768), dtype=np.float32)
    for id in id_entName.keys():
        print("id_entName{}".format(id))
        entName = id_entName.get(id)
        entEmbedding = liter_encoder.encodeText(str(entName))
        EH[id] = entEmbedding

    attrNum = len(id_attrName)
    AH = np.zeros((attrNum, 768), dtype=np.float32)
    for id in id_attrName.keys():
        print("id_attrName{}".format(id))
        attrName = id_attrName.get(id)
        attrEmbedding = liter_encoder.encodeText(str(attrName))
        AH[id] = attrEmbedding

    if isContainAttr:
        AVH = np.zeros((entityNum, maxAttrNum + 1, 768 * 2), dtype=np.float32)
        AVH_mask = np.zeros((entityNum, maxAttrNum + 1), dtype=np.int64)
        for id in id_entName.keys():
            AVH[id][0] = np.concatenate(([EH[id]], [EH[id]]), axis=1)
            AVH_mask[id][0] = 1

        i = 0
        for ent_attr_triple in ent_attr_triples:
            print(i)
            en_id = int(ent_attr_triple[0])
            attr_id = int(ent_attr_triple[1])
            value = ent_attr_triple[2]

            mask = AVH_mask[en_id, :]
            id = np.argmax(mask==0)
            if id != 0:
                valueEmbedding = liter_encoder.encodeText(str(value))
                AVH[en_id][id] = np.concatenate(([AH[attr_id]], [valueEmbedding]), axis=1)
            i += 1
    else:
        AVH = np.zeros((entityNum, maxAttrNum + 1, 768), dtype=np.float32)
        AVH_mask = np.zeros((entityNum, maxAttrNum + 1), dtype=np.int64)
        for id in id_entName.keys():
            AVH[id][0] = np.concatenate(([EH[id]]), axis=1)
            AVH_mask[id][0] = 1
        for ent_attr_triple in ent_attr_triples:
            en_id = int(ent_attr_triple[0])
            value = ent_attr_triple[2]

            mask = AVH_mask[en_id, :]
            id = np.argmax(mask == 0)
            if id != 0:
                valueEmbedding = liter_encoder.encodeText(str(value))
                AVH[en_id][id] = np.concatenate(([valueEmbedding]), axis=1)

    return AVH
