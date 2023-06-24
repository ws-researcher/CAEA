import os


def load_triples(file_name):
    triples = []
    entity = set()
    rel = set()
    for line in open(file_name, 'r'):
        head, r, tail = [item.strip() for item in line.split("\t")]
        entity.add(head)
        entity.add(tail)
        rel.add(r)
        triples.append((head, r, tail))
    return entity, rel, triples

def load_triples_A(file_name, EID):
    triples = []
    attr = set()
    for line in open(file_name, 'r'):
        head, a, tail = [item.strip() for item in line.split("\t")]
        assert head in EID.keys()
        attr.add(a)
        triples.append((head, a, tail))
    return attr, triples

def load_alignment_pair(file_name, EID1, EID2):
    alignment_pair = []
    for line in open(file_name, 'r'):
        es = [item.strip() for item in line.split("\t")]
        assert es[0] in EID1.keys()
        assert es[1] in EID2.keys()
        eid1, eid2 = EID1[es[0]], EID2[es[1]]
        alignment_pair.append((eid1, eid2))
    return alignment_pair

def ER2ID(set1, set2):
    list1, list2 = list(set1), list(set2)
    num_list1, num_list2 = len(list1), len(list2)
    IDER1 = dict(zip(range(num_list1), list1))
    IDER2 = dict(zip(range(num_list1, num_list1 + num_list2), list2))

    ERID1 = dict(zip(list1, range(num_list1)))
    ERID2 = dict(zip(list2, range(num_list1, num_list1 + num_list2)))

    return IDER1, IDER2, ERID1, ERID2

def ET2ID(triples, EID, RID):
    IDtriples = []
    for t in triples:
        headID = EID[t[0]]
        rID = RID[t[1]]
        tailID = EID[t[2]]
        IDtriples.append((headID, rID, tailID))
    return IDtriples

def AT2ID(triples, EID, AID):
    IDtriples = []
    for t in triples:
        headID = EID[t[0]]
        aID = AID[t[1]]
        string = t[2]
        IDtriples.append((headID, aID, string))
    return IDtriples

def writeFile(path, obj):
    with open(path, "w") as w:
        if isinstance(obj, dict):
            for key in obj.keys():
                value = obj[key]
                w.write(str(key) + "\t" + value + "\n")
        elif isinstance(obj, list):
            for t in obj:
                if len(t) == 3:
                    line = str(t[0]) + "\t" + str(t[1]) + "\t" + str(t[2]) + "\n"
                elif len(t) == 2:
                    line = str(t[0]) + "\t" + str(t[1]) + "\n"
                w.write(line)


if __name__ == '__main__':
    dataPath = "data/MED-BBK"
    datasetPath = "dataSet/MED-BBK"

    entity1, rel1, triples1 = load_triples(os.path.join(dataPath, 'rel_triples_1'))
    entity2, rel2, triples2 = load_triples(os.path.join(dataPath, 'rel_triples_2'))

    IDE1, IDE2, EID1, EID2 = ER2ID(entity1, entity2)
    IDR1, IDR2, RID1, RID2 = ER2ID(rel1, rel2)

    alignment_pair = load_alignment_pair(os.path.join(dataPath, 'ent_links'), EID1, EID2)

    attr1, Atriples1 = load_triples_A(os.path.join(dataPath, 'attr_triples_1'), EID1)
    attr2, Atriples2 = load_triples_A(os.path.join(dataPath, 'attr_triples_2'), EID2)

    IDA1, IDA2, AID1, AID2 = ER2ID(attr1, attr2)

    IDtriples1 = ET2ID(triples1, EID1, RID1)
    IDtriples2 = ET2ID(triples2, EID2, RID2)
    IDAtriples1 = AT2ID(Atriples1, EID1, AID1)
    IDAtriples2 = AT2ID(Atriples2, EID2, AID2)

    writeFile(os.path.join(datasetPath, 'ent_ids_1'), IDE1)
    writeFile(os.path.join(datasetPath, 'ent_ids_2'), IDE2)
    writeFile(os.path.join(datasetPath, 'triples_1'), IDtriples1)
    writeFile(os.path.join(datasetPath, 'triples_2'), IDtriples2)

    writeFile(os.path.join(datasetPath, 'rel_ids_1'), IDR1)
    writeFile(os.path.join(datasetPath, 'rel_ids_2'), IDR2)
    writeFile(os.path.join(datasetPath, 'attr_ids_1'), IDA1)
    writeFile(os.path.join(datasetPath, 'attr_ids_2'), IDA2)

    writeFile(os.path.join(datasetPath, 'ent_attr_1'), IDAtriples1)
    writeFile(os.path.join(datasetPath, 'ent_attr_2'), IDAtriples2)

    writeFile(os.path.join(datasetPath, 'ref_ent_ids'), alignment_pair)