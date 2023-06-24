import rdflib as rdflib
from rdflib import URIRef


def writeFile(path, obj):
    with open(path, "w") as w:
        for t in obj:
            if len(t) == 3:
                line = str(t[0]) + "\t" + str(t[1]) + "\t" + str(t[2]) + "\n"
            elif len(t) == 2:
                line = str(t[0]) + "\t" + str(t[1]) + "\n"
            w.write(line)

CTIKGpath = "./CSKG/CTIKG/CyberSecurity.nt"
SEPSESpath_cwe = "./CSKG/SEPSES_CKB/graph000005_000001.ttl"
SEPSESpath_capec = "./CSKG/SEPSES_CKB/graph000004_000001.ttl"
SEPSESpath_cpe = "./CSKG/SEPSES_CKB/graph000002_000001.ttl"
SEPSESpath_cve1 = "./CSKG/SEPSES_CKB/graph000001_000001.ttl"
SEPSESpath_cve2 = "./CSKG/SEPSES_CKB/graph000001_000002.ttl"
SEPSESpath_cve3 = "./CSKG/SEPSES_CKB/graph000001_000003.ttl"
SEPSESpath_cve4 = "./CSKG/SEPSES_CKB/graph000001_000004.ttl"
SEPSESpath_cve5 = "./CSKG/SEPSES_CKB/graph000001_000005.ttl"
SEPSESpath_cve = "./CSKG/SEPSES_CKB/graph0000001_cve.ttl"

CTIKG = rdflib.Graph()
CTIKG.parse(CTIKGpath, format='nt')


SEPSES = rdflib.Graph()
SEPSES.parse(SEPSESpath_cwe, format='ttl')
SEPSES.parse(SEPSESpath_capec, format='ttl')
# SEPSES.parse(SEPSESpath_cve, format='ttl')
# SEPSES.parse(SEPSESpath_cpe, format='ttl')
# SEPSES.parse(SEPSESpath_cve1, format='ttl')
# SEPSES.parse(SEPSESpath_cve2, format='ttl')
# SEPSES.parse(SEPSESpath_cve3, format='ttl')
# SEPSES.parse(SEPSESpath_cve4, format='ttl')
# SEPSES.parse(SEPSESpath_cve5, format='ttl')

CTIKGnodes = set([node for node in CTIKG.all_nodes() if isinstance(node, rdflib.URIRef)])
SEPSESnodes = set([node for node in SEPSES.all_nodes() if isinstance(node, rdflib.URIRef)])

alignNodes = set()
for CTIKGnode in CTIKGnodes:
    num = 0
    for SEPSESnode in SEPSESnodes:

        if str(CTIKGnode).split("/")[-1].startswith("AP") and str(SEPSESnode).split("/")[-1].startswith("CAPEC"):
            APID_CTIKG = str(CTIKGnode).split("/")[-1].split("_")[-1]
            APID_SEPSES = str(SEPSESnode).split("/")[-1].split("-")[-1]

            if APID_CTIKG == APID_SEPSES:
                alignNodes.add((str(CTIKGnode), str(SEPSESnode)))
                num += 1

        if str(CTIKGnode).split("/")[-1].startswith("CWE") and str(SEPSESnode).split("/")[-1].startswith("CWE"):
            CWEID_CTIKG = str(CTIKGnode).split("/")[-1].split("_")[-1]
            CWEID_SEPSES = str(SEPSESnode).split("/")[-1].split("-")[-1]

            if CWEID_CTIKG == CWEID_SEPSES:
                alignNodes.add((str(CTIKGnode), str(SEPSESnode)))
                num += 1

        if "det-" in str(CTIKGnode).split("/")[-1] and "detectionMethod" in str(SEPSESnode):
            detMethod_CTIKG = str(CTIKGnode).split("/")[-1][4:].replace("_", " ")
            detMethod_SEPSESs = set()
            for detMethod_SEPSES in SEPSES.objects(SEPSESnode, URIRef("http://w3id.org/sepses/vocab/ref/cwe#detectionMethod")):
                detMethod_SEPSESs.add(str(detMethod_SEPSES))
            if detMethod_CTIKG == list(detMethod_SEPSESs)[0]:
                alignNodes.add((str(CTIKGnode), str(SEPSESnode)))
                num += 1

        if str(CTIKGnode).split("/")[-1].startswith("CVE") and str(SEPSESnode).split("/")[-1].startswith("CVE"):
            CVEID_CTIKG = str(CTIKGnode).split("/")[-1]
            CVEID_SEPSES = str(SEPSESnode).split("/")[-1]

            if CVEID_CTIKG == CVEID_SEPSES:
                alignNodes.add((str(CTIKGnode), str(SEPSESnode)))
                num += 1

    if num > 1:
        print("s")

rel_triples_1, rel_triples_2 = set(), set()
attr_triples_1, attr_triples_2 = set(), set()
for subj, pred, obj in CTIKG:
    if isinstance(obj, rdflib.URIRef):
        rel_triples_1.add((subj, pred, obj))
    if isinstance(obj, rdflib.Literal):
        obj = str(obj).replace("\n", " ").replace("\t", " ")
        attr_triples_1.add((subj, pred, obj))

for subj, pred, obj in SEPSES:
    if isinstance(obj, rdflib.URIRef):
        rel_triples_2.add((subj, pred, obj))
    if isinstance(obj, rdflib.Literal):
        obj =str(obj).replace("\n", " ").replace("\t", " ")
        attr_triples_2.add((subj, pred, obj))

print(len(alignNodes))
writeFile("./ent_links", alignNodes)
writeFile("./rel_triples_1", rel_triples_1)
writeFile("./rel_triples_2", rel_triples_2)
writeFile("./attr_triples_1", attr_triples_1)
writeFile("./attr_triples_2", attr_triples_2)