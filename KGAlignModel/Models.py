from tensorflow import keras
import tensorflow as tf

from KGAlignModel.BaseLayer import MutilHeadAttention, atte_aggr, avg_aggr
from KGAlignModel.KGAlignLayer import AVlyaer, FHopGATlayer, FHopGCNlayer
from KGAlignModel.Loss import get_loss_func


class AlignModel(keras.Model):
    def __init__(self, ADJ, AVH, RH, kgs, args):
        super().__init__()
        self.args = args
        self.kgs = kgs
        self.ADJ = ADJ
        self.ADJ1 = dict()
        self.ADJ2 = dict()
        self.AVlayer1 = None
        self.AVlayer2 = None
        self.FHopGATlayer1 = None
        self.FHopGATlayer2 = None

        self.AVH = AVH
        self.RH = RH

        # self.ADJ1["adjs"] = self.ADJ["adjs1"]
        # self.ADJ1["adjp"] = self.ADJ["adjp1"]
        # self.ADJ1["adjc"] = self.ADJ["adjc1"]
        #
        # self.ADJ2["adjs"] = self.ADJ["adjs1"]
        # self.ADJ2["adjp"] = self.ADJ["adjp1"]
        # self.ADJ2["adjc"] = self.ADJ["adjc1"]

        self.AVlayer1 = AVlyaer(self.args)


        self.FHopGATlayer1 = FHopGATlayer(self.args, self.ADJ, self.kgs.entities_num)

        # self.FHopGATlayer1 = FHopGCNlayer(self.args, self.ADJ, self.kgs.entities_num)

        # self.aggr = avg_aggr(self.args.AClyaerDim)
        self.aggr = atte_aggr(self.args.AClyaerDim)

        self.d = tf.keras.layers.Dense(300, activation="softmax")

        pass


    # def call(self, input):
    #     input1 = input[0]
    #     input2 = input[1]
    #
    #     h1 = self.AVlayer1(input1)
    #     h2 = self.AVlayer2(input2)
    #
    #     ah1 = self.FHopGATlayer1(h1)
    #     ah2 = self.FHopGATlayer1(h2)
    #     return ah1, ah2


    def getLoss(self, pos_links, neg_links):

        h1 = self.AVlayer1(self.AVH)

        h1 = self.FHopGATlayer1(h1, training=True)

        h1 = self.aggr(h1)

        h1 = self.d(h1)

        h1 = tf.nn.l2_normalize(h1, 1)

        index1 = pos_links[:, 0]
        index2 = pos_links[:, 1]
        neg_index1 = neg_links[:, 0]
        neg_index2 = neg_links[:, 1]

        input1 = tf.nn.embedding_lookup(h1, tf.cast(index1, tf.int32))
        input2 = tf.nn.embedding_lookup(h1, tf.cast(index2, tf.int32))

        neg_input1 = tf.nn.embedding_lookup(h1, tf.cast(neg_index1, tf.int32))
        neg_input2 = tf.nn.embedding_lookup(h1, tf.cast(neg_index2, tf.int32))

        pos_loss = tf.reduce_sum(tf.reduce_sum(tf.square(input1 - input2), 1))
        neg_distance = tf.reduce_sum(tf.square(neg_input1 - neg_input2), 1)
        neg_loss = tf.reduce_sum(tf.keras.activations.relu(self.args.neg_margin - neg_distance))

        ali_loss = (pos_loss + self.args.neg_margin_balance * neg_loss)/len(index1)


        # kg1index1 = list(zip(*kg1Batch))[0]
        # kg1index2 = list(zip(*kg1Batch))[1]
        # kg1input1 = tf.nn.embedding_lookup(h1, tf.cast(kg1index1, tf.int32))
        # kg1input2 = tf.nn.embedding_lookup(h1, tf.cast(kg1index2, tf.int32))
        # distance1 = tf.reduce_sum(tf.square(kg1input1 - kg1input2), 1)
        # discreteLoss1 = tf.reduce_sum(tf.keras.activations.relu(self.args.neg_margin - distance1))/len(kg1index1)
        #
        #
        # kg2index1 = list(zip(*kg2Batch))[0]
        # kg2index2 = list(zip(*kg2Batch))[1]
        # kg2input1 = tf.nn.embedding_lookup(h1, tf.cast(kg2index1, tf.int32))
        # kg2input2 = tf.nn.embedding_lookup(h1, tf.cast(kg2index2, tf.int32))
        # distance2 = tf.reduce_sum(tf.square(kg2input1 - kg2input2), 1)
        # discreteLoss2 = tf.reduce_sum(tf.keras.activations.relu(self.args.neg_margin - distance2))/len(kg2index1)

        loss = ali_loss

        return loss, h1


        # loss  = get_loss_func(pes, pas, pvs, nes, nas, nvs, self.args)
