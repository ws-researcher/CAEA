import os
import time



import math
import random
from tensorflow import keras
import tensorflow as tf

from KGAlignModel.Batch import generate_relation_triple_batch, generate_align_batch, generate_ent_batch, \
    generate_entityPair_batch
from KGAlignModel.Models import AlignModel
from KGAlignModel.evaluation import test, valid, early_stop
from modules.utils.util import buildInput


class kgalign(keras.Model):

    def set_kgs(self, kgs):
        self.kgs = kgs
        self.ADJ, self.AVH, self.RH, self.ADJR = buildInput("./dataSerialization/", self.kgs, self.args, hop=self.args.hop)

    def set_args(self, args):
        self.args = args
        # self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
        #                                       self.__class__.__name__)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = None
        # self.out_folder = None
        self.kgs = None
        self.ADJ = None
        # self.ADJ1 = dict()
        # self.ADJ2 = dict()
        self.AVH = None
        self.RH = None
        # self.AVlayer1 = None
        # self.AVlayer2 = None
        # self.FHopGATlayer1 = None
        # self.FHopGATlayer2 = None
        #
        # self.ADJR1 = None
        # self.ADJR2 = None
        self.AlignModel = None
        self.ent_embeds = None

        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = False
        self.mapping_mat = None

    def initModel(self):
        self.AlignModel = AlignModel(self.ADJ,self.AVH, self.RH, self.kgs, self.args)


    def train(self):
        # strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])

        kg1 = self.kgs.kg1
        kg2 = self.kgs.kg2


        checkpoint = tf.train.Checkpoint(myModel=self.AlignModel)

        # if os.path.exists('/home/Edwin-01/workplace/KGAlign/KGAlign1/run/model_save/checkpoint'):  # 判断模型是否存在
        #     checkpoint.restore(tf.train.latest_checkpoint('/home/Edwin-01/workplace/KGAlign/KGAlign1/run/model_save'))


        for epoach in range(self.args.max_epoch):
            epoachLoss = 0
            triBatchList1 = generate_relation_triple_batch(kg1, self.args)
            triBatchList2 = generate_relation_triple_batch(kg2, self.args)


            batchNum = len(triBatchList1)
            # batchNum = max(math.ceil(kg1.entities_num/self.args.kg_batch_size), math.ceil(kg2.entities_num/self.args.kg_batch_size))
            for batch in range(batchNum):
                # batch1 = triBatchList1[batch]
                # batch2 = triBatchList2[batch]
                # pos_batch1 = batch1[0]
                # neg_batch1 = batch1[1]
                # pos_batch2 = batch2[0]
                # neg_batch2 = batch2[1]
                #
                # loss1 = self.TransdLayer1.get_batch_loss(pos_batch1, neg_batch1)
                # loss2 = self.TransdLayer2.get_batch_loss(pos_batch2, neg_batch2)

                alignBatch = generate_align_batch(self.kgs, self.args, neighbors1=None, neighbors2=None)
                pos_batch = alignBatch[0]
                neg_batch = alignBatch[1]

                # kg1Batch, kg2Batch = generate_ent_batch(self.kgs, self.args)

                # loss, self.ent_embeds = self.AlignModel.getLoss(pos_batch, neg_batch)
                # with strategy.scope():
                with tf.GradientTape() as tape:
                    tape.watch(self.variables)
                    # loss, self.ent_embeds = self.AlignModel.getLoss(pos_batch, neg_batch, kg1Batch, kg2Batch)
                    loss, self.ent_embeds = self.AlignModel.getLoss(pos_batch, neg_batch)
                    g = tape.gradient(loss, self.trainable_variables)
                    epoachLoss += loss
                tf.keras.optimizers.Adam(self.args.learning_rate).apply_gradients(zip(g, self.trainable_variables))
            if epoach%50 == 0 and epoach != 0:
                if not os.path.exists('/home/ws/Workplace/KGAlign/run/model_save'):
                    os.mkdir('/home/ws/Workplace/KGAlign/run/model_save')
                checkpoint.save('/home/ws/Workplace/KGAlign/run/model_save/model{}.ckpt'.format(epoach))
            print("epoach:{},epoachLoss:{}".format(epoach, loss))

            if epoach >= self.args.start_valid and epoach % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or epoach == self.args.max_epoch:
                    break
            pass

    def test(self):
        checkpoint_test = tf.train.Checkpoint(myModel=self.AlignModel)  # 实例化Checkpoint，指定恢复对象为model
        checkpoint_test.restore(tf.train.latest_checkpoint('../model_save'))

        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities1)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities2)
        mapping = self.mapping_mat if self.mapping_mat is not None else None
        rest_12, _, _ = test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
                             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls, accurate=True)

    #     print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))


    def valid(self, stop_metric):
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities1)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities2)
        mapping = self.mapping_mat if self.mapping_mat is not None else None
        hits1_12, mrr_12 = valid(embeds1, embeds2, mapping, self.args.top_k,
                                 self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12