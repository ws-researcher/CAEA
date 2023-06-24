import os
import random
from keras.layers import Embedding
from tensorflow import keras
from tensorflow.keras import layers, initializers, activations
from tensorflow.python.keras.backend import gather, concatenate, dot, expand_dims
import tensorflow as tf
from tensorflow.python.keras.layers.core import Dropout
import numpy as np

from moduleUtil import AVlyaer

def set_seeds(seed=0):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = str(seed)

set_seeds(0)

class ADEA(keras.Model):

    def __init__(self, arg, node_size, rel_size, attr_size, index_matrix=None, all_matix=None, attr_matrix=None, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.args = arg
        self.node_size = node_size
        self.rel_size = rel_size
        self.attr_size = attr_size

        self.ent_emb_layer = TokenEmbedding(node_size, self.args.input_dim, trainable=True)
        self.rel_emb_layer = TokenEmbedding(rel_size, self.args.rel_dim, trainable=True)
        self.attr_emb_layer = TokenEmbedding(attr_size, self.args.attr_dim, trainable=True)

        self.ent_emb = self.ent_emb_layer(1)
        self.rel_emb = self.rel_emb_layer(1)
        self.attr_emb = self.attr_emb_layer(1)

        # self.litral_emb_layer = AVlyaer(self.args)

        self.entLayer = GRAT(arg)
        # self.entLayer = GCAT(arg)

        self.tanh = activations.get('tanh')
        self.relu = activations.get('relu')
        self.LeakyReLU = keras.layers.LeakyReLU(alpha=0.3)

    def call(self, AVH=None, index_matrix=None, all_matix=None, attr_matrix=None, training=None, mask=None):
        # -----------------------------------------------------------------------------------
        # AVH = AVH[:,:,0:1]
        # litral_emb = self.litral_emb_layer(AVH)
        # -----------------------------------------------------------------------------------
        ent_rel_index = all_matix[:, 3:5].astype(np.int64)
        rel_index, rel_id = tf.raw_ops.UniqueV2(x=ent_rel_index, axis=[0])
        rel_adj = tf.SparseTensor(indices=rel_index, values=tf.ones_like(rel_index[:, 0], dtype="float32"),
                                  dense_shape=(self.node_size, self.rel_size))
        rel_adj = tf.sparse.softmax(rel_adj)
        concept_rel = tf.sparse.sparse_dense_matmul(rel_adj, self.rel_emb)
        concept_rel = self.relu(concept_rel)

        # ------------------------------------------------------------------------------------
        ent_attr_index = attr_matrix[:, 0:2].astype(np.int64)
        attr_index, _ = tf.raw_ops.UniqueV2(x=ent_attr_index, axis=[0])
        attr_adj = tf.SparseTensor(indices=attr_index, values=tf.ones_like(attr_index[:, 0], dtype="float32"),
                                   dense_shape=(self.node_size, self.attr_size))
        attr_adj = tf.sparse.softmax(attr_adj)
        concept_attr = tf.sparse.sparse_dense_matmul(attr_adj, self.attr_emb)
        concept_attr = self.relu(concept_attr)

        # ------------------------------------------------------------------------------------
        ent_emb = self.entLayer(self.ent_emb, rel_emb=self.rel_emb, attr_emb=self.attr_emb,
                                index_matrix=index_matrix, all_matix=all_matix, attr_matrix=attr_matrix,
                                concept_attr=concept_attr, concept_rel=concept_rel)

        # ------------------------------------------------------------------------------------
        ent_emb = concatenate([ent_emb, concept_rel, concept_attr])
        # ent_emb = concatenate([ent_emb, concept_rel])
        # ent_emb = concatenate([ent_emb, concept_attr])
        # ent_emb = concatenate([ent_emb, litral_emb, concept_rel, concept_attr])
        # ent_emb = concatenate([ent_emb])
        conc_emb = concatenate([concept_rel, concept_attr])
        if training:
            ent_emb = Dropout(self.args.dropout_rate)(ent_emb)
            return ent_emb, conc_emb
        else:
            return ent_emb, conc_emb


class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        # self.embeddings_initializer = initializers.get("glorot_uniform")

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings


class GRAT(layers.Layer):
    def __init__(self, arg, **kwargs):
        super().__init__(**kwargs)
        self.args = arg

        self.ent_kernels = []
        self.rel_kernels = []

        self.ent_attn_kernels = []
        self.rel_attn_kernels = []

        self.concept_rel_attn_kernels = []
        self.concept_rel_kernels = []

        self.concept_attr_attn_kernels = []
        self.concept_attr_kernels = []

        self.LeakyReLU = keras.layers.LeakyReLU(alpha=0.3)
        self.tanh = activations.get('tanh')
        self.relu = activations.get('relu')

    def build(self, input_shape):
        for l in range(self.args.num_layers):
            self.ent_attn_kernels.append([])
            self.rel_attn_kernels.append([])

            self.ent_kernels.append([])
            self.rel_kernels.append([])

            self.concept_rel_attn_kernels.append([])
            self.concept_rel_kernels.append([])

            self.concept_attr_attn_kernels.append([])
            self.concept_attr_kernels.append([])

            num_head = self.args.head
            output_dim_head = int(self.args.output_dim / num_head)
            rel_dim_head = self.args.rel_dim
            attr_dim_head = self.args.attr_dim

            for head in range(num_head):
                ent_attn_kernel = self.add_weight(name="ent_attn_kernel_{}".format(head),
                                                  shape=(2 * output_dim_head, 1),
                                                  initializer=initializers.get("glorot_uniform"),
                                                  trainable=True)

                concept_attr_attn_kernel = self.add_weight(name="concept_attr_attn_kernel_{}".format(head),
                                                           shape=(2 * attr_dim_head + 2 * rel_dim_head, 1),
                                                           initializer=initializers.get("glorot_uniform"),
                                                           trainable=True)

                ent_kernel = self.add_weight(name="ent_kernel_{}".format(head),
                                             shape=(output_dim_head, output_dim_head),
                                             initializer=initializers.get("glorot_uniform"),
                                             trainable=True)

                concept_attr_kernel = self.add_weight(name="concept_attr_kernel_{}".format(head),
                                                      shape=(2 * attr_dim_head, 2 * attr_dim_head),
                                                      initializer=initializers.get("glorot_uniform"),
                                                      trainable=True)

                concept_rel_kernel = self.add_weight(name="concept_rel_kernel_{}".format(head),
                                                     shape=(2 * attr_dim_head, 2 * attr_dim_head),
                                                     initializer=initializers.get("glorot_uniform"),
                                                     trainable=True)

                self.ent_attn_kernels[l].append(ent_attn_kernel)

                self.ent_kernels[l].append(ent_kernel)

                self.concept_rel_kernels[l].append(concept_rel_kernel)

                self.concept_attr_attn_kernels[l].append(concept_attr_attn_kernel)
                self.concept_attr_kernels[l].append(concept_attr_kernel)

    def call(self, inputs, **kwargs):
        ent_emb = inputs
        concept_attr = kwargs.get("concept_attr")
        concept_rel = kwargs.get("concept_rel")
        node_size = ent_emb.shape[0]

        all_matix = kwargs.get("all_matix")
        ent_ent_index = all_matix[:, 0:2].astype(np.int64)
        index, idx = tf.raw_ops.UniqueV2(x=ent_ent_index, axis=[0])
        ent_ent_value = all_matix[:, 2]
        ent_ent_value = tf.math.segment_mean(ent_ent_value, idx)

        ent_adj = tf.SparseTensor(indices=index, values=tf.cast(ent_ent_value, dtype="float32"),
                                  dense_shape=(node_size, node_size))
        ent_adj = tf.sparse.softmax(ent_adj)
        ent_emb = tf.sparse.sparse_dense_matmul(ent_adj, ent_emb)

        outputs = []
        for l in range(self.args.num_layers):
            ent_emb = self.relu(ent_emb)
            head_emb_list = tf.transpose(tf.reshape(ent_emb, (node_size, self.args.head, -1)), perm=[1, 0, 2])
            head_feature_list = []
            for head in range(self.args.head):
                ent_emb = head_emb_list[head]

                ent_attn_kernel = self.ent_attn_kernels[l][head]
                concept_attn_kernel = self.concept_attr_attn_kernels[l][head]

                ent_kernel = self.ent_kernels[l][head]
                concept_rel_kernel = self.concept_rel_kernels[l][head]
                concept_attr_kernel = self.concept_attr_kernels[l][head]

                neighs_concept_rel_feature = gather(concept_rel, index[:, 1])
                neighs_concept_attr_feature = gather(concept_attr, index[:, 1])
                neighs_concept_feature = concatenate([neighs_concept_rel_feature, neighs_concept_attr_feature])
                # neighs_concept_feature = concatenate([neighs_concept_rel_feature])
                w_neighs_concept_feature = tf.matmul(neighs_concept_feature, concept_rel_kernel)
                w_neighs_concept_feature = self.relu(w_neighs_concept_feature)

                self_concept_rel_feature = gather(concept_rel, index[:, 0])
                self_concept_attr_feature = gather(concept_attr, index[:, 0])
                self_concept_feature = concatenate([self_concept_rel_feature, self_concept_attr_feature])
                # self_concept_feature = concatenate([self_concept_rel_feature])
                w_self_concept_feature = tf.matmul(self_concept_feature, concept_attr_kernel)
                w_self_concept_feature = self.relu(w_self_concept_feature)

                neighs_feature = gather(ent_emb, index[:, 1])
                self_feature = gather(ent_emb, index[:, 0])

                w_neighs_feature = tf.matmul(neighs_feature, ent_kernel)
                w_neighs_feature = self.relu(w_neighs_feature)
                w_self_feature = tf.matmul(self_feature, ent_kernel)
                w_self_feature = self.relu(w_self_feature)

                ent_attn = tf.squeeze(dot(concatenate([w_self_feature, w_neighs_feature]), ent_attn_kernel), axis=-1)
                ent_attn = self.LeakyReLU(ent_attn)

                concept_attn = tf.squeeze(
                    dot(concatenate([w_self_concept_feature, w_neighs_concept_feature]), concept_attn_kernel), axis=-1)
                concept_attn = self.LeakyReLU(concept_attn)

                # attn = ent_attn * concept_attn
                attn = ent_attn
                # attn = tf.ones_like(attn)
                attn = tf.nn.softmax(attn, axis=-1)
                attn = tf.SparseTensor(indices=index, values=attn, dense_shape=(node_size, node_size))
                attn = tf.sparse.softmax(attn)

                new_ent_emb = tf.math.segment_sum(neighs_feature * expand_dims(attn.values, axis=-1),
                                                  index[:, 0])

                head_feature_list.append(new_ent_emb)

            ent_feature = concatenate(head_feature_list)
            ent_feature = self.tanh(ent_feature)

            ent_emb = ent_feature
            outputs.append(ent_feature)

        outputs = concatenate(outputs)
        return outputs


class GCAT(layers.Layer):
    def __init__(self, arg, **kwargs):
        super().__init__(**kwargs)
        self.args = arg

        self.attn_kernels = []
        # self.concept_kernels = []

        self.LeakyReLU = keras.layers.LeakyReLU(alpha=0.3)
        self.tanh = activations.get('tanh')
        self.relu = activations.get('relu')

    def build(self, input_shape):

        num_head = self.args.head
        output_dim_head = int(self.args.output_dim / num_head)
        rel_dim_head = self.args.rel_dim
        attr_dim_head = self.args.attr_dim

        for l in range(self.args.num_layers):

            self.attn_kernels.append([])

            for head in range(num_head):

                attn_kernel = self.add_weight(name="attn_kernel",
                                              shape=((output_dim_head + rel_dim_head + attr_dim_head) * 2 + rel_dim_head, 1),
                                              initializer=initializers.get("glorot_uniform"),
                                              trainable=True)

                self.attn_kernels[l].append(attn_kernel)

    def call(self, inputs, **kwargs):
        ent_emb = inputs
        rel_emb = kwargs.get("rel_emb")
        attr_emb = kwargs.get("attr_emb")

        node_size = ent_emb.shape[0]
        rel_size = rel_emb.shape[0]
        attr_size = attr_emb.shape[0]

        all_matix = kwargs.get("all_matix")
        attr_matrix = kwargs.get("attr_matrix")

        ent_ent_index = all_matix[:, 0:2].astype(np.int64)
        index, idx = tf.raw_ops.UniqueV2(x=ent_ent_index, axis=[0])
        ent_ent_value = all_matix[:, 2]
        ent_ent_value = tf.math.segment_mean(ent_ent_value, idx)

        ent_rel_index = all_matix[:, 3:5].astype(np.int64)
        rel_index, rel_id = tf.raw_ops.UniqueV2(x=ent_rel_index, axis=[0])
        rel_adj = tf.SparseTensor(indices=rel_index, values=tf.ones_like(rel_index[:, 0], dtype="float32"),
                                  dense_shape=(node_size, rel_size))
        rel_adj = tf.sparse.softmax(rel_adj)
        concept_rel = tf.sparse.sparse_dense_matmul(rel_adj, rel_emb)
        concept_rel = self.relu(concept_rel)

        ent_attr_index = attr_matrix[:, 0:2].astype(np.int64)
        attr_index, _ = tf.raw_ops.UniqueV2(x=ent_attr_index, axis=[0])
        attr_adj = tf.SparseTensor(indices=attr_index, values=tf.ones_like(attr_index[:, 0], dtype="float32"),
                                   dense_shape=(node_size, attr_size))
        attr_adj = tf.sparse.softmax(attr_adj)
        concept_attr = tf.sparse.sparse_dense_matmul(attr_adj, attr_emb)
        concept_attr = self.relu(concept_attr)

        ent_ent_value = tf.ones((index.shape[0], ))
        ent_adj = tf.SparseTensor(indices=index, values=tf.cast(ent_ent_value, dtype="float32"),
                                  dense_shape=(node_size, node_size))
        ent_adj = tf.sparse.softmax(ent_adj)
        ent_emb = tf.sparse.sparse_dense_matmul(ent_adj, ent_emb)

        outputs = []
        for l in range(self.args.num_layers):
            ent_emb = self.relu(ent_emb)
            head_emb_list = tf.transpose(tf.reshape(ent_emb, (node_size, self.args.head, -1)), perm=[1, 0, 2])
            head_feature_list = []
            for head in range(self.args.head):
                ent_emb = head_emb_list[head]

                self_feature = gather(ent_emb, index[:, 0])
                self_concept_rel_feature = gather(concept_rel, index[:, 0])
                self_concept_attr_feature = gather(concept_attr, index[:, 0])
                self_concept_feature = concatenate([self_concept_rel_feature, self_concept_attr_feature])

                neighs_feature = gather(ent_emb, index[:, 1])
                neighs_concept_rel_feature = gather(concept_rel, index[:, 1])
                neighs_concept_attr_feature = gather(concept_attr, index[:, 1])
                neighs_concept_feature = concatenate([neighs_concept_rel_feature, neighs_concept_attr_feature])

                rels_feature = gather(rel_emb, ent_rel_index[:, 1])
                rels_feature = tf.math.segment_mean(rels_feature, idx)

                attn_kernel = self.attn_kernels[l][head]
                attn = tf.squeeze(
                    dot(concatenate([self_feature, self_concept_feature, rels_feature, neighs_concept_feature, neighs_feature]), attn_kernel),
                    axis=-1)
                attn = self.LeakyReLU(attn)
                # attn = tf.ones_like(attn)
                attn = tf.nn.softmax(attn, axis=-1)
                attn = tf.SparseTensor(indices=index, values=attn, dense_shape=(node_size, node_size))
                attn = tf.sparse.softmax(attn)

                ent_emb = head_emb_list[head]
                neighs_feature = gather(ent_emb, index[:, 1])
                new_ent_emb = tf.math.segment_sum(neighs_feature * expand_dims(attn.values, axis=-1),
                                                  index[:, 0])

                head_feature_list.append(new_ent_emb)

            ent_feature = concatenate(head_feature_list)
            ent_feature = self.tanh(ent_feature)

            ent_emb = ent_feature
            outputs.append(ent_feature)

        outputs = concatenate(outputs)
        return outputs


class GAT(layers.Layer):
    def __init__(self, arg, **kwargs):
        super().__init__(**kwargs)
        self.args = arg

        self.ent_attn_kernels = []
        self.rel_attn_kernels = []

        self.ent_kernels = []

        self.LeakyReLU = keras.layers.LeakyReLU(alpha=0.3)
        self.tanh = activations.get('tanh')
        self.relu = activations.get('relu')

    def build(self, input_shape):
        for l in range(self.args.num_layers):
            self.ent_attn_kernels.append([])

            self.ent_kernels.append([])

            num_head = self.args.head
            output_dim_head = int(self.args.output_dim / num_head)

            for head in range(num_head):
                ent_attn_kernel = self.add_weight(name="ent_attn_kernel_{}".format(head),
                                                  shape=(2 * output_dim_head, 1),
                                                  initializer=initializers.get("glorot_uniform"),
                                                  trainable=True)

                ent_kernel = self.add_weight(name="ent_kernel_{}".format(head),
                                             shape=(output_dim_head, output_dim_head),
                                             initializer=initializers.get("glorot_uniform"),
                                             trainable=True)

                self.ent_attn_kernels[l].append(ent_attn_kernel)

                self.ent_kernels[l].append(ent_kernel)

    def call(self, inputs, **kwargs):
        ent_emb = inputs
        node_size = ent_emb.shape[0]

        all_matix = kwargs.get("all_matix")
        ent_ent_index = all_matix[:, 0:2].astype(np.int64)
        index, idx = tf.raw_ops.UniqueV2(x=ent_ent_index, axis=[0])
        ent_ent_value = all_matix[:, 2]
        ent_ent_value = tf.math.segment_mean(ent_ent_value, idx)

        ent_adj = tf.SparseTensor(indices=index, values=tf.cast(ent_ent_value, dtype="float32"),
                                  dense_shape=(node_size, node_size))
        ent_adj = tf.sparse.softmax(ent_adj)
        ent_emb = tf.sparse.sparse_dense_matmul(ent_adj, ent_emb)

        outputs = []
        for l in range(self.args.num_layers):
            ent_emb = self.relu(ent_emb)
            head_emb_list = tf.transpose(tf.reshape(ent_emb, (node_size, self.args.head, -1)), perm=[1, 0, 2])
            head_feature_list = []
            for head in range(self.args.head):
                ent_emb = head_emb_list[head]

                ent_attn_kernel = self.ent_attn_kernels[l][head]

                ent_kernel = self.ent_kernels[l][head]

                neighs_feature = gather(ent_emb, index[:, 1])
                self_feature = gather(ent_emb, index[:, 0])

                w_neighs_feature = tf.matmul(neighs_feature, ent_kernel)
                w_neighs_feature = self.relu(w_neighs_feature)
                w_self_feature = tf.matmul(self_feature, ent_kernel)
                w_self_feature = self.relu(w_self_feature)

                ent_attn = tf.squeeze(dot(concatenate([w_self_feature, w_neighs_feature]), ent_attn_kernel), axis=-1)
                ent_attn = self.LeakyReLU(ent_attn)

                attn = tf.nn.softmax(ent_attn, axis=-1)
                attn = tf.SparseTensor(indices=index, values=attn, dense_shape=(node_size, node_size))
                attn = tf.sparse.softmax(attn)

                new_ent_emb = tf.math.segment_sum(neighs_feature * expand_dims(attn.values, axis=-1),
                                                  index[:, 0])

                head_feature_list.append(new_ent_emb)

            ent_feature = concatenate(head_feature_list)
            ent_feature = self.tanh(ent_feature)

            ent_emb = ent_feature
            outputs.append(ent_feature)

        outputs = concatenate(outputs)
        return outputs
