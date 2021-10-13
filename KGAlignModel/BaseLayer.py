import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, initializers, regularizers, constraints, activations
import numpy as np

def scaled_dot_product_attention(q, k, v, mask):
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # attention 乘上value
    output = tf.matmul(attention_weights, v) # （.., seq_len_v, depth）

    return output, attention_weights

class MutilHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MutilHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0
        # 分头后的维度
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        # 分头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) # (batch_size, seq_len_v, num_heads, depth)

        # 合并多头
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # 全连接重塑
        output = self.dense(concat_attention)
        return output, attention_weights


class GraphAttentionLayer(layers.Layer):

    def __init__(self,
                 output_dim,
                 adj,
                 nodes_num,
                 dropout_rate=0.0,
                 **kwargs):
        super(GraphAttentionLayer, self).__init__()
        self.activation = activations.get('tanh')
        self.kernel_initializer = initializers.GlorotUniform()
        self.bias_initializer = initializers.zeros()
        self.kernel_regularizer = regularizers.get('l2')
        self.bias_regularizer = regularizers.get('l2')

        self.output_dim = output_dim
        self.adj = tf.SparseTensor(indices=adj[0], values=adj[1], dense_shape=adj[2])
        self.dropout_rate = dropout_rate
        self.nodes_num = nodes_num

        self.kernel, self.kernel1, self.kernel2 = None, None, None

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=(input_shape[-1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        self.kernel1 = self.add_weight('kernel1',
                                       shape=(input_shape[-1], input_shape[-1]),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       trainable=True)
        self.kernel2 = self.add_weight('kernel1',
                                       shape=(input_shape[-1], input_shape[-1]),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       trainable=True)

    def call(self, inputs, training=True):
        inputs = tf.reshape(inputs, (inputs.shape[0], inputs.shape[-1]))

        inputs = tf.keras.layers.BatchNormalization()(inputs)
        mapped_inputs = tf.matmul(inputs, self.kernel)
        attention_inputs1 = tf.matmul(inputs, self.kernel1)
        attention_inputs2 = tf.matmul(inputs, self.kernel2)
        con_sa_1 = tf.reduce_sum(tf.multiply(attention_inputs1, inputs), 1, keepdims=True)
        con_sa_2 = tf.reduce_sum(tf.multiply(attention_inputs2, inputs), 1, keepdims=True)
        con_sa_1 = tf.keras.activations.tanh(con_sa_1)
        con_sa_2 = tf.keras.activations.tanh(con_sa_2)
        if training and self.dropout_rate > 0.0:
            con_sa_1 = tf.nn.dropout(con_sa_1, self.dropout_rate)
            con_sa_2 = tf.nn.dropout(con_sa_2, self.dropout_rate)
        con_sa_1 = tf.cast(self.adj, dtype=tf.float32) * con_sa_1
        con_sa_2 = tf.cast(self.adj, dtype=tf.float32) * tf.transpose(con_sa_2, [1, 0])
        weights = tf.sparse.add(con_sa_1, con_sa_2)
        weights = tf.SparseTensor(indices=weights.indices,
                                  values=tf.nn.leaky_relu(weights.values),
                                  dense_shape=weights.dense_shape)
        attention_adj = tf.sparse.softmax(weights)
        attention_adj = tf.sparse.reshape(attention_adj, shape=[self.nodes_num, self.nodes_num])
        value = tf.sparse.sparse_dense_matmul(attention_adj, mapped_inputs)
        return self.activation(value)

class GCNLayer(layers.Layer):

    def __init__(self,
                 output_dim,
                 adj,
                 nodes_num,
                 dropout_rate=0.0,
                 **kwargs):
        super(GCNLayer, self).__init__()
        self.activation = activations.get('tanh')
        self.kernel_initializer = initializers.GlorotUniform()
        self.bias_initializer = initializers.zeros()
        self.kernel_regularizer = regularizers.get('l2')
        self.bias_regularizer = regularizers.get('l2')

        self.output_dim = output_dim
        self.adj = tf.SparseTensor(indices=adj[0], values=adj[1], dense_shape=adj[2])
        self.dropout_rate = dropout_rate
        self.nodes_num = nodes_num

        self.kernel, self.kernel1, self.kernel2 = None, None, None

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=(input_shape[-1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

    def call(self, inputs, training=True):
        inputs = tf.keras.layers.BatchNormalization()(inputs)
        mapped_inputs = tf.matmul(inputs, self.kernel)


        attention_adj = tf.sparse.softmax(self.adj)
        attention_adj = tf.sparse.reshape(attention_adj, shape=[self.nodes_num, self.nodes_num])

        attention_adj = tf.cast(attention_adj, dtype=tf.float32)
        value = tf.sparse.sparse_dense_matmul(attention_adj, mapped_inputs)
        return self.activation(value)

class avg_aggr(layers.Layer):
    def __init__(self, d_model):
        super(avg_aggr, self).__init__()
        self.w = tf.keras.layers.Dense(d_model)

    def call(self, h):
        h = self.w(h)  # (batch_size, seq_len, d_model)
        mean, variance = tf.nn.moments(h, axes=1)
        return mean

class atte_aggr(layers.Layer):
    def __init__(self, d_model):
        super(atte_aggr, self).__init__()
        self.d_model = d_model

    def build(self, input_shape):
        self.MutilHeadAttentionLyaer = MutilHeadAttention(d_model=self.d_model, num_heads=1)
        pass

    def call(self, h):
        output, _ = self.MutilHeadAttentionLyaer(h, h, h, mask=None)
        mean, variance = tf.nn.moments(output, axes=1)
        # output = output[:, 0, :]
        return mean

class HighwayLayer(layers.Layer):
    """Highway layer."""

    def __init__(self,
                 output_dim,
                 args):
        super(HighwayLayer, self).__init__()
        self.activation = activations.get('tanh')
        self.kernel_initializer = initializers.GlorotUniform()
        self.dropout_rate = args.dropout_rate
        self.output_dim = output_dim
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=[input_shape[-1], self.output_dim],
                                      initializer=self.kernel_initializer,
                                      dtype='float32',
                                      trainable=True)

    def call(self, inputs, training=True):
        inputs = tf.reshape(inputs, (2, inputs.shape[1], inputs.shape[-1]))
        input1 = inputs[0]
        input2 = inputs[1]
        input1 = tf.keras.layers.BatchNormalization()(input1)
        input2 = tf.keras.layers.BatchNormalization()(input2)
        gate = tf.matmul(input1, self.kernel)
        gate = tf.keras.activations.tanh(gate)
        if training and self.dropout_rate > 0.0:
            gate = tf.nn.dropout(gate, self.dropout_rate)
        gate = tf.keras.activations.relu(gate)
        output = tf.add(tf.multiply(input2, 1 - gate), tf.multiply(input1, gate))
        return self.activation(output)


def highway(layer1, layer2):
    kernel_gate = glorot([self.dim, self.dim])
    bias_gate = zeros([self.dim])
    transform_gate = tf.matmul(layer1, kernel_gate) + bias_gate
    transform_gate = tf.nn.sigmoid(transform_gate)
    carry_gate = 1.0 - transform_gate
    return transform_gate * layer2 + carry_gate * layer1


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)