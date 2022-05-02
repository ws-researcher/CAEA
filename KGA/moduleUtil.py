import tensorflow as tf
from tensorflow.keras import layers, initializers
from transformers import BertTokenizer, TFBertModel


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


class Bert_encoder:
    def __init__(self):
        bertPath = "/home/ws/Workplace/KGAlign/bert-base-uncased"
        # bertPath = "/home/Edwin-01/workplace/KGAlign/KGAlign2/bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(bertPath)
        self.model = TFBertModel.from_pretrained(bertPath, local_files_only=True)

    def encodeText(self, text: str):
        encoded_input = self.tokenizer(text, return_tensors='tf')
        last_hidden_state = self.model(**encoded_input)[0]

        out = tf.reduce_mean(last_hidden_state[0], 0).numpy()

        return out

    def encodeTextSet(self, texts: set):
        encodingDict = dict()
        for text in texts:
            encodingDict[text] = self.encodeText(text)

        return encodingDict


class AVlyaer(layers.Layer):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.kernel_initializer = None
        self.bias_initializer = None
        self.outputDim = None
        self.kernel = None
        self.bias = None
        self.MutilHeadAttentionLyaer = None
        self.d = None
        self.initLayer()


    def initLayer(self):
        self.kernel_initializer = initializers.GlorotUniform()
        self.bias_initializer = initializers.zeros()
        self.outputDim = self.args.input_dim
        self.d = tf.keras.layers.Dense(self.outputDim)

        # self.aggr = avg_aggr(self.outputDim)
        self.aggr = atte_aggr(self.outputDim)
        pass

    def call(self, inputs, *args, **kwargs):
        wh = self.d(inputs)
        AVh = self.aggr(wh)
        h = tf.keras.activations.tanh(AVh)
        return h


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
