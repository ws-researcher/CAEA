from tensorflow.keras import layers, initializers, regularizers
import tensorflow as tf

from KGAlignModel.BaseLayer import GraphAttentionLayer, atte_aggr, avg_aggr, GCNLayer

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
        self.outputDim = self.args.AClyaerDim
        self.d = tf.keras.layers.Dense(self.outputDim)

        # self.aggr = avg_aggr(self.outputDim)
        self.aggr = atte_aggr(self.outputDim)
        pass

    def call(self, inputs, *args, **kwargs):
        wh = self.d(inputs)
        AVh = self.aggr(wh)
        h = tf.keras.activations.tanh(AVh)
        return h

class FHopGATlayer(layers.Layer):

    def __init__(self, arg, ADJ: dict, entities_num, **kwargs):
        super().__init__(**kwargs)

        self.args = arg
        self.ADJ = ADJ
        self.entities_num = entities_num
        self.GraphAttentionLayers_s = list()
        self.GraphAttentionLayers_p = list()
        self.GraphAttentionLayers_c = list()
        self.GraphAttentionLayers_a = list()
        self.HighwayLayer_s = None
        self.HighwayLayer_p = None
        self.HighwayLayer_c = None

        self.initLyaer()
        pass

    def initLyaer(self):
        for dim in self.args.GATLayerDim:
            GraphAttentionLayer_s = GraphAttentionLayer(dim, self.ADJ["adjs"], self.entities_num, self.args.dropout_rate)
            self.GraphAttentionLayers_s.append(GraphAttentionLayer_s)

        for dim in self.args.GATLayerDim:
            GraphAttentionLayer_p = GraphAttentionLayer(dim, self.ADJ["adjp"], self.entities_num, self.args.dropout_rate)
            self.GraphAttentionLayers_p.append(GraphAttentionLayer_p)

        for dim in self.args.GATLayerDim:
            GraphAttentionLayer_c = GraphAttentionLayer(dim, self.ADJ["adjc"], self.entities_num, self.args.dropout_rate)
            self.GraphAttentionLayers_c.append(GraphAttentionLayer_c)

        for dim in self.args.GATLayerDim:
            GraphAttentionLayer_a = GraphAttentionLayer(dim, self.ADJ["adj"], self.entities_num, self.args.dropout_rate)
            self.GraphAttentionLayers_a.append(GraphAttentionLayer_a)
        pass

    def call(self, inputs, training=False):
        outputList = list()
        if self.args.A:
            self.args.S = False
            self.args.P = False
            self.args.C = False
        if self.args.AL:
            inputs = tf.expand_dims(inputs, axis=1)
            outputList.append(inputs)

        if self.args.S:
            attentionLayer_output_s = inputs
            attentionLayer_outputs_s = list()
            for GraphAttentionLayer_s in self.GraphAttentionLayers_s:
                attentionLayer_output_s = GraphAttentionLayer_s(attentionLayer_output_s, training)

                attentionLayer_output_s = tf.expand_dims(attentionLayer_output_s, axis=1)

                attentionLayer_outputs_s.append(attentionLayer_output_s)
            output_s = tf.concat(attentionLayer_outputs_s, axis=1)
            outputList.append(output_s)

        if self.args.P:
            attentionLayer_output_p = inputs
            attentionLayer_outputs_p = list()
            for GraphAttentionLayer_p in self.GraphAttentionLayers_p:
                attentionLayer_output_p = GraphAttentionLayer_p(attentionLayer_output_p, training)

                attentionLayer_output_p = tf.expand_dims(attentionLayer_output_p, axis=1)

                attentionLayer_outputs_p.append(attentionLayer_output_p)
            output_p = tf.concat(attentionLayer_outputs_p, axis=1)
            outputList.append(output_p)

        if self.args.C:
            attentionLayer_output_c = inputs
            attentionLayer_outputs_c = list()
            for GraphAttentionLayer_c in self.GraphAttentionLayers_c:
                attentionLayer_output_c = GraphAttentionLayer_c(attentionLayer_output_c, training)

                attentionLayer_output_c = tf.expand_dims(attentionLayer_output_c, axis=1)

                attentionLayer_outputs_c.append(attentionLayer_output_c)
            output_c = tf.concat(attentionLayer_outputs_c, axis=1)
            outputList.append(output_c)

        if self.args.A:
            attentionLayer_output_a = inputs
            attentionLayer_outputs_a = list()
            for GraphAttentionLayer_a in self.GraphAttentionLayers_a:
                attentionLayer_output_a = GraphAttentionLayer_a(attentionLayer_output_a, training)

                attentionLayer_output_a = tf.expand_dims(attentionLayer_output_a, axis=1)

                attentionLayer_outputs_a.append(attentionLayer_output_a)
            output_a = tf.concat(attentionLayer_outputs_a, axis=1)
            outputList.append(output_a)

        out = tf.concat(outputList, axis=1)

        return out
        pass

class FHopGCNlayer(layers.Layer):

    def __init__(self, arg, ADJ: dict, entities_num, **kwargs):
        super().__init__(**kwargs)

        self.args = arg
        self.ADJ = ADJ
        self.entities_num = entities_num
        self.GraphAttentionLayers_s = list()
        self.GraphAttentionLayers_p = list()
        self.GraphAttentionLayers_c = list()
        self.GraphAttentionLayers_a = list()
        self.HighwayLayer_s = None
        self.HighwayLayer_p = None
        self.HighwayLayer_c = None
        self.initLyaer()
        pass

    def initLyaer(self):
        for dim in self.args.GATLayerDim:
            GraphAttentionLayer_s = GCNLayer(dim, self.ADJ["adjs"], self.entities_num, self.args.dropout_rate)
            self.GraphAttentionLayers_s.append(GraphAttentionLayer_s)

        for dim in self.args.GATLayerDim:
            GraphAttentionLayer_p = GCNLayer(dim, self.ADJ["adjc"], self.entities_num, self.args.dropout_rate)
            self.GraphAttentionLayers_p.append(GraphAttentionLayer_p)

        for dim in self.args.GATLayerDim:
            GraphAttentionLayer_c = GCNLayer(dim, self.ADJ["adjp"], self.entities_num, self.args.dropout_rate)
            self.GraphAttentionLayers_c.append(GraphAttentionLayer_c)

        for dim in self.args.GATLayerDim:
            GraphAttentionLayer_a = GCNLayer(dim, self.ADJ["adj"], self.entities_num, self.args.dropout_rate)
            self.GraphAttentionLayers_a.append(GraphAttentionLayer_a)
        pass

    def call(self, inputs, training=False):
        outputList = list()
        if self.args.A:
            self.args.S = False
            self.args.P = False
            self.args.C = False

        if self.args.AL:
            outputList.append(inputs)

        if self.args.S:
            attentionLayer_output_s = inputs
            attentionLayer_outputs_s = list()
            for GraphAttentionLayer_s in self.GraphAttentionLayers_s:
                attentionLayer_output_s = GraphAttentionLayer_s(attentionLayer_output_s, training)
                attentionLayer_outputs_s.append(attentionLayer_output_s)
            output_s = tf.concat(attentionLayer_outputs_s, axis=0)
            outputList.append(output_s)

        if self.args.P:
            attentionLayer_output_p = inputs
            attentionLayer_outputs_p = list()
            for GraphAttentionLayer_p in self.GraphAttentionLayers_p:
                attentionLayer_output_p = GraphAttentionLayer_p(attentionLayer_output_p, training)
                attentionLayer_outputs_p.append(attentionLayer_output_p)
            output_p = tf.concat(attentionLayer_outputs_p, axis=0)
            outputList.append(output_p)


        if self.args.C:
            attentionLayer_output_c = inputs
            attentionLayer_outputs_c = list()
            for GraphAttentionLayer_c in self.GraphAttentionLayers_c:
                attentionLayer_output_c = GraphAttentionLayer_c(attentionLayer_output_c, training)
                attentionLayer_outputs_c.append(attentionLayer_output_c)
            output_c = tf.concat(attentionLayer_outputs_c, axis=0)
            outputList.append(output_c)

        if self.args.A:
            attentionLayer_output_a = inputs
            attentionLayer_outputs_a = list()
            for GraphAttentionLayer_a in self.GraphAttentionLayers_a:
                attentionLayer_output_a = GraphAttentionLayer_a(attentionLayer_output_a, training)
                attentionLayer_outputs_a.append(attentionLayer_output_a)
            output_a = tf.concat(attentionLayer_outputs_a, axis=0)
            outputList.append(output_a)

        return tf.concat(outputList, axis=0)
        pass
