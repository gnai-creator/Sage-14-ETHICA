# SAGE-14: ETHICA — The Value Aligner
# Codinome: The Agent That Judges
# Author: Felipe Maya Muniz

import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, GRUCell, GlobalAveragePooling1D, GlobalMaxPooling1D

class ValueSystem(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.value_vector = tf.Variable(tf.random.normal([1, dim]), trainable=False)
        self.internal_ethics = Dense(dim, activation='tanh')
        self.alignment_gate = Dense(1, activation='sigmoid')

    def call(self, x):
        projection = self.internal_ethics(x)
        gate = self.alignment_gate(x)
        self.value_vector.assign(0.9 * self.value_vector + 0.1 * projection)
        ethical_aligned = x * (1 - gate) + self.value_vector * gate
        return ethical_aligned, gate


class EthicalConflict(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, action, value):
        return tf.reduce_mean(tf.abs(action - value))  # quanto maior, maior conflito moral


class ReflectiveMoralAgent(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.cell = GRUCell(dim)
        self.state = tf.Variable(tf.zeros([1, dim]), trainable=False)
        self.reflect = Dense(dim, activation='relu')

    def call(self, x):
        x = self.reflect(tf.reduce_mean(x, axis=1, keepdims=True))
        out, state = self.cell(x, [self.state])
        self.state.assign(state[0])
        return out


class Sage14Ethica(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = Dense(hidden_dim, activation='relu')
        self.attn = MultiHeadAttention(num_heads=6, key_dim=hidden_dim // 6)
        self.norm = LayerNormalization()

        self.agent = ReflectiveMoralAgent(hidden_dim)
        self.value_system = ValueSystem(hidden_dim)
        self.ethical_conflict = EthicalConflict()
        self.decoder = Dense(output_dim)
        self.pool_avg = GlobalAveragePooling1D()
        self.pool_max = GlobalMaxPooling1D()

    def call(self, x):
        x = self.encoder(x)
        x = self.attn(x, x, x)
        x = self.norm(x)

        agent_out = self.agent(tf.expand_dims(x, 1))
        aligned, gate = self.value_system(agent_out)
        conflict_score = self.ethical_conflict(agent_out, self.value_system.value_vector)

        pooled = self.pool_avg(tf.expand_dims(aligned, 1)) + self.pool_max(tf.expand_dims(aligned, 1))
        output = self.decoder(pooled + tf.expand_dims(conflict_score, -1))

        return output, conflict_score, gate, self.value_system.value_vector
