import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

# ==============================================================================
# PORT FROM: memory.py
# ==============================================================================

class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    Faithful port of the student's implementation.
    """
    def __init__(self, probs):
        if tf.reduce_sum(probs) > 1:
            probs = probs / tf.reduce_sum(probs)
        
        K = len(probs)
        self.prob = np.zeros(K)
        self.alias = np.zeros(K, dtype=np.int32)
        
        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        
        # We process this on CPU using numpy as it's an init step
        probs_np = probs.numpy() if hasattr(probs, 'numpy') else np.array(probs)
        
        for kk, prob in enumerate(probs_np):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

        # Convert to TF constants for usage in graph
        self.prob = tf.constant(self.prob, dtype=tf.float32)
        self.alias = tf.constant(self.alias, dtype=tf.int32)

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = tf.shape(self.alias)[0]

        # Draw random indices
        kk = tf.random.uniform((N,), minval=0, maxval=tf.cast(K, tf.int32), dtype=tf.int32)
        
        prob = tf.gather(self.prob, kk)
        alias = tf.gather(self.alias, kk)
        
        # b is whether a random number is greater than q
        # torch.bernoulli(prob) -> tf.random.uniform < prob
        b = tf.cast(tf.random.uniform(tf.shape(prob)) < prob, tf.float32)
        
        oq = tf.cast(kk, tf.float32) * b
        oj = tf.cast(alias, tf.float32) * (1 - b)

        return tf.cast(oq + oj, tf.int64)


class ContrastMemory(keras.layers.Layer):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, **kwargs):
        super(ContrastMemory, self).__init__(**kwargs)
        self.nLem = outputSize
        
        # Replicating the "unnecessary" initialization of uniform probs
        self.unigrams = tf.ones(self.nLem) 
        self.multinomial = AliasMethod(self.unigrams)
        
        self.K = K
        self.T = T
        self.momentum = momentum
        
        # register_buffer equivalent in Keras (Non-trainable weights)
        stdv = 1. / math.sqrt(inputSize / 3)
        self.memory_v1 = self.add_weight(
            "memory_v1", shape=(outputSize, inputSize),
            initializer=tf.random_uniform_initializer(-stdv, stdv),
            trainable=False
        )
        self.memory_v2 = self.add_weight(
            "memory_v2", shape=(outputSize, inputSize),
            initializer=tf.random_uniform_initializer(-stdv, stdv),
            trainable=False
        )

        # Params buffer: [K, T, Z_v1, Z_v2, momentum]
        # We store Z_v1 and Z_v2 as state variables
        self.Z_v1 = self.add_weight("Z_v1", initializer=tf.constant_initializer(-1), trainable=False)
        self.Z_v2 = self.add_weight("Z_v2", initializer=tf.constant_initializer(-1), trainable=False)

    def call(self, v1, v2, y, idx=None):
        # inputs: v1 (Batch, Dim), v2 (Batch, Dim), y (Batch, Labels/Indices)
        batchSize = tf.shape(v1)[0]
        outputSize = self.memory_v1.shape[0]
        inputSize = self.memory_v1.shape[1]

        # original score computation
        if idx is None:
            # Replicating logic: idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx = self.multinomial.draw(batchSize * (self.K + 1))
            idx = tf.reshape(idx, (batchSize, -1))
            
            # idx.select(1, 0).copy_(y.data) -> Replace first col with y
            # We construct the tensor [y, idx[:, 1:]]
            y = tf.cast(y, tf.int64)
            idx_slice = idx[:, 1:]
            y_expanded = tf.expand_dims(y, 1)
            idx = tf.concat([y_expanded, idx_slice], axis=1) # (Batch, K+1)

        idx = tf.cast(idx, tf.int32)
        y = tf.cast(y, tf.int32)

        # sample v1
        # weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = tf.gather(self.memory_v1, tf.reshape(idx, [-1]))
        weight_v1 = tf.reshape(weight_v1, (batchSize, self.K + 1, inputSize))
        
        # out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = tf.matmul(weight_v1, tf.expand_dims(v2, -1))
        out_v2 = tf.exp(tf.squeeze(out_v2, -1) / self.T)

        # sample v2
        weight_v2 = tf.gather(self.memory_v2, tf.reshape(idx, [-1]))
        weight_v2 = tf.reshape(weight_v2, (batchSize, self.K + 1, inputSize))
        
        out_v1 = tf.matmul(weight_v2, tf.expand_dims(v1, -1))
        out_v1 = tf.exp(tf.squeeze(out_v1, -1) / self.T)

        # set Z if haven't been set yet
        # Note: In TF Graph execution, we use tf.cond/custom logic. 
        # For faithfulness, we update Z using assign.
        
        # Logic: if Z_v1 < 0: set it.
        def update_z1():
            new_z = tf.reduce_mean(out_v1) * tf.cast(outputSize, tf.float32)
            self.Z_v1.assign(new_z)
            return new_z
        
        curr_z1 = tf.cond(self.Z_v1 < 0, update_z1, lambda: self.Z_v1)
        
        def update_z2():
            new_z = tf.reduce_mean(out_v2) * tf.cast(outputSize, tf.float32)
            self.Z_v2.assign(new_z)
            return new_z
            
        curr_z2 = tf.cond(self.Z_v2 < 0, update_z2, lambda: self.Z_v2)

        # compute out_v1, out_v2
        out_v1 = out_v1 / curr_z1
        out_v2 = out_v2 / curr_z2

        # update memory
        # l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
        l_pos = tf.gather(self.memory_v1, y)
        # l_pos.mul_(momentum).add_(torch.mul(v1, 1 - momentum))
        l_pos_new = l_pos * self.momentum + v1 * (1 - self.momentum)
        # l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        l_norm = tf.norm(l_pos_new, axis=1, keepdims=True)
        # updated_v1 = l_pos.div(l_norm)
        updated_v1 = l_pos_new / (l_norm + 1e-9)
        # self.memory_v1.index_copy_(0, y, updated_v1)
        self.memory_v1.scatter_nd_update(tf.expand_dims(y, 1), updated_v1)

        # Same for v2
        ab_pos = tf.gather(self.memory_v2, y)
        ab_pos_new = ab_pos * self.momentum + v2 * (1 - self.momentum)
        ab_norm = tf.norm(ab_pos_new, axis=1, keepdims=True)
        updated_v2 = ab_pos_new / (ab_norm + 1e-9)
        self.memory_v2.scatter_nd_update(tf.expand_dims(y, 1), updated_v2)

        return out_v1, out_v2


# ==============================================================================
# PORT FROM: criterion.py
# ==============================================================================

class ContrastLoss(keras.layers.Layer):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data, **kwargs):
        super(ContrastLoss, self).__init__(**kwargs)
        self.n_data = n_data
        self.eps = 1e-7

    def call(self, x):
        bsz = tf.cast(tf.shape(x)[0], tf.float32)
        m = tf.cast(tf.shape(x)[1] - 1, tf.float32)

        # noise distribution
        Pn = 1.0 / float(self.n_data)

        # loss for positive pair
        # P_pos = x.select(1, 0)
        P_pos = x[:, 0]
        # log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()
        log_D1 = tf.math.log(P_pos / (P_pos + m * Pn + self.eps))

        # loss for K negative pair
        # P_neg = x.narrow(1, 1, m)
        P_neg = x[:, 1:]
        # log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()
        # Note: Cloning and filling with m*Pn is the noise term in NCE
        # D0 = (m*Pn) / (P_neg + m*Pn)
        log_D0 = tf.math.log((m * Pn) / (P_neg + m * Pn + self.eps))

        # loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz
        loss = - (tf.reduce_sum(log_D1) + tf.reduce_sum(log_D0)) / bsz
        
        return loss

class Embed(keras.layers.Layer):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128, **kwargs):
        super(Embed, self).__init__(**kwargs)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = keras.layers.Dense(dim_out, input_shape=(dim_in,))
        
    def call(self, x):
        # x = x.view(x.shape[0], -1)
        if len(x.shape) > 2:
            x = keras.layers.Flatten()(x)
        x = self.linear(x)
        # x = self.l2norm(x)
        x = tf.math.l2_normalize(x, axis=1)
        return x

class CRDLoss(keras.layers.Layer):
    """CRD Loss function"""
    def __init__(self, s_dim, t_dim, feat_dim, n_data, nce_k, nce_t, nce_m, **kwargs):
        super(CRDLoss, self).__init__(**kwargs)
        self.embed_s = Embed(s_dim, feat_dim)
        self.embed_t = Embed(t_dim, feat_dim)
        self.contrast = ContrastMemory(feat_dim, n_data, nce_k, nce_t, nce_m)
        self.criterion_t = ContrastLoss(n_data)
        self.criterion_s = ContrastLoss(n_data)

    def call(self, f_s, f_t, idx, contrast_idx=None):
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


# ==============================================================================
# PORT FROM: AT.py
# ==============================================================================

class Attention(keras.layers.Layer):
    """Paying More Attention to Attention"""
    def __init__(self, p=2, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.p = p

    def call(self, g_s, g_t):
        # return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
        # TF Keras doesn't like returning lists of scalars usually, but we sum them later
        loss = 0.0
        for f_s, f_t in zip(g_s, g_t):
            loss += self.at_loss(f_s, f_t)
        return loss

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[1], f_t.shape[1] # TF is (B, H, W, C) usually, verify if channels last
        
        # Student Code assumes (B, C, H, W) for PyTorch.
        # If input is (B, H, W, C), the heights are at index 1 and 2.
        # Assuming TF default "channels_last"
        
        if s_H > t_H:
            # f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
            # TF equivalent: avg_pool or resize
            diff = s_H - t_H
            f_s = tf.nn.avg_pool2d(f_s, ksize=diff+1, strides=1, padding='VALID')
        elif s_H < t_H:
            diff = t_H - s_H
            f_t = tf.nn.avg_pool2d(f_t, ksize=diff+1, strides=1, padding='VALID')
        else:
            pass
        
        return tf.reduce_mean(tf.square(self.at(f_s) - self.at(f_t)))

    def at(self, f):
        # return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))
        # PyTorch mean(1) is mean over Channels (if NCHW)
        # TF Channels are at -1 (if NHWC). 
        
        x = tf.pow(f, self.p)
        x = tf.reduce_mean(x, axis=-1) # Mean over channels
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        return tf.math.l2_normalize(x, axis=1)


# ==============================================================================
# PORT FROM: FitNet.py
# ==============================================================================

class HintLoss(keras.layers.Layer):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self, **kwargs):
        super(HintLoss, self).__init__(**kwargs)
        # self.crit = nn.MSELoss()

    def call(self, g_s, g_t):
        # loss = [self.crit(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
        loss = 0.0
        for f_s, f_t in zip(g_s, g_t):
            loss += tf.reduce_mean(tf.square(f_s - f_t))
        return loss


# ==============================================================================
# PORT FROM: VID.py
# ==============================================================================

class VIDLoss(keras.layers.Layer):
    """Variational Information Distillation"""
    def __init__(self, num_input_channels, num_mid_channel, num_target_channels, 
                 init_pred_var=5.0, eps=1e-5, **kwargs):
        super(VIDLoss, self).__init__(**kwargs)
        
        # def conv1x1 ...
        self.regressor = keras.Sequential([
            keras.layers.Conv2D(num_mid_channel, 1, strides=1, padding='valid', use_bias=False),
            keras.layers.ReLU(),
            keras.layers.Conv2D(num_mid_channel, 1, strides=1, padding='valid', use_bias=False),
            keras.layers.ReLU(),
            keras.layers.Conv2D(num_target_channels, 1, strides=1, padding='valid', use_bias=False),
        ])
        
        # self.log_scale = torch.nn.Parameter(...)
        init_val = np.log(np.exp(init_pred_var - eps) - 1.0)
        self.log_scale = self.add_weight(
            "log_scale", 
            shape=(num_target_channels,),
            initializer=tf.constant_initializer(init_val),
            trainable=True
        )
        self.eps = eps

    def call(self, input, target):
        # pool for dimension match
        s_H, t_H = input.shape[1], target.shape[1]
        
        if s_H > t_H:
             diff = s_H - t_H
             input = tf.nn.avg_pool2d(input, ksize=diff+1, strides=1, padding='VALID')
        elif s_H < t_H:
             diff = t_H - s_H
             target = tf.nn.avg_pool2d(target, ksize=diff+1, strides=1, padding='VALID')
        
        pred_mean = self.regressor(input)
        
        # pred_var = torch.log(1.0+torch.exp(self.log_scale))+self.eps
        pred_var = tf.math.log(1.0 + tf.exp(self.log_scale)) + self.eps
        # pred_var = pred_var.view(1, -1, 1, 1) -> (1, 1, 1, C) for NHWC
        pred_var = tf.reshape(pred_var, (1, 1, 1, -1))
        
        neg_log_prob = 0.5 * (
            tf.square(pred_mean - target) / pred_var + tf.math.log(pred_var)
        )
        loss = tf.reduce_mean(neg_log_prob)
        return loss


# ==============================================================================
# PORT FROM: WSL.py
# ==============================================================================

class WSLLoss(keras.layers.Layer):
    def __init__(self, T, **kwargs):
        super(WSLLoss, self).__init__(**kwargs)
        self.T = T

    def call(self, g_s, g_t, target):
        s_input_for_softmax = g_s / self.T
        t_input_for_softmax = g_t / self.T

        t_soft_label = tf.nn.softmax(t_input_for_softmax)

        # softmax_loss = - torch.sum(t_soft_label * self.logsoftmax(s_input_for_softmax), 1, keepdim=True)
        s_log_softmax = tf.nn.log_softmax(s_input_for_softmax)
        softmax_loss = -tf.reduce_sum(t_soft_label * s_log_softmax, axis=1, keepdims=True)

        fc_s_auto = tf.stop_gradient(g_s)
        fc_t_auto = tf.stop_gradient(g_t)
        log_softmax_s = tf.nn.log_softmax(fc_s_auto)
        log_softmax_t = tf.nn.log_softmax(fc_t_auto)
        
        # one_hot_label = F.one_hot(target, num_classes=100).float()
        # Assumes target is already one-hot or int. If one-hot:
        if len(target.shape) > 1 and target.shape[1] > 1:
            one_hot_label = tf.cast(target, tf.float32)
        else:
            # Assuming 100 classes as per student code, or infer from logits
            one_hot_label = tf.one_hot(tf.cast(target, tf.int32), depth=g_s.shape[-1])

        softmax_loss_s = -tf.reduce_sum(one_hot_label * log_softmax_s, axis=1, keepdims=True)
        softmax_loss_t = -tf.reduce_sum(one_hot_label * log_softmax_t, axis=1, keepdims=True)

        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = tf.zeros(1)
        focal_weight = tf.maximum(focal_weight, ratio_lower)
        focal_weight = 1 - tf.exp(-focal_weight)
        
        softmax_loss = focal_weight * softmax_loss
        soft_loss = (self.T ** 2) * tf.reduce_mean(softmax_loss)

        return soft_loss


# ==============================================================================
# PORT FROM: KD.py
# ==============================================================================

class DistillKLD(keras.layers.Layer):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T, **kwargs):
        super(DistillKLD, self).__init__(**kwargs)
        self.T = T

    def call(self, y_s, y_t):
        p_s = tf.nn.log_softmax(y_s / self.T, axis=1)
        p_t = tf.nn.softmax(y_t / self.T, axis=1)
        # loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        # TF KLD(target, pred)
        loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)(p_t, p_s)
        loss = loss * (self.T ** 2) / tf.cast(tf.shape(y_s)[0], tf.float32)
        return loss


# ==============================================================================
# PORT FROM: util.py
# ==============================================================================

class ConvReg(keras.layers.Layer):
    """Convolutional regression for FitNet"""
    def __init__(self, s_shape, t_shape, use_relu=True, **kwargs):
        super(ConvReg, self).__init__(**kwargs)
        self.use_relu = use_relu
        # s_shape: (N, H, W, C) in TF
        s_H, s_W, s_C = s_shape[1], s_shape[2], s_shape[3]
        t_H, t_W, t_C = t_shape[0], t_shape[1], t_shape[2]
        
        if s_H == 2 * t_H:
             self.conv = keras.layers.Conv2D(t_C, 3, strides=2, padding='same')
        elif s_H * 2 == t_H:
             self.conv = keras.layers.Conv2DTranspose(t_C, 4, strides=2, padding='same')
        elif s_H >= t_H:
             self.conv = keras.layers.Conv2D(t_C, (1+s_H-t_H, 1+s_W-t_W), padding='valid')
        else:
             # raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
             # Fallback 1x1
             self.conv = keras.layers.Conv2D(t_C, 1, padding='same')
             
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


# ==============================================================================
# THE MAIN WRAPPER (Faithful to train_student.py logic)
# ==============================================================================

class ExternalDistiller(keras.Model):
    """
    Acts as the 'train_kd' function from train_student.py, wrapping the student model
    and the distillation modules.
    """
    def __init__(self, student, teacher, mode='hint', n_data=50000, feat_dim=128):
        super(ExternalDistiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.mode = mode
        self.n_data = n_data
        self.feat_dim = feat_dim
        
        # Modules list (replicating module_list from train_student.py)
        self.regressors = []
        self.crd_loss_module = None
        self.vid_loss_modules = []
        
        # We will initialize specific losses in compile or build based on shapes,
        # mimicking the setup in train_student.py which happens before the loop.
        # But since we need shapes, we might do it lazily or assume standard shapes.
        
        # Standard losses
        self.criterion_cls = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.criterion_div = DistillKLD(T=4.0) # Default T
        self.criterion_kd_module = None # Placeholder for hint/attention/etc

    def compile(self, optimizer, metrics, student_loss_fn, 
                alpha=0.9, gamma=1.0, beta=0.0, temperature=4.0):
        super(ExternalDistiller, self).compile(optimizer=optimizer, metrics=metrics)
        # Update params
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.temperature = temperature
        
        # Update T for DistillKLD
        self.criterion_div.T = temperature

        # Instantiate specific criterion based on mode (like train_student.py lines 480+)
        if self.mode == 'kd':
            self.criterion_kd_module = DistillKLD(temperature)
        elif self.mode == 'hint':
            self.criterion_kd_module = HintLoss()
        elif self.mode == 'attention':
            self.criterion_kd_module = Attention()
        elif self.mode == 'wsl':
            self.criterion_div = WSLLoss(temperature) # Replaces Div
            self.criterion_kd_module = Attention() # WSL_att uses Attention often
        elif 'crd' in self.mode:
            # Requires dimensions. We will initialize in first train_step if not set.
            pass
        elif self.mode == 'vid':
            pass # Init in train step due to shape dependency

    def train_step(self, data):
        # Unpack data
        if len(data) == 3: x, y, indices = data
        else: x, y = data; indices = None

        # 1. Teacher Forward
        t_out = self.teacher(x, training=False)
        if isinstance(t_out, (tuple, list)):
            t_logits, t_feats = t_out[0], t_out[1:]
        else:
            t_logits, t_feats = t_out, []

        with tf.GradientTape() as tape:
            # 2. Student Forward
            s_out = self.student(x, training=True)
            if isinstance(s_out, (tuple, list)):
                s_logits, s_feats = s_out[0], s_out[1:]
            else:
                s_logits, s_feats = s_out, []

            # 3. Calculate Losses
            
            # Classification
            loss_cls = self.criterion_cls(y, s_logits)

            # Divergence (KL)
            if 'wsl' in self.mode:
                loss_div = self.criterion_div(s_logits, t_logits, y)
            else:
                loss_div = self.criterion_div(s_logits, t_logits)

            # Other Distillation Losses
            loss_other = 0.0
            
            if self.mode == 'hint':
                # Lazy init regressors if needed
                if not self.regressors:
                    for i, (s, t) in enumerate(zip(s_feats, t_feats)):
                        reg = ConvReg(s.shape, t.shape[1:]) # t_shape passed as H,W,C
                        self.regressors.append(reg)
                
                # Apply regressors
                s_regressed = [reg(f) for reg, f in zip(self.regressors, s_feats)]
                loss_other = self.criterion_kd_module(s_regressed, t_feats)

            elif self.mode == 'attention':
                loss_other = self.criterion_kd_module(s_feats, t_feats)

            elif self.mode == 'vid':
                if not self.vid_loss_modules:
                     for s, t in zip(s_feats, t_feats):
                         self.vid_loss_modules.append(
                             VIDLoss(s.shape[-1], s.shape[-1], t.shape[-1])
                         )
                
                loss_other = 0.0
                for mod, s, t in zip(self.vid_loss_modules, s_feats, t_feats):
                    loss_other += mod(s, t)

            elif 'crd' in self.mode:
                if self.crd_loss_module is None:
                    # s_dim, t_dim, feat_dim, n_data, nce_k, nce_t, nce_m
                    self.crd_loss_module = CRDLoss(
                        s_feats[-1].shape[-1], t_feats[-1].shape[-1], 
                        self.feat_dim, self.n_data, 16384, 0.07, 0.5
                    )
                
                if indices is None:
                     # Fake indices if not provided (Safety fallback)
                     indices = tf.random.uniform((tf.shape(x)[0],), maxval=self.n_data, dtype=tf.int32)
                
                loss_other = self.crd_loss_module(s_feats[-1], t_feats[-1], indices)

            # Weighted Sum (Exact formula from train_student.py line 588)
            # loss = loss_kd * alpha + loss_cls * gamma + loss_other * beta
            # Note: In student code, 'loss_kd' is the KL div.
            total_loss = (loss_div * self.alpha) + (loss_cls * self.gamma) + (loss_other * self.beta)

        # 4. Backprop
        trainable_vars = self.student.trainable_variables
        
        # Add aux variables from modules
        if self.regressors:
            for reg in self.regressors: trainable_vars += reg.trainable_variables
        if self.vid_loss_modules:
            for mod in self.vid_loss_modules: trainable_vars += mod.trainable_variables
        if self.crd_loss_module:
            trainable_vars += self.crd_loss_module.trainable_variables
            
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Metrics
        self.compiled_metrics.update_state(y, s_logits)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": total_loss, "kd_loss": loss_div, "feat_loss": loss_other})
        return results