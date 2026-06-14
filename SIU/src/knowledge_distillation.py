import tensorflow as tf
from tensorflow import keras
import numpy as np

# Try to import the external distiller for the "Alternative" option
try:
    from external_distiller import ExternalDistiller
except ImportError:
    ExternalDistiller = None

def create_student_model(teacher_model, reduction_factor=2):
    """
    Creates a smaller student model.
    [MODIFIED FOR RESNET]: Added support for BatchNormalization and GlobalAvgPool.
    """
    student_config = teacher_model.get_config()
    # Handle input shape safely
    input_shape = teacher_model.input_shape[1:]
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    for layer in teacher_model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
            
        if isinstance(layer, keras.layers.Dense):
            new_units = max(16, int(layer.units / reduction_factor))
            if layer == teacher_model.layers[-1]:
                new_units = layer.units
            x = keras.layers.Dense(
                units=new_units,
                activation=layer.activation,
                name=f"student_{layer.name}"
            )(x)
            
        elif isinstance(layer, keras.layers.Flatten):
            x = keras.layers.Flatten()(x)
            
        elif isinstance(layer, keras.layers.Conv2D):
            new_filters = max(8, int(layer.filters / reduction_factor))
            x = keras.layers.Conv2D(
                filters=new_filters,
                kernel_size=layer.kernel_size,
                activation=layer.activation,
                padding=layer.padding,
                name=f"student_{layer.name}"
            )(x)
        
        elif isinstance(layer, keras.layers.MaxPooling2D):
            x = keras.layers.MaxPooling2D(pool_size=layer.pool_size)(x)
            
        elif isinstance(layer, keras.layers.Dropout):
            x = keras.layers.Dropout(layer.rate)(x)

        # --- RESNET SUPPORT ADDITIONS ---
        elif isinstance(layer, keras.layers.GlobalAveragePooling2D):
            x = keras.layers.GlobalAveragePooling2D()(x)
            
        elif isinstance(layer, keras.layers.BatchNormalization):
            # Student gets its own fresh BN layer
            x = keras.layers.BatchNormalization()(x)
            
        # Note: 'Add' layers (Skip Connections) cannot be automatically cloned 
        # in a linear loop like this. Complex ResNets usually require 
        # creating a fresh model definition rather than cloning.

    student_model = keras.Model(inputs=inputs, outputs=x, name="student_model")
    return student_model

class Distiller(keras.Model):
    """
    Your Original RKD Distiller.
    """
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False  # Freeze teacher

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3, beta=1.0):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        self.beta = beta # RKD Weight

    def _pdist(self, vectors):
        """
        Calculates pair-wise Euclidean distance matrix for a batch of vectors.
        """
        # vectors shape: (Batch, Features)
        # r_a: (Batch, 1) sum of squares
        r_a = tf.reduce_sum(tf.square(vectors), axis=1, keepdims=True)
        # r_b: (1, Batch) sum of squares
        r_b = tf.transpose(r_a)
        
        # dist^2 = a^2 + b^2 - 2ab
        dist_sq = r_a - 2 * tf.matmul(vectors, tf.transpose(vectors)) + r_b
        
        # Clip to avoid negative values due to float errors, then sqrt
        dist = tf.sqrt(tf.maximum(dist_sq, 1e-12))
        return dist

    def _rkd_loss(self, teacher_features, student_features):
        """
        Relation-Based Knowledge Distillation.
        """
        # --- FIXED FOR KERAS 3 SYMBOLIC TENSORS ---
        # We use tf.rank() or check .shape length safely
        if len(teacher_features.shape) > 2:
            teacher_features = tf.reshape(teacher_features, (tf.shape(teacher_features)[0], -1))
        if len(student_features.shape) > 2:
            student_features = tf.reshape(student_features, (tf.shape(student_features)[0], -1))

        # 1. Compute Pairwise Distances
        t_dist = self._pdist(teacher_features)
        s_dist = self._pdist(student_features)
        
        # 2. Normalize by the mean distance
        t_mean = tf.reduce_mean(t_dist) + 1e-9
        s_mean = tf.reduce_mean(s_dist) + 1e-9
        
        t_norm = t_dist / t_mean
        s_norm = s_dist / s_mean
        
        # 3. Huber Loss
        loss = tf.keras.losses.Huber()(t_norm, s_norm)
        return loss

    def train_step(self, data):
        x, y = data

        # Forward pass of Teacher (Inference only)
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of Student
            student_predictions = self.student(x, training=True)

            # 1. Standard Loss
            student_loss = self.student_loss_fn(y, student_predictions)

            # 2. Distillation Loss (KL Div)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            
            # 3. RKD Loss (Your addition)
            rkd_loss = self._rkd_loss(teacher_predictions, student_predictions)

            # Combine
            total_loss = ((1 - self.alpha) * student_loss) + \
                         (self.alpha * distillation_loss) + \
                         (self.beta * rkd_loss)

        # Backpropagation
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, student_predictions)
        
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "student_loss": student_loss,
            "distill_loss": distillation_loss,
            "rkd_loss": rkd_loss,
            "total_loss": total_loss
        })
        return results

    def test_step(self, data):
        x, y = data
        y_pred = self.student(x, training=False)
        
        teacher_predictions = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)
        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
            tf.nn.softmax(y_pred / self.temperature, axis=1),
        )
        
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss, "distill_loss": distillation_loss})
        return results

# ==============================================================================
# FACTORY: TO SWITCH BETWEEN YOURS AND EXTERNAL
# ==============================================================================
def get_distiller(distill_type, student, teacher, n_data=None):
    """
    Returns either your RKD Distiller or the Senior Student's External Distiller.
    """
    if distill_type == "ours":
        return Distiller(student, teacher)
        
    elif "external" in distill_type:
        if ExternalDistiller is None:
            raise ImportError("ExternalDistiller not found in external_distiller.py")
        
        # The senior code needs n_data for the Memory Bank
        distiller = ExternalDistiller(student, teacher)
        distiller.n_data = n_data 
        return distiller
        
    else:
        # Fallback to yours
        return Distiller(student, teacher)