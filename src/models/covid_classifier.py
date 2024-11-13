from typing import Tuple, Dict

import tensorflow as tf
from tensorflow.keras import layers, Model

class CovidClassifier:
    """COVID-19 X-ray classification model."""
    
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 num_classes: int,
                 model_type: str = 'vgg19'):
        """
        Initialize the classifier.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of classes to predict
            model_type: Base model architecture ('vgg19', 'resnet50', etc.)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        
    def build_model(self) -> Model:
        """Build and return the model architecture."""
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Base model
        if self.model_type == 'vgg19':
            base_model = tf.keras.applications.VGG19(
                include_top=False,
                weights='imagenet',
                input_tensor=inputs
            )
        elif self.model_type == 'resnet50':
            base_model = tf.keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_tensor=inputs
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        # Freeze base model
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def compile_model(self, model: Model, learning_rate: float = 0.001) -> Model:
        """Compile the model with appropriate optimizer and loss."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self, patience: int = 5) -> list:
        """Get training callbacks."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        return callbacks
