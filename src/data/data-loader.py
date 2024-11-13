import os
import pathlib
from typing import Tuple, List

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

class DataLoader:
    """Handles loading and preprocessing of the COVID-19 X-ray dataset."""
    
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the COVID-19 Radiography Dataset
            img_size: Target size for the images (height, width)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        
    def _get_file_paths(self) -> Tuple[List[str], List[str]]:
        """Get all image file paths and their corresponding labels."""
        filepaths = []
        labels = []
        
        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            images_dir = os.path.join(class_dir, 'images')
            if not os.path.exists(images_dir):
                continue
                
            for img_name in os.listdir(images_dir):
                img_path = os.path.join(images_dir, img_name)
                filepaths.append(img_path)
                labels.append(class_name)
                
        return filepaths, labels
    
    def create_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame containing file paths and labels."""
        filepaths, labels = self._get_file_paths()
        return pd.DataFrame({
            'filepath': filepaths,
            'label': labels
        })
    
    def split_data(self, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        df = self.create_dataframe()
        
        # First split: train and temporary
        train_df, temp_df = train_test_split(
            df, 
            train_size=train_size,
            stratify=df['label'],
            random_state=42
        )
        
        # Second split: validation and test
        valid_df, test_df = train_test_split(
            temp_df,
            train_size=0.5,
            stratify=temp_df['label'],
            random_state=42
        )
        
        return train_df, valid_df, test_df
    
    def create_dataset(self, 
                      dataframe: pd.DataFrame, 
                      batch_size: int = 32,
                      shuffle: bool = True,
                      augment: bool = False) -> tf.data.Dataset:
        """Create a TensorFlow dataset from a DataFrame."""
        
        def preprocess(filepath: str, label: str) -> Tuple[tf.Tensor, tf.Tensor]:
            # Read image
            img = tf.io.read_file(filepath)
            img = tf.image.decode_png(img, channels=3)
            
            # Resize
            img = tf.image.resize(img, self.img_size)
            
            # Normalize
            img = tf.cast(img, tf.float32) / 255.0
            
            return img, label
        
        def augmentation(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            # Random flip
            image = tf.image.random_flip_left_right(image)
            
            # Random brightness
            image = tf.image.random_brightness(image, 0.2)
            
            # Random contrast
            image = tf.image.random_contrast(image, 0.8, 1.2)
            
            return image, label
        
        # Create label encoder
        unique_labels = dataframe['label'].unique()
        label_to_index = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            dataframe['filepath'].values,
            dataframe['label'].map(label_to_index).values
        ))
        
        # Apply preprocessing
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Apply augmentation if requested
        if augment:
            dataset = dataset.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset, label_to_index
