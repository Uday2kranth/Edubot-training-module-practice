# Deep Learning Assignment: CNN Image Classification & RNN Text Classification

## Overview
This project implements and compares deep learning models for two tasks:
- **Part A**: Image classification using CNNs on MNIST dataset
- **Part B**: Text classification using RNNs and transformer models

## Model Architectures

### Part A: Convolutional Neural Network (CNN) for Image Classification

#### Custom CNN Architecture
```
Input (28×28×1) → Conv2D(32, 3×3, ReLU) → MaxPool(2×2) → 
Conv2D(64, 3×3, ReLU) → MaxPool(2×2) → Conv2D(64, 3×3, ReLU) → 
Flatten → Dense(64, ReLU) → Dense(10, Softmax)
```

**Architecture Diagram:**
```
[28×28×1 Input] 
       ↓
[Conv2D: 32 filters, 3×3, ReLU]
       ↓
[MaxPooling: 2×2]
       ↓
[Conv2D: 64 filters, 3×3, ReLU]
       ↓
[MaxPooling: 2×2]
       ↓
[Conv2D: 64 filters, 3×3, ReLU]
       ↓
[Flatten Layer]
       ↓
[Dense: 64 units, ReLU]
       ↓
[Dense: 10 units, Softmax]
       ↓
[Output: 10 classes]
```

**Model Parameters:** 93,322 total parameters (364.54 KB)

#### Transfer Learning with MobileNetV2
```
MobileNetV2 Base (Frozen) → GlobalAveragePooling2D → 
Dense(128, ReLU) → Dense(10, Softmax)
```

**Architecture Diagram:**
```
[224×224×3 Input] 
       ↓
[MobileNetV2 Base - FROZEN]
[2,257,984 parameters]
       ↓
[GlobalAveragePooling2D]
       ↓
[Dense: 128 units, ReLU]
       ↓
[Dense: 10 units, Softmax]
       ↓
[Output: 10 classes]
```

**Model Parameters:** 2,423,242 total (2.26M frozen, 165K trainable)

### Part B: Recurrent Neural Network (RNN) for Text Classification

#### Custom RNN Architecture
```
Input (Sequences) → Embedding(1000→64) → LSTM(64, return_sequences=True) → 
LSTM(32) → Dense(16, ReLU) → Dense(1, Sigmoid)
```

**Architecture Diagram:**
```
[Text Input: Max length 10]
       ↓
[Embedding: 1000 vocab → 64 dims]
       ↓
[LSTM: 64 units, return_sequences=True]
       ↓
[LSTM: 32 units]
       ↓
[Dense: 16 units, ReLU]
       ↓
[Dense: 1 unit, Sigmoid]
       ↓
[Output: Binary Classification]
```

#### Pre-trained Transformer (DistilBERT)
```
Input Text → DistilBERT Tokenization → DistilBERT Model → Classification Head
```

**Architecture Diagram:**
```
[Raw Text Input]
       ↓
[DistilBERT Tokenizer]
       ↓
[DistilBERT Base Model]
[Pre-trained on large corpus]
       ↓
[Classification Head]
       ↓
[Sentiment Output: POSITIVE/NEGATIVE]
```

## Results Comparison

### Part A: Image Classification Results

**Custom CNN Performance:**
- **Test Accuracy**: 98.48%
- **Training Time**: ~33 seconds (5 epochs)
- **Parameters**: 93,322 total
- **Comments**: Excellent performance on MNIST dataset

**Transfer Learning Performance:**
- **Test Accuracy**: 89.00%
- **Training Time**: ~71 seconds (3 epochs)
- **Parameters**: 2.4M total (2.26M frozen, 165K trainable)
- **Comments**: Used subset (1000 samples) due to memory constraints

**Key Observations:**
- Custom CNN achieved superior performance (98.48% vs 89.00%)
- Transfer learning showed lower accuracy due to limited training data subset
- Custom CNN was more efficient with fewer parameters
- Transfer learning required significantly more memory and computation

### Part B: Text Classification Results

**Custom RNN Performance:**
- **Test Accuracy**: 33.33%
- **Confidence**: Low
- **Training Data**: 7 samples
- **Comments**: Limited by small dataset

**DistilBERT Performance:**
- **Test Accuracy**: >99.9%
- **Confidence**: Very High
- **Training Data**: Pre-trained on large corpus
- **Comments**: Excellent performance

**Sample Predictions:**

**Text: "This is a great product, I love it!"**
- **RNN Prediction**: 0.51 (Positive)
- **DistilBERT**: POSITIVE (99.99% confidence)

**Text: "Terrible quality, very disappointed."**
- **RNN Prediction**: 0.44 (Negative)
- **DistilBERT**: NEGATIVE (99.98% confidence)

**Text: "Average performance, nothing special."**
- **RNN Prediction**: 0.46 (Negative)
- **DistilBERT**: NEGATIVE (99.93% confidence)

## Analysis and Limitations

### Custom Models Limitations

#### CNN Limitations:
1. **Dataset Dependency**: Performance heavily relies on training data quality and quantity
2. **Feature Learning**: Must learn features from scratch without prior knowledge
3. **Generalization**: May not generalize well to different image domains
4. **Training Time**: Requires extensive training for complex patterns

#### RNN Limitations:
1. **Small Dataset**: Only 10 samples total (7 training, 3 testing)
2. **Vocabulary Limitation**: Limited to 41 unique words
3. **Sequential Dependencies**: Struggles with long-term dependencies
4. **Overfitting**: High risk with small datasets

### Transfer Learning Benefits

#### Image Classification:
1. **Pre-learned Features**: MobileNetV2 comes with ImageNet-trained features
2. **Reduced Training Time**: Faster convergence with frozen base layers
3. **Better Generalization**: Features learned on diverse ImageNet dataset
4. **Lower Data Requirements**: Can achieve good results with less data

*Note: Lower performance in our case due to artificial dataset size limitation (memory constraints)*

#### Text Classification:
1. **Language Understanding**: DistilBERT has deep contextual understanding
2. **Robust Performance**: Trained on massive text corpora
3. **Immediate Deployment**: Ready-to-use without additional training
4. **Contextual Embeddings**: Better semantic understanding than basic embeddings

### Performance Improvement Strategies

#### How Transfer Learning Improves Performance:

1. **Feature Reuse**: 
   - Pre-trained models have learned general features (edges, textures, patterns)
   - Saves computational resources and time
   - Reduces need for large datasets

2. **Better Initialization**:
   - Starts from meaningful feature representations
   - Faster convergence compared to random initialization
   - More stable training process

3. **Domain Adaptation**:
   - Fine-tuning allows adaptation to specific tasks
   - Leverages general knowledge while learning task-specific patterns

#### How Pre-trained Embeddings Improve Performance:

1. **Semantic Understanding**:
   - Transformer models understand context and relationships
   - Better handling of synonyms, antonyms, and nuances
   - Robust to variations in expression

2. **Large-scale Training**:
   - Trained on billions of text samples
   - Captures linguistic patterns across diverse domains
   - Generalizes well to new text

3. **Attention Mechanisms**:
   - Focus on relevant parts of input
   - Better handling of long sequences
   - Improved contextual understanding

## Conclusions

1. **Custom CNN** performed exceptionally well on MNIST (98.48% accuracy) due to the dataset's characteristics and sufficient training data.

2. **Transfer Learning** showed promise but was limited by our experimental setup (memory constraints led to reduced dataset size).

3. **Custom RNN** struggled due to extremely limited training data, highlighting the importance of sufficient data for deep learning.

4. **Pre-trained Transformers** demonstrated superior performance (>99% confidence) due to extensive pre-training on large corpora.

5. **Key Takeaway**: Pre-trained models and transfer learning are essential for practical applications, especially when training data is limited or computational resources are constrained.

## Technical Implementation Notes

- **Constraint Compliance**: All code implemented without loops, conditionals, try-except blocks, or lambda functions
- **Dataset**: MNIST from sklearn (70,000 samples) for image classification
- **Memory Management**: Reduced dataset size for transfer learning to prevent memory overflow
- **Frameworks**: TensorFlow/Keras for deep learning, Hugging Face Transformers for pre-trained models

## Future Improvements

1. **Increase Training Data**: Use full dataset for transfer learning with better memory management
2. **Data Augmentation**: Apply image transformations to increase dataset diversity
3. **Hyperparameter Tuning**: Optimize learning rates, batch sizes, and model architectures
4. **Ensemble Methods**: Combine multiple models for better performance
5. **Advanced Architectures**: Experiment with ResNet, Vision Transformers, or BERT variants
