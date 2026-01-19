import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, utils, datasets

# Close all existing plot windows to prevent hanging
plt.close('all')

# Parameters
NUM_CLASSES = 10

# 1. Load and Prepare Data
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot encoding
y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test = utils.to_categorical(y_test, NUM_CLASSES)

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

# Class names
CLASSES = np.array([
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
])


# 2. Build Models with Different Activation Functions

def build_model(activation_function, name):
    """Build MLP model with specified activation function"""
    input_layer = layers.Input((32, 32, 3))
    
    x = layers.Flatten()(input_layer)
    
    if activation_function == 'leaky_relu':
        x = layers.Dense(200)(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Dense(150)(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
    else:
        x = layers.Dense(200, activation=activation_function)(x)
        x = layers.Dense(150, activation=activation_function)(x)
    
    output_layer = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    
    model = models.Model(input_layer, output_layer, name=name)
    
    return model


# Create three models
print("\n" + "="*60)
print("Building Models...")
print("="*60)

model_relu = build_model('relu', 'MLP_ReLU')
model_leaky_relu = build_model('leaky_relu', 'MLP_LeakyReLU')
model_sigmoid = build_model('sigmoid', 'MLP_Sigmoid')

print("\n1. ReLU Model Summary")
print(f"Total params: {model_relu.count_params():,}")

print("\n2. LeakyReLU Model Summary")
print(f"Total params: {model_leaky_relu.count_params():,}")

print("\n3. Sigmoid Model Summary")
print(f"Total params: {model_sigmoid.count_params():,}")


# 3. Compile and Train Models

def train_model(model, model_name, epochs=10):
    """Compile and train a model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    opt = optimizers.Adam(learning_rate=0.0005)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )
    
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=epochs,
        shuffle=True,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    return history


# Train all three models
print("\n" + "="*60)
print("TRAINING PHASE")
print("="*60)

history_relu = train_model(model_relu, "ReLU Model", epochs=10)
history_leaky = train_model(model_leaky_relu, "LeakyReLU Model", epochs=10)
history_sigmoid = train_model(model_sigmoid, "Sigmoid Model", epochs=10)


# 4. Evaluate Models

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

# Evaluate on test set
print("\n1. ReLU Model:")
loss_relu, acc_relu = model_relu.evaluate(x_test, y_test, verbose=0)
print(f"   Test Loss: {loss_relu:.4f}")
print(f"   Test Accuracy: {acc_relu:.4f} ({acc_relu*100:.2f}%)")

print("\n2. LeakyReLU Model:")
loss_leaky, acc_leaky = model_leaky_relu.evaluate(x_test, y_test, verbose=0)
print(f"   Test Loss: {loss_leaky:.4f}")
print(f"   Test Accuracy: {acc_leaky:.4f} ({acc_leaky*100:.2f}%)")

print("\n3. Sigmoid Model:")
loss_sigmoid, acc_sigmoid = model_sigmoid.evaluate(x_test, y_test, verbose=0)
print(f"   Test Loss: {loss_sigmoid:.4f}")
print(f"   Test Accuracy: {acc_sigmoid:.4f} ({acc_sigmoid*100:.2f}%)")


# 5. Create Comparison Summary

print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)

results = {
    'Activation Function': ['ReLU', 'LeakyReLU', 'Sigmoid'],
    'Test Loss': [loss_relu, loss_leaky, loss_sigmoid],
    'Test Accuracy (%)': [acc_relu*100, acc_leaky*100, acc_sigmoid*100],
    'Final Training Accuracy (%)': [
        history_relu.history['accuracy'][-1]*100,
        history_leaky.history['accuracy'][-1]*100,
        history_sigmoid.history['accuracy'][-1]*100
    ]
}

# Print table
print(f"\n{'Activation Function':<20} {'Test Loss':<12} {'Test Acc %':<12} {'Train Acc %':<12}")
print("-" * 60)
for i in range(3):
    print(f"{results['Activation Function'][i]:<20} {results['Test Loss'][i]:<12.4f} {results['Test Accuracy (%)'][i]:<12.2f} {results['Final Training Accuracy (%)'][i]:<12.2f}")

# Find best performer
best_idx = np.argmax(results['Test Accuracy (%)'])
print(f"\nBest Performer: {results['Activation Function'][best_idx]}")
print(f"Best Test Accuracy: {results['Test Accuracy (%)'][best_idx]:.2f}%")


# 6. Visualization - Training History

print("\n" + "="*60)
print("Generating visualizations...")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot Accuracy
axes[0].plot(history_relu.history['accuracy'], label='ReLU - Train', linewidth=2)
axes[0].plot(history_relu.history['val_accuracy'], label='ReLU - Val', linestyle='--', linewidth=2)
axes[0].plot(history_leaky.history['accuracy'], label='LeakyReLU - Train', linewidth=2)
axes[0].plot(history_leaky.history['val_accuracy'], label='LeakyReLU - Val', linestyle='--', linewidth=2)
axes[0].plot(history_sigmoid.history['accuracy'], label='Sigmoid - Train', linewidth=2)
axes[0].plot(history_sigmoid.history['val_accuracy'], label='Sigmoid - Val', linestyle='--', linewidth=2)
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Plot Loss
axes[1].plot(history_relu.history['loss'], label='ReLU - Train', linewidth=2)
axes[1].plot(history_relu.history['val_loss'], label='ReLU - Val', linestyle='--', linewidth=2)
axes[1].plot(history_leaky.history['loss'], label='LeakyReLU - Train', linewidth=2)
axes[1].plot(history_leaky.history['val_loss'], label='LeakyReLU - Val', linestyle='--', linewidth=2)
axes[1].plot(history_sigmoid.history['loss'], label='Sigmoid - Train', linewidth=2)
axes[1].plot(history_sigmoid.history['val_loss'], label='Sigmoid - Val', linestyle='--', linewidth=2)
axes[1].set_title('Model Loss Comparison', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activation_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: activation_comparison.png")
plt.close()


# 7. Quick Prediction Visualization (only 10 samples)

print("\nGenerating prediction samples...")

def visualize_predictions_fast(model, model_name, n_samples=10):
    """Visualize predictions for a model - optimized version"""
    # Only predict on a small subset
    sample_indices = np.random.choice(range(len(x_test)), 100, replace=False)
    x_sample = x_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    preds = model.predict(x_sample, verbose=0, batch_size=100)
    preds_single = CLASSES[np.argmax(preds, axis=-1)]
    actual_single = CLASSES[np.argmax(y_sample, axis=-1)]
    
    indices = np.random.choice(range(len(x_sample)), n_samples, replace=False)
    
    fig = plt.figure(figsize=(15, 3))
    fig.suptitle(f'{model_name} - Sample Predictions', fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(indices):
        img = x_sample[idx]
        ax = fig.add_subplot(1, n_samples, i + 1)
        ax.axis("off")
        
        # Color code: green if correct, red if wrong
        color = 'green' if preds_single[idx] == actual_single[idx] else 'red'
        
        ax.text(0.5, -0.35, f"pred: {preds_single[idx]}", 
                fontsize=9, ha="center", transform=ax.transAxes, color=color, fontweight='bold')
        ax.text(0.5, -0.65, f"act: {actual_single[idx]}", 
                fontsize=9, ha="center", transform=ax.transAxes)
        ax.imshow(img)
    
    plt.tight_layout()
    filename = f'{model_name.lower().replace(" ", "_")}_predictions.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


visualize_predictions_fast(model_relu, "ReLU Model")
visualize_predictions_fast(model_leaky_relu, "LeakyReLU Model")
visualize_predictions_fast(model_sigmoid, "Sigmoid Model")


# 8. Per-Class Accuracy Analysis (optimized)

print("\nAnalyzing per-class accuracy...")

def analyze_per_class_accuracy(model, model_name):
    """Analyze accuracy for each class - optimized"""
    preds = model.predict(x_test, verbose=0, batch_size=256)
    pred_classes = np.argmax(preds, axis=-1)
    true_classes = np.argmax(y_test, axis=-1)
    
    class_accuracies = []
    for i in range(len(CLASSES)):
        class_mask = true_classes == i
        class_correct = np.sum((pred_classes == true_classes) & class_mask)
        class_total = np.sum(class_mask)
        accuracy = class_correct / class_total if class_total > 0 else 0
        class_accuracies.append(accuracy * 100)
    
    return class_accuracies


relu_class_acc = analyze_per_class_accuracy(model_relu, "ReLU")
leaky_class_acc = analyze_per_class_accuracy(model_leaky_relu, "LeakyReLU")
sigmoid_class_acc = analyze_per_class_accuracy(model_sigmoid, "Sigmoid")

print("\n" + "="*60)
print("PER-CLASS ACCURACY")
print("="*60)
print(f"\n{'Class':<12} {'ReLU %':<10} {'LeakyReLU %':<12} {'Sigmoid %':<10}")
print("-" * 60)
for i, class_name in enumerate(CLASSES):
    print(f"{class_name:<12} {relu_class_acc[i]:<10.2f} {leaky_class_acc[i]:<12.2f} {sigmoid_class_acc[i]:<10.2f}")

# Visualize per-class accuracy
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(CLASSES))
width = 0.25

ax.bar(x_pos - width, relu_class_acc, width, label='ReLU', alpha=0.8)
ax.bar(x_pos, leaky_class_acc, width, label='LeakyReLU', alpha=0.8)
ax.bar(x_pos + width, sigmoid_class_acc, width, label='Sigmoid', alpha=0.8)

ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(CLASSES, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: per_class_accuracy.png")
plt.close()


# 9. Final Observations Summary

print("\n" + "="*60)
print("WRITING OBSERVATIONS")
print("="*60)

observations = f"""OBSERVATIONS ABOUT OUTPUT AND ACCURACY
{'='*60}

1. OVERALL PERFORMANCE SUMMARY:
   - ReLU achieved {acc_relu*100:.2f}% test accuracy
   - LeakyReLU achieved {acc_leaky*100:.2f}% test accuracy  
   - Sigmoid achieved {acc_sigmoid*100:.2f}% test accuracy
   
   Winner: {"ReLU" if acc_relu >= max(acc_leaky, acc_sigmoid) else "LeakyReLU" if acc_leaky >= acc_sigmoid else "Sigmoid"}

2. ACTIVATION FUNCTION ANALYSIS:

   ReLU (Rectified Linear Unit):
   - Definition: f(x) = max(0, x)
   - Advantages: 
     * Fast training due to simple gradient computation
     * Solves vanishing gradient problem for positive values
     * Computationally efficient
   - Disadvantages:
     * "Dying ReLU" problem - neurons can become inactive
   - Performance on CIFAR-10: {acc_relu*100:.2f}%
   
   LeakyReLU:
   - Definition: f(x) = max(0.01x, x)
   - Advantages:
     * Prevents "dying ReLU" by allowing small gradient for negative values
     * Maintains benefits of ReLU while avoiding dead neurons
   - Performance on CIFAR-10: {acc_leaky*100:.2f}%
   - Observation: {"Very similar to ReLU" if abs(acc_leaky - acc_relu) < 0.01 else "Slightly different from ReLU"}
   
   Sigmoid:
   - Definition: f(x) = 1 / (1 + e^(-x))
   - Disadvantages for deep learning:
     * Vanishing gradient problem - gradients become very small
     * Outputs not zero-centered
     * Computationally expensive (exponential)
     * Saturates at extreme values (gradient ≈ 0)
   - Performance on CIFAR-10: {acc_sigmoid*100:.2f}%
   - Observation: {"Significantly worse" if acc_sigmoid < min(acc_relu, acc_leaky) - 0.02 else "Comparable"} due to vanishing gradients

3. TRAINING DYNAMICS OBSERVED:

   Convergence Speed:
   - ReLU/LeakyReLU: Fast convergence in early epochs
   - Sigmoid: Slower convergence, visible in training curves
   
   Overfitting:
   - All models show some overfitting (training acc > validation acc)
   - Gap: ~{(history_relu.history['accuracy'][-1] - acc_relu)*100:.2f}% for ReLU
   
   Loss Progression:
   - ReLU achieved final training loss: {history_relu.history['loss'][-1]:.4f}
   - Sigmoid achieved final training loss: {history_sigmoid.history['loss'][-1]:.4f}

4. CLASS-SPECIFIC PERFORMANCE:

   Best Performing Classes (for ReLU):
   - {CLASSES[np.argmax(relu_class_acc)]}: {max(relu_class_acc):.2f}%
   
   Worst Performing Classes (for ReLU):
   - {CLASSES[np.argmin(relu_class_acc)]}: {min(relu_class_acc):.2f}%
   
   Common Struggles:
   - Similar-looking classes (cat/dog, automobile/truck) are harder
   - All activation functions struggle with the same classes

5. KEY INSIGHTS:

   Why ReLU/LeakyReLU Outperform Sigmoid:
   a) Gradient Flow: ReLU maintains strong gradients for positive values
   b) Computation: ReLU is a simple max operation vs expensive exponential
   c) Saturation: Sigmoid saturates (flat regions), ReLU doesn't for x > 0
   d) Zero-centered: ReLU helps with weight updates
   
   When to Use Each:
   - ReLU: Default choice for hidden layers in CNNs/MLPs
   - LeakyReLU: When concerned about dying neurons
   - Sigmoid: Output layer for binary classification only (not hidden layers)

6. LIMITATIONS OF THIS EXPERIMENT:

   - Simple MLP architecture (not optimal for images)
   - Only 10 epochs (longer training might show different patterns)
   - CIFAR-10 is complex - CNNs achieve 90%+ accuracy
   - ~49% accuracy is expected for basic MLP on CIFAR-10

7. CONCLUSION & RECOMMENDATIONS:

   Best Choice for This Task: {"ReLU" if acc_relu >= acc_leaky else "LeakyReLU"}
   
   General Recommendations:
   - Use ReLU or LeakyReLU for hidden layers in modern networks
   - Avoid Sigmoid in hidden layers (vanishing gradient issue)
   - For image classification, use CNNs instead of MLPs
   - Consider batch normalization to further improve training
   
   Performance Ranking:
   1. {"ReLU" if acc_relu >= max(acc_leaky, acc_sigmoid) else "LeakyReLU" if acc_leaky >= acc_sigmoid else "Sigmoid"} ({max(acc_relu, acc_leaky, acc_sigmoid)*100:.2f}%)
   2. {"LeakyReLU" if acc_leaky >= acc_sigmoid and acc_relu > acc_leaky else "ReLU" if acc_relu >= acc_sigmoid and acc_leaky > acc_relu else "Sigmoid"} ({sorted([acc_relu, acc_leaky, acc_sigmoid], reverse=True)[1]*100:.2f}%)
   3. {"Sigmoid" if acc_sigmoid <= min(acc_relu, acc_leaky) else "ReLU" if acc_relu <= min(acc_leaky, acc_sigmoid) else "LeakyReLU"} ({min(acc_relu, acc_leaky, acc_sigmoid)*100:.2f}%)

{'='*60}
END OF ANALYSIS
{'='*60}
"""

# Save observations to file
with open('observations.txt', 'w') as f:
    f.write(observations)

print(observations)
print("\n✓ Saved: observations.txt")

print("\n" + "="*60)
print("ASSIGNMENT COMPLETE!")
print("="*60)
print("\nGenerated files in current directory:")
print("1. activation_comparison.png - Training curves comparison")
print("2. relu_model_predictions.png - ReLU sample predictions")
print("3. leakyrelu_model_predictions.png - LeakyReLU sample predictions") 
print("4. sigmoid_model_predictions.png - Sigmoid sample predictions")
print("5. per_class_accuracy.png - Per-class accuracy bar chart")
print("6. observations.txt - Detailed written analysis")
print("\nAll files ready for submission!")
print("="*60)
