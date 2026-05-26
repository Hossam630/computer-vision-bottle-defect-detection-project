Here’s a **clear, structured, deep explanation** of each of your five questions — not just definitions, but *why they matter* for training and how they affect your model’s behavior.

---

# 1️⃣ **Loss functions** — what other options exist and why they matter

The loss function defines **what “error” means** for your model. Changing it changes *what the model learns to optimize*.

### Common alternatives to `CrossEntropyLoss` for classification
- **NLLLoss** — used with log‑softmax outputs  
- **BCEWithLogitsLoss** — for multi‑label classification  
- **Focal Loss** — handles class imbalance by down‑weighting easy examples  
- **Label Smoothing Loss** — reduces overconfidence and improves generalization  

### For regression tasks
- **MSELoss** — mean squared error  
- **L1Loss** — more robust to outliers  
- **Huber Loss** — hybrid of L1 and MSE  

### Why the choice matters
- It changes **gradient shape**, which affects how fast and how stably the model learns.  
- Some losses handle **imbalanced datasets** better.  
- Some losses encourage **smoother** or **more confident** predictions.  
- The wrong loss can make training **unstable** or **fail entirely**.

---

# 2️⃣ **Optimizers** — alternatives to Adam and how parameters affect training

### Popular alternatives
- **SGD**  
  - Often with momentum  
  - More stable, sometimes better generalization  
- **AdamW**  
  - Adam + proper weight decay  
  - Standard for transformers  
- **RMSprop**  
  - Good for RNNs  
- **Adagrad**  
  - Adapts learning rate per parameter  
- **Adadelta**  
  - Variant of Adagrad with less aggressive decay  

### Important optimizer parameters
- **learning_rate (lr)**  
  - Most important hyperparameter  
  - Too high → divergence  
  - Too low → slow training  

- **momentum (SGD)**  
  - Helps escape local minima  
  - Smooths updates  

- **weight_decay**  
  - Regularization  
  - Prevents overfitting  

- **betas (Adam/AdamW)**  
  - Control how fast moving averages adapt  
  - Affects stability and convergence  

### Why optimizer choice matters
- Determines **how fast** and **how smoothly** the model converges  
- Affects **generalization**  
- Some optimizers handle **sparse gradients** better  
- Some are more stable for **large models** (AdamW)

---

# 3️⃣ **Checkpoint saving** — does it save only the best model?

Yes — in your code, **only the best validation model** is saved.

```python
if phase == "val" and epoch_acc > best_acc:
    torch.save(best_weights, best_model_path)
```

This means:
- No per‑epoch checkpoints  
- No saving of the last model  
- Only the **best validation accuracy** model is stored  

This is a common and safe strategy.

---

# 4️⃣ **Epoch** — what it means

An **epoch** is **one full pass through the entire training dataset**.

If you have 10,000 images and your batch size is 100:
- 1 epoch = 100 batches  
- After 1 epoch, the model has seen every training sample once  

Training usually takes **multiple epochs** because the model needs repeated exposure to the data to learn patterns.

---

# 5️⃣ **Batches** — what they are and why they matter

A **batch** is a small subset of the dataset processed at once.

If batch size = 32:
- The model processes 32 samples  
- Computes loss  
- Computes gradients  
- Updates weights  

### Why we use batches instead of full dataset
- Fits in GPU memory  
- Faster updates  
- Adds noise → helps generalization  
- Allows parallelization  

### Batch size effects
- **Small batch (8–32)**  
  - Noisy gradients → better generalization  
  - Slower per epoch  

- **Large batch (128–1024+)**  
  - Faster per epoch  
  - Requires more memory  
  - Can overfit or converge to sharp minima  

---