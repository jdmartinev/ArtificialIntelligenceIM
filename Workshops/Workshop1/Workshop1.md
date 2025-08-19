# MiniTorch Workshop Instructions

Welcome to the **MiniTorch Workshop**.  
In this activity, you will implement a simple deep learning framework from scratch and then use it to participate in an AI competition.

---

## 📌 Stage 1: Complete the MiniTorch Workshop

Start by completing the notebook where you will build the basic components of a neural network (layers, activations, loss functions, forward/backward passes):

👉 [MiniTorchWorkshop.ipynb](https://github.com/jdmartinev/ArtificialIntelligenceIM/blob/main/Workshops/Workshop1/MiniTorchWorkshop.ipynb)

### Goals
- Implement forward and backward passes for:
  - Linear layers
  - Activation functions (ReLU, etc.)
  - Cost functions (Cross-Entropy)
- Extend the framework to include:
  - Batch Normalization
  - Dropout
- Train and evaluate a neural network on MNIST.

---

## 📌 Stage 2: Apply Your Network Library in a Kaggle Competition

Once your MiniTorch framework is working, you will use it to solve a real classification challenge.

👉 [AI Competition Notebook](https://www.kaggle.com/code/juanmartinezv4399/ai-competition01)

### Goals
- Import your `minitorch.py` library into the Kaggle environment.
- Design different network architectures using the layers you implemented.
- Train and validate your models on the provided dataset.
- Compare results and submit predictions to the competition.

---

## ✅ Deliverables

1. **Completed `MiniTorchWorkshop.ipynb`**  
   with all forward and backward passes implemented.

2. **`minitorch.py` file**  
   containing your neural network library.

3. **Competition Notebook**  
   where you experiment with different architectures and submit results.

4. **Report (short)**  
   - Best model architecture
   - Training and validation curves
   - Final accuracy or score
   - Short reflection on BatchNorm/Dropout impact

---

## 💡 Tips

- Keep your code modular and clean.  
- Use `#TODO` markers to track pending implementations.  
- Run small experiments before training deeper networks.  
- Share and discuss your results with peers to improve your models.

---

### 🚀 Outcome

By the end of this workshop, you will have:
- Implemented your own neural network framework.
- Understood the mechanics of forward and backward propagation.
- Experimented with regularization techniques like BatchNorm and Dropout.
- Applied your framework to a real-world dataset in Kaggle.
