# MiniTorch Workshop Instructions

Welcome to the **MiniTorch Workshop**.
In this activity, you will implement a simple deep learning framework from scratch and then use it to participate in an AI competition.

---

## 📌 Stage 1: Complete the MiniTorch Workshop

Start by completing the notebook where you will build the basic components of a neural network (layers, activations, loss functions, forward/backward passes):

👉 [MiniTorchWorkshop.ipynb](https://github.com/jdmartinev/ArtificialIntelligenceIM/blob/main/Workshops/Workshop1/MiniTorchWorkshop.ipynb)

### Goals

* Implement forward and backward passes for:

  * Linear layers
  * Activation functions (ReLU, etc.)
  * Cost functions (Cross-Entropy)
* Extend the framework to include:

  * Batch Normalization
  * Dropout
* Train and evaluate a neural network on MNIST.

---

## 📌 Stage 2: Apply Your Network Library in a Kaggle Competition

Once your MiniTorch framework is working, you will use it to solve a real classification challenge.

👉 [AI Competition Notebook](https://www.kaggle.com/code/juanmartinezv4399/ai-competition01)

### Goals

* Import your `minitorch.py` library into the Kaggle environment.
* Design different network architectures using the layers you implemented.
* Train and validate your models on the provided dataset.
* Compare results and submit predictions to the competition.

---

## 📌 Stage 3: Submit Your Predictions via the Hugging Face Competition

You must also upload your final predictions to the official competition space:

👉 https://huggingface.co/spaces/MLEAFIT/AIComp01202601

### 🎯 Objective

Provide a standardized submission so results can be evaluated and compared across all participants.

---

## 🧾 Submission Steps

### 1️⃣ Generate Predictions

From your competition notebook, export a CSV file with the following format:

```
id,pred \
image_0001.jpg,0 \
image_0002.jpg,1 \
image_0003.jpg,2
...
```

Where:

* **id** → sample identifier (same order as the test set)
* **pred** → predicted class label

Save the file as:

```
submission.csv
```

---

### 2️⃣ Go to the Competition Space

➡️ https://huggingface.co/spaces/MLEAFIT/AIComp01202601

---

### 3️⃣ Upload Your File

Inside the interface:

1. Locate the **Upload / Submit Predictions** section
2. Drag and drop your `submission.csv` file
   **or** click to browse your computer
3. Confirm the upload

---

### 4️⃣ Verify Your Submission

After uploading:

* Check that the file appears in the submissions list
* Confirm that no format errors are reported

---

## 📊 Evaluation

Your score will be computed automatically based on:

* Prediction accuracy (or competition metric)
* Correct file format

The leaderboard will update periodically.

---

## ⚠️ Important Rules

* You may submit **multiple times**, but only your best score counts.
* The CSV must:

  * Contain **only two columns**
  * Have **no missing rows**
  * Preserve the original test order

---

## ✅ Deliverables

1. **Completed `MiniTorchWorkshop.ipynb`**
   with all forward and backward passes implemented.

2. **`minitorch.py` file**
   containing your neural network library.

3. **Competition Notebook**
   where you experiment with different architectures and submit results.

4. **Report (short)**

   * Best model architecture
   * Training and validation curves
   * Final accuracy or score
   * Short reflection on BatchNorm/Dropout impact

---

## 💡 Tips

* Keep your code modular and clean.
* Use `#TODO` markers to track pending implementations.
* Run small experiments before training deeper networks.
* Share and discuss your results with peers to improve your models.

---

## 🚀 Outcome

By the end of this workshop, you will have:

* Implemented your own neural network framework.
* Understood the mechanics of forward and backward propagation.
* Experimented with regularization techniques like BatchNorm and Dropout.
* Applied your framework to a real-world dataset in Kaggle.
* Submitted predictions through the Hugging Face competition platform.

---
