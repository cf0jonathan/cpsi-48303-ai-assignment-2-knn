# Assignment 2 — Your First AI Model: Classification with K-Nearest Neighbors

**Course:** FALL 2026 CPSI 48303-01 — Artificial Intelligence  
**Due:** 10/4/26, 11:59 PM (CDT)  
**Points:** 100  
**Submission:** One Python notebook (`.ipynb`)

---

## Objective

In Assignment 1, you learned how to:

- Load a dataset using pandas
- Explore the dataset using `.head()`, `.info()`, and `.describe()`
- Create scatter plots and pair plots using matplotlib
- Manually identify features and labels
- Distinguish features from labels and explain why each column fits its role

Now you will use that data to build your **first AI model**.

You will learn how to:

- Prepare data for machine learning
- Split data into training and testing sets
- Create a K-Nearest Neighbors (KNN) classifier
- Train the model on training data
- Make predictions on new data
- Evaluate model accuracy
- Interpret model predictions
- Experiment with model parameters
- Compare model performance with different features
- Reflect on the AI development process

---

## Part 1 — Prepare the Dataset

**Import the required libraries:**

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

```

**Load the Iris dataset:**

```python
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

data = pd.read_csv(url)
```

**Define the features:**

```python
X = data[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
]
```

**Define the target:**

```python
y = data["species"]
```

**Split the dataset:**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## Task 1 — Create Your First KNN Model

Create a K-Nearest Neighbors classifier:

```python
model = KNeighborsClassifier(n_neighbors=3)
```

**Answer:** Explain what `KNeighborsClassifier` does and what `n_neighbors=3` means.

---

## Task 2 — Train the Model

Train the model:

```python
model.fit(X_train, y_train)
```

**Answer:** Explain what happens during the `fit()` step.

---

## Task 3 — Make Predictions

Use the trained model:

```python
predictions = model.predict(X_test)
```

Display the predictions:

```python
print(predictions)
```

Display the actual answers:

```python
print(y_test)
```

**Answer:** Compare the predictions to the actual values. Are they close?

---

## Task 4 — Measure Accuracy

Calculate the accuracy:

```python
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
```

**Record your result:** My model accuracy: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Answer:** What does this accuracy value mean? Is it good?

---

## Task 5 — Compare Actual and Predicted Results

Create a table:

```python
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print(results)
```

Add a column indicating whether each prediction is correct:

```python
results["Correct"] = results["Actual"] == results["Predicted"]

print(results)
```

**Answer:** How many predictions were correct? Were there any surprising misclassifications?

---

## Task 6 — Predict a New Flower

Create a new flower:

```python
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
)

prediction = model.predict(new_flower)

print("Predicted Species:", prediction[0])
```

**Record:**

```
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2

Predicted Species: __________________
```

**Answer:** Does the prediction make sense compared with the visualizations you created in Assignment 1? Explain.

---

## Task 7 — Experiment With Three New Flowers

Create your own three flowers by choosing different measurements.

### Flower 1

```
Sepal Length:
Sepal Width:
Petal Length:
Petal Width:

Predicted Species:
```

### Flower 2

```
Sepal Length:
Sepal Width:
Petal Length:
Petal Width:

Predicted Species:
```

### Flower 3

```
Sepal Length:
Sepal Width:
Petal Length:
Petal Width:

Predicted Species:
```

**Answer:** Are your predictions what you expected based on the scatter plots from Assignment 1?

---

## Task 8 — Experiment With Different Values of K

So far, we used:

```python
n_neighbors=3
```

Now experiment with:

- K = 1
- K = 3
- K = 5
- K = 10

Example:

```python
model_1 = KNeighborsClassifier(n_neighbors=1)

model_1.fit(X_train, y_train)

prediction_1 = model_1.predict(X_test)

accuracy_1 = accuracy_score(y_test, prediction_1)

print(accuracy_1)
```

Repeat for K = 3, 5, and 10. Complete the table:

| K | Accuracy |
|---|----------|
| 1 |          |
| 3 |          |
| 5 |          |
|10 |          |

**Answer:** Which value of K gave the best accuracy? Why might that be?

---

## Task 9 — Visualize K vs. Accuracy

Create a graph showing **K vs. Accuracy**.

Create lists containing your experimental values:

```python
k_values = [1, 3, 5, 10]

accuracies = [
    accuracy_1,
    accuracy_3,
    accuracy_5,
    accuracy_10
]
```

Create the figure:

```python
plt.plot(k_values, accuracies, marker="o")

plt.title("KNN Accuracy vs. Number of Neighbors")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")

plt.show()
```

**Answer:** Describe the trend you see. Does more neighbors always mean better accuracy?

---

## Task 10 — Connect KNN to Your Assignment 1 Visualization

Return to the **scatter plot from Assignment 1**.

Think about how the different species formed groups.

**Answer:**
- How did the three species separate in the scatter plot?
- How does KNN decide which species a new flower belongs to?
- How does looking at nearby points relate to the clusters you saw in Assignment 1?
- Why might petal length and petal width be more useful for classification than sepal measurements?

You are not expected to provide a mathematical explanation. Explain using your understanding of nearby data points.

---

## Task 11 — Use Fewer Features

Your original model uses four features:

```
Sepal Length
Sepal Width
Petal Length
Petal Width
```

Now create another model using only:

```
Petal Length
Petal Width
```

Create the new feature set:

```python
X2 = data[
    [
        "petal_length",
        "petal_width"
    ]
]
```

Split the data:

```python
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2,
    y,
    test_size=0.2,
    random_state=42
)
```

Create the model:

```python
model_2features = KNeighborsClassifier(n_neighbors=3)
```

Train:

```python
model_2features.fit(X2_train, y2_train)
```

Predict:

```python
predictions_2features = model_2features.predict(X2_test)
```

Calculate accuracy:

```python
accuracy_2features = accuracy_score(
    y2_test,
    predictions_2features
)

print("Two-Feature Accuracy:", accuracy_2features)
```

Complete the comparison table:

| Model | Accuracy |
|---|---|
| Four Features |  |
| Petal Length + Petal Width |  |

**Answer:**
- How does the two-feature accuracy compare to the four-feature accuracy?
- Why might using only two features work nearly as well (or exactly as well)?
- What does this tell you about which measurements matter most for identifying Iris species?

---

## Task 12 — Think Like an AI Researcher

Answer the following questions in your own words.

### Question 1
What is the difference between `model.fit()` and `model.predict()`?

### Question 2
What is the difference between training data and testing data?

### Question 3
What are the features in this AI problem?

### Question 4
What is the label?

### Question 5
Why is this a classification problem instead of a regression problem?

### Question 6
Where does KNN obtain the information it uses to predict the species of a new flower?

### Question 7
Is KNN the same as manually writing:

```python
if petal_length < 2:
    species = "setosa"
```

Explain the difference.

### Question 8
What might happen if many training examples contained incorrect species labels?

### Question 9
What might happen if the model were trained using only five flowers?

### Question 10
Why do we evaluate an AI model using data it did not use during training?

### Question 11
Why can visualization be useful before and after building a machine-learning model?

### Question 12
What was the most interesting result you observed in your experiments?

---

## Complete AI Workflow

You have now completed:

```
Real-World Data
       ↓
Explore the Data
       ↓
Visualize Patterns
       ↓
Identify Features and Labels
       ↓
Prepare Training and Testing Data
       ↓
Choose an Algorithm
       ↓
Train the Model
       ↓
Make Predictions
       ↓
Evaluate Accuracy
       ↓
Change Model Parameters
       ↓
Compare Results
       ↓
Improve the Model
```

Write one sentence explaining what happens at each step.

---

## Submission Requirements

Submit one Python notebook (`.ipynb`) containing:

1. All code cells with the code shown in this assignment
2. Output for every code cell
3. Explanatory text cells (markdown) for each task
4. Clear labels for each task and sub-question
5. All answers to all questions (Tasks 1–12)
6. Your accuracy comparison table (Task 8)
7. Your K vs. Accuracy plot (Task 9)
8. Your feature comparison table (Task 11)
9. The complete AI workflow explanation
10. A clean, organized layout that is easy to follow

Your notebook should run from beginning to end without errors.

---

## Grading — 100 Points

| Component | Points |
|---|---|
| Dataset and model preparation | 5 |
| Create KNN model | 5 |
| Train model correctly | 10 |
| Generate and interpret predictions | 10 |
| Calculate and interpret accuracy | 10 |
| Actual vs. predicted comparison | 5 |
| Three new-flower experiments | 10 |
| Different K experiments | 15 |
| K vs. accuracy visualization | 10 |
| Feature-selection experiment | 10 |
| AI understanding and reflection | 10 |
| **Total** | **100** |