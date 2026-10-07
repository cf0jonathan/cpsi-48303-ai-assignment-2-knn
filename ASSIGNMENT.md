# Assignment 2 — Your First AI Model: Classification with K-Nearest Neighbors

**Course:** FALL 2026 CPSI 48303-01 — Artificial Intelligence  
**Due:** 10/4/26, 11:59 PM (CDT)  
**Points:** 100  
**Submission:** One Python notebook (.ipynb)

---

## Objective

In Assignment 1, you learned how to:

- Load a dataset using pandas
- Explore the dataset using `.head()`, `.info()`, and `.describe()`
- Create scatter plots and pair plots using matplotlib
- Manually identify features and labels
- Distinguish features from labels and explain why each column fits its role

Now you will use that data to build your first AI model.

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

**Answer:**

1. What does KNN stand for?
2. What do you think a "neighbor" means in this dataset?
3. What does n_neighbors=3 mean?
4. Why might an AI system examine similar examples when making a prediction?

---

## Task 2 — Train the Model

Train the model:

```python
model.fit(X_train, y_train)
```

**Answer:**

1. What does .fit() do?
2. Which data are being used for training?
3. Are X_test and y_test being used during training?
4. Why should testing data remain separate from training data?

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

**Answer:**

1. What does .predict() do?
2. Compare the predictions with the actual labels.
3. Do most predictions appear correct?
4. Did the model make any incorrect predictions?

---

## Task 4 — Measure Accuracy

Calculate the accuracy:

```python
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
```

**Record your result:**

```text
My model accuracy: __________________
```

**Answer:**

1. What does accuracy measure?
2. What does an accuracy of 1.0 mean?
3. What would an accuracy of 0.80 mean?
4. Does high accuracy automatically mean an AI system is perfect? Explain.

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

**Answer:**

1. Find one correctly classified flower.
2. Find one incorrectly classified flower, if one exists.
3. For an incorrect prediction, which species did the model predict instead?

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

```text
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2

Predicted Species: __________________
```

**Answer:**

### Does the prediction make sense compared with the visualizations you created in Assignment 1?

Explain.

---

## Task 7 — Experiment With Three New Flowers

Create your own three flowers by choosing different measurements.

### Flower 1

```text
Sepal Length:
Sepal Width:
Petal Length:
Petal Width:

Predicted Species:
```

### Flower 2

```text
Sepal Length:
Sepal Width:
Petal Length:
Petal Width:

Predicted Species:
```

### Flower 3

```text
Sepal Length:
Sepal Width:
Petal Length:
Petal Width:

Predicted Species:
```

**Answer:**

1. Did changing the measurements change the predicted species?
2. Which measurements seemed to have the greatest influence based on your experiments?

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

Repeat for K = 3, 5, and 10.

Complete:

| K | Accuracy |
|---|----------|
| 1 |          |
| 3 |          |
| 5 |          |
| 10 |          |

**Answer:**

1. Did changing K change the model accuracy?
2. Which value produced the highest accuracy?
3. Did multiple K values produce the same accuracy?
4. Why do you think changing the number of neighbors can influence a prediction?

---

## Task 9 — Visualize K vs. Accuracy

Create a graph showing:

**Number of Neighbors vs. Model Accuracy**

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

**Answer:**

1. Which value of K would you choose based on your experiment?
2. Why?
3. How does the graph make comparing models easier than simply reading numbers?

---

## Task 10 — Connect KNN to Your Assignment 1 Visualization

Return to the **Petal Length vs. Petal Width** scatter plot from Assignment 1.

Think about how the different species formed groups.

**Answer:**

1. How does the scatter plot help explain KNN?
2. If a new flower appears close to several Setosa flowers, what would you expect KNN to predict?
3. What could happen if a new flower appears near the boundary between Versicolor and Virginica?
4. Why might increasing K sometimes make predictions more stable?
5. Why might using a very large K also cause problems?

You are not expected to provide a mathematical explanation. Explain using your understanding of nearby data points.

---

## Task 11 — Use Fewer Features

Your original model uses four features:

```text
Sepal Length
Sepal Width
Petal Length
Petal Width
```

Now create another model using only:

```text
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

Complete:

| Model | Accuracy |
|---|---|
| Four Features |  |
| Petal Length + Petal Width |  |

**Answer:**

1. Did using only two features increase, decrease, or maintain accuracy?
2. Do AI models always need every available feature?
3. Why might petal length and petal width be particularly useful?
4. How did the visualizations from Assignment 1 help you understand this experiment?

---

## Task 12 — Think Like an AI Researcher

Answer the following questions in your own words.

### Question 1

What is the difference between:

```python
model.fit()
```

and:

```python
model.predict()
```

?

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

```text
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

Submit one Python notebook (.ipynb) containing:

1. Dataset loading
2. Features and labels
3. Training/testing split
4. KNN classifier with K = 3
5. Model training
6. Model predictions
7. Accuracy calculation
8. Actual vs. predicted table
9. Three new-flower experiments
10. Experiments using K = 1, 3, 5, and 10
11. K vs. accuracy table
12. K vs. accuracy visualization
13. Two-feature KNN experiment
14. Comparison of two-feature and four-feature performance
15. Answers to all questions

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
