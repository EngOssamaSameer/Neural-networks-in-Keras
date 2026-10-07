# Neural Networks in Keras

A practical Jupyter Notebook project demonstrating how to build and train neural networks with **TensorFlow/Keras** for classification, image recognition, and regression.

## Project Overview

This notebook covers three main neural-network tasks:

1. **Binary classification** using a simple 2-feature dataset.
2. **Handwritten digit recognition** using the MNIST dataset.
3. **House price regression** using a Hyderabad housing dataset.

The notebook also includes data inspection, preprocessing, model building, training, and evaluation.

## Technologies Used

- Python
- NumPy
- Pandas
- TensorFlow
- Keras
- Matplotlib
- Scikit-learn concepts for machine-learning preprocessing

## 1. Binary Classification

The notebook starts with a dataset called `one_circle.csv`.

### Data Preparation

- Loads the dataset with Pandas.
- Removes the `Unnamed: 0` column.
- Separates the features `X` from the target `y`.
- Converts the target into categorical form using `to_categorical`.

### Neural Network Architecture

The classification model uses:

- Input: 2 features
- Dense layer: 128 neurons, ReLU
- Dropout: 20%
- Dense layer: 64 neurons, ReLU
- Dropout: 20%
- Output layer: 2 neurons, Softmax

### Compilation

The model is compiled with:

- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Metric: `accuracy`

The model is trained for 100 epochs with a batch size of 10.

## 2. Handwritten Digit Recognition with MNIST

The second part uses the built-in **MNIST** dataset from Keras.

MNIST contains grayscale images of handwritten digits from **0 to 9**.

### Loading the Dataset

```python
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
```

The notebook demonstrates how to display:

- A single MNIST image.
- Multiple images using a loop.
- The corresponding digit labels.

### Image Shape

Each MNIST image has a size of:

```text
28 × 28 pixels
```

### Preprocessing

The original notebook first demonstrates reshaping the images:

```python
X_train_reshaped = X_train.reshape(-1, 28 * 28)
X_test_reshaped = X_test.reshape(-1, 28 * 28)
```

The labels are converted to categorical form:

```python
y_train_cate = to_categorical(y_train, 10)
y_test_cate = to_categorical(y_test, 10)
```

### Neural Network Architecture

The first MNIST model contains:

- Input: 784 features (`28 × 28`)
- Dense layer: 128 neurons, ReLU
- Dropout: 20%
- Dense layer: 24 neurons, ReLU
- Dropout: 20%
- Output layer: 10 neurons, Softmax

The model is trained for 10 epochs with a batch size of 10.

### Prediction and Evaluation

The notebook generates predictions using:

```python
prediction_vector = model.predict(X_train_reshaped)
```

The predicted class is extracted using `argmax`.

The model is evaluated on the test data using:

```python
loss, accuracy = model.evaluate(X_test_reshaped, y_test_cate)
```

## 3. Improved MNIST Model

The notebook also contains an improved version of the MNIST model.

### Improvements

The improved version:

- Normalizes pixel values from `0–255` to `0–1`.
- Uses the `Flatten` layer instead of manually reshaping the images.
- Uses a 128-neuron hidden layer.
- Uses a 64-neuron hidden layer.
- Uses Dropout for regularization.
- Uses 10 output neurons for the ten MNIST classes.
- Uses `sparse_categorical_crossentropy` because the labels remain integer class labels.
- Uses a validation split during training.

### Model Architecture

```text
Input: 28 × 28 image
        ↓
Flatten
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.2)
        ↓
Dense(64, ReLU)
        ↓
Dropout(0.2)
        ↓
Dense(10, Softmax)
```

### Training

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)
```

### Testing

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_accuracy)
```

## 4. House Price Regression

The final section applies a neural network to a housing-price prediction problem using the Hyderabad housing dataset.

### Dataset

The dataset contains **2,518 records and 40 columns**, including:

- `Price`
- `Area`
- `Location`
- `No. of Bedrooms`
- Property facilities and amenities

The `Location` column contains **243 unique locations**.

### Current Feature Selection

In the notebook, `Location` and `Price` are removed from the feature dataframe:

```python
feature = housing.drop(['Location', 'Price'], axis=1)
```

The target is:

```python
labels = housing['Price']
```

This leaves 38 numerical features for the regression model.

### Neural Network Architecture

The regression model contains:

- Input: 38 features
- Dense layer: 38 neurons, ReLU
- Dropout: 20%
- Dense layer: 128 neurons, ReLU
- Dropout: 20%
- Dense layer: 64 neurons, ReLU
- Dropout: 20%
- Output layer: 1 neuron

The final output neuron produces the predicted house price.

### Compilation

The model uses:

- Loss: `Mean Squared Error`
- Optimizer: `Adam`
- Metric: `Root Mean Squared Error`

```python
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=[keras.metrics.RootMeanSquaredError()]
)
```

The model is trained for 10 epochs with a batch size of 10.

## Project Structure

```text
.
├── neural-networks-in-keras.ipynb
└── README.md
```

## Installation

Install the main dependencies with:

```bash
pip install numpy pandas tensorflow matplotlib scikit-learn
```

## How to Run

1. Clone or download the project.
2. Open `neural-networks-in-keras.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.
3. Make sure the required datasets are available.
4. Run the notebook cells in order.

> The original notebook uses Kaggle dataset paths such as `/kaggle/input/...`. If you run it outside Kaggle, update these paths to the location of your datasets.

## Key Concepts Covered

- Neural Networks
- Dense Layers
- Activation Functions
- ReLU
- Softmax
- Dropout
- Forward Prediction
- Classification
- Regression
- One-Hot Encoding
- Image Flattening
- Image Normalization
- MNIST
- Model Training
- Model Evaluation
- Accuracy
- Mean Squared Error
- Root Mean Squared Error

## Notes

The notebook is designed as a learning project and demonstrates different neural-network workflows in Keras. The house-price section currently excludes `Location` from the model rather than encoding its 243 unique values.

## Author

**Ossama Samir**

Machine Learning / AI Learning Project
