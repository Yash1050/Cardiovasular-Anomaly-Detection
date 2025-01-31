# AML Project - Anomaly Detection in ECG Data

This project focuses on detecting anomalies in ECG (Electrocardiogram) data using machine learning techniques. The dataset used in this project is the PTB Diagnostic ECG Database, which contains both normal and abnormal ECG signals. The goal is to classify ECG signals as normal or anomalous using various data analysis and machine learning methods.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

## Project Overview

The project involves the following steps:

1. **Data Loading**: Load the normal and abnormal ECG datasets.
2. **Data Exploration**: Visualize the distribution of labels and basic statistics of the datasets.
3. **Data Preprocessing**: Handle missing values, normalize data, and split the dataset into training and testing sets.
4. **Feature Engineering**: Perform Principal Component Analysis (PCA) to reduce the dimensionality of the data.
5. **Model Training**: Train a machine learning model (e.g., a neural network) to classify ECG signals.
6. **Evaluation**: Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1-score.
7. **Visualization**: Create interactive visualizations to explore the data and model results.

## Dataset

The dataset used in this project is the PTB Diagnostic ECG Database, which contains ECG signals labeled as normal or abnormal. The dataset is split into two CSV files:

- `ptbdb_normal.csv`: Contains normal ECG signals.
- `ptbdb_abnormal.csv`: Contains abnormal ECG signals.

Each file contains 187 columns, where the first 186 columns represent the ECG signal features, and the last column is the label (0 for normal, 1 for abnormal).

## Dependencies

The following Python libraries are required to run this project:

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `tensorflow`
- `plotly`

You can install these dependencies using `pip`:

```bash
pip install numpy pandas seaborn matplotlib scipy scikit-learn tensorflow plotly
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/AML-Project.git
   ```

2. Navigate to the project directory:

   ```bash
   cd AML-Project
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Loading and Exploration**:
   - Load the normal and abnormal ECG datasets.
   - Visualize the distribution of labels and basic statistics.

2. **Data Preprocessing**:
   - Handle missing values using `SimpleImputer`.
   - Normalize the data.
   - Split the dataset into training and testing sets.

3. **Feature Engineering**:
   - Perform PCA to reduce the dimensionality of the data.

4. **Model Training**:
   - Train a neural network using TensorFlow/Keras.

5. **Evaluation**:
   - Evaluate the model's performance using accuracy, precision, recall, and F1-score.

6. **Visualization**:
   - Create interactive visualizations using Plotly to explore the data and model results.

## Results

The project aims to achieve high accuracy in classifying ECG signals as normal or abnormal. The results will be evaluated using various metrics, and the performance of the model will be visualized using interactive plots.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.



---

For any questions or further information, please contact the project maintainer.
