# Student Performance Prediction

## About the Project
This project focuses on analyzing student performance and predicting performance levels using Python and Machine Learning. It explores the relationship between study habits, previous scores, educational support, and the learning environment with students' exam scores.

## Objectives
1. To analyze the distribution of students' exam scores.
2. To study the relationship between study habits and exam performance.
3. To analyze the relationship between previous scores and current exam scores.
4. To examine educational support factors related to exam performance.
5. To investigate learning environment factors associated with exam performance.

## Dataset
- **Dataset Name:** Student Performance Factors
- **Source:** Kaggle
- **Target Variable:** `Exam_Score`

The dataset contains information about students' study hours, attendance, sleep, previous scores, parental involvement, teacher quality, and other factors.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Data Analysis
The project includes:
- Data exploration and summary statistics
- Missing value handling and duplicate removal
- Exam score distribution analysis
- Scatter plots and correlation analysis
- Comparison of average exam scores across different factors

## Machine Learning Models
Four classification models are used:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Random Forest Classifier

The models are evaluated using accuracy, precision, recall, and macro F1-score. The best model is selected based on the macro F1-score.

**Note:** Performance categories are defined for this project and do not represent official academic grades.

## How to Run the Project

1. Download or clone this repository.
2. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Keep `StudentPerformanceFactors.csv` in the same folder as the Python script.
4. Open the Python script in Spyder or another Python IDE.
5. Run the script to view the analysis, graphs, and model evaluation results.

## Project Structure

```text
Student-Performance-Prediction/
├── StudentPerformanceFactors.csv
├── student_performance_prediction_simple_separate_models.py
└── README.md
```

## Conclusion
This project uses exploratory data analysis and machine learning to study patterns in student exam performance. It compares four classification models and identifies the best-performing model based on the macro F1-score.

## Author
**Anitya Tripathi**
