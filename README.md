# K-Nearest Neighbors

This project is a practice to understand the internal workings of the KNN algorithm, studied at the University of Guanajuato in the Artificial Intelligence class and in the IBM Machine Learning certification course to achieve certification. During the tests, the performance of my own implementation was compared with that of sklearn.

- You can have a look at the KNN algorithm code [here](src/knn_algorithm.py).
- Check the KNN model created [here](knn_model.ipynb).


## Whats is the KNN algorithm?
The k-nearest neighbors (KNN) algorithm is a non-parametric supervised learning classifier that uses proximity to make classifications or predictions about the grouping of an individual data point. It is one of the most popular and straightforward classifiers for classification and regression used in machine learning today.

### Functioning in Classification

For classification problems, KNN assigns a class label based on the most frequent label among the nearest neighbors, often referred to as "majority vote." This term is commonly used even though it technically implies more than 50% of the votes, which is mainly relevant for binary classification. In multi-class scenarios, a label can be assigned with a vote greater than the proportion of classes, such as 25% for four categories.

<img src="img/img-01.png" alt="KNN distances image" width="80%" />

### Distance Metrics

**Euclidean Distance (p=2):** Measures straight-line distance between points.

<img src="img/img-02.png" alt="Euclidean distance formula" width="50%" />

**Manhattan Distance (p=1):** Measures absolute difference between points, visualized on a grid.

<img src="img/img-03.png" alt="Manhattan distance formula" width="50%" />

**Minkowski Distance:** Generalized form of Euclidean and Manhattan distances, with parameter p.

<img src="img/img-04.png" alt="Minkowski distance formula" width="50%" />

**Hamming Distance:** Used for boolean or string vectors, counts mismatches.

<img src="img/img-05.png" alt="Hamming distance formula" width="50%" />

### Define K value

The value of k in KNN determines how many neighbors are considered for classifying a query point. For example, if k=1, the instance is classified based on its nearest neighbor. Choosing k is a balance between overfitting and underfitting. Lower k values may have high variance but low bias, while higher k values may have high bias but low variance. The optimal k depends on the data, with higher k values often better for noisy data. An odd k value is recommended to avoid ties, and cross-validation can help find the best k.


* [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering))

## Model Applications

The KNN algorithm has been used in a variety of applications, primarily within classification. Some of these use cases include:

- **Data Preprocessing**: KNN can estimate missing values in datasets through missing data imputation.
- **Recommendation Engines**: KNN uses clickstream data to provide automatic content recommendations. However, it may not be optimal for larger datasets due to scaling issues. [Research](https://www.researchgate.net/publication/267572060_Automated_Web_Usage_Data_Mining_and_Recommendation_System_using_K-Nearest_Neighbor_KNN_Classification_Method) shows its application in user behavior analysis.
- **Finance**: KNN helps assess loan risks and creditworthiness, and is used in stock market forecasting, foreign exchange rates, trading futures, and money laundering analyses. [Paper](https://iopscience.iop.org/article/10.1088/1742-6596/1025/1/012114/pdf) and [journal](https://www.ijera.com/papers/Vol3_issue5/DI35605610.pdf) provide detailed insights.
- **Healthcare**: KNN predicts risks of heart attacks and prostate cancer by calculating probable gene expressions.
- **Pattern Recognition**: KNN is used in text classification and digit recognition, aiding in identifying handwritten numbers on forms or envelopes. [Research](https://www.researchgate.net/profile/D-Adu-Gyamfi/publication/332880911_Improved_Handwritten_Digit_Recognition_using_Quantum_K-Nearest_Neighbor_Algorithm/links/5d77dca692851cacdb30c14d/Improved-Handwritten-Digit-Recognition-using-Quantum-K-Nearest-Neighbor-Algorithm.pdf) highlights its effectiveness.

## Advantages and Disadvantages

| Advantages | Disadvantages |
|------------|---------------|
| **Easy to Implement**: Due to its simplicity and accuracy, KNN is one of the first classifiers a new data scientist will learn. | **Doesn't Scale Well**: Since KNN is a lazy algorithm, it requires more memory and data storage compared to other classifiers. This can be costly both in terms of time and money. |
| **Easily Adaptable**: As new training samples are added, the algorithm adjusts to account for any new data since all training data is stored in memory. | **Curse of Dimensionality**: The KNN algorithm tends to fall victim to the curse of dimensionality, meaning it doesn't perform well with high-dimensional data inputs. |
| **Few Hyperparameters**: KNN only requires a value for k and a distance metric, which is low compared to other machine learning algorithms. | **Prone to Overfitting**: Due to the "curse of dimensionality", KNN is also more prone to overfitting. Lower values of k may overfit the data, while higher values of k tend to "smooth out" prediction values. |


## References

- [What is the k-nearest neighbors algorithm? | IBM](https://www.ibm.com/topics/knn)
