# Virtual Internship
# Anomaly detection in time series data of manufacturing parts


The workflow for this project were as follows:

•	A synthetic dataset was generated taking the values of different parameters like spindle load, spindle speed, spindle temperature, current, flow rate etc. at timestamp value of 1 second following the given parameter range. (As the original time series data from machines were not available due the ongoing pandemic situation.) The code for this can be found in #Final_intern_data.ipynb file.

•	Next the time series data was stored in MongoDB. This code can be found in #MongoDB Intern 31.07.20.ipynb file.


•	Then the analysis of the data was done as follows:

	At first the data was loaded as pandas dataframe from MongoDB.

	Then certain columns were cleaned and the columns containing null values were filled with mean values mostly.


	Next the feature scaling was done using min-max scaler, robust scaler and standard scaler but then proceeded with minmax scaler mostly because the distribution plot of the features did not show a normal distribution.

	Then formatting of data was done to prepare the final dataset and many columns were removed like program number, part count, part ID, running tool number, tool changed and material.


	Then many plots like line plot, box plot, heat map, Pearson and Spearman correlation coefficient  were done using the matplotlib and seaborn library to get a better intuition of the data. The box plot showed many outliers were present in many feature columns and those showing outlier % greater than 45% were removed. These features include spindle speed, cutting speed, tolerance, diameter, flow rate.

	Then proceeded further with features like spindle load, current, spindle temperature, feed, vibration x, vibration z and vibration spindle.


	As the labels were assigned randomly so at first tried to proceed with the clustering techniques which is an unsupervised learning technique so labels are not required. 

	PCA and t-SNE were used for dimensionality reduction for better visualization of clusters.


	Then to determine the optimal number of clusters for the clustering algorithm used an elbow plot which showed the elbow point to be at 3 and silhouette score graph which showed peak at 2 so, 2 was used as the value of n_cluster.

	Then different clustering techniques like k-means clustering, mini-batch kmeans, dbscan, gaussian mixture, Bayesian gaussian mixture, agglomerative clustering and affinity propagation were used. The implementation was done using sklearn library. These results were compared using Silhouette score, Calinski-Harabasz Index and Davies-Bouldin Index . These metrics were chosen because it did not require the knowledge of ground truth. The analysis result showed best performance by agglomerative clustering  on the basis of silhouette score. The clusters were non-overlapping and prominent for most of the algorithms. But the issue was that the two clusters were of almost equal size. 

	Then the issue of imbalanced dataset were handled using random oversampling of the minority class, random downsampling of the majority class, Tomek link, SMOTE and combination of SMOTE and Tomek link. Finally used the combination of SMOTE and Tomek link. Then the dataset was divided into train, test and validation dataset. Then a simple deep learning sequential model was built using keras but the result showed that it was a case of over-fitting. So then k-fold cross-validation and repeated k-fold cross-validation was used to address this issue.  Then one conv1D layer followed by one max pool layer was introduced and the result of which was fed into a LSTM layer. The accuracy was good for the testing set. Currently thinking of improving this model's accuracy and also to apply autoencoder as an approach here.


	Finally, the classification techniques were used like Decision tree, SVM, and ensemble methods like random forest, adaboost and gradient boosting. For hyperparameter tuning GridSearch CV was used and the results were compared with the help of precision, recall, f1, confusion matrix, etc. For repeated k-fold cross validation with n-split of 2 and n-repeats of 2 four combinations were obtained and among them SVM always performed better than other classifiers based on f1 score.

All this code can be found in #ML_final_29_07_20.ipynb file.

