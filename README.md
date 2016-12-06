# CS256_pca_practice_assmt
Following was learned from this assignment:-

1. svd.fit(X,[,y]) can be used to generate a model.
2. svd.transform(x) can be used to transform the data to the model

Following are the places where changes have been implemented:-

Line 55-60 :- Here we have reduced dimensionality of both the training and testing data.

Line 209-277 :- Here code is written to extract test data from BRAF_test_moe.csv. test data is named as X2,y2.

Line 297-318 :- Here code is written to train and test the original as well as reduced dimesion data on SVM for prediction.


Output:- The dimension are reduced from 355 to 35(90% reduction)
and with prediction accuracy of 96.66%.
Snippet of output below:-

Original Train Data Dimension:-
(243L, 355L)
Original Test Data Dimension:-
(60L, 355L)

Reduced Train Data Dimension:-
(243L, 35L)
Reduced Test Data Dimension:-
(60L, 35L)

clf.score(X_test_transformed, y_test) = 96.66%
