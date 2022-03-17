from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

class Model_Finder:
    """
                This class is used to find the best model with high accuracy and AUC score.

    """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf = RandomForestClassifier()
        self.knn = KNeighborsClassifier()
        self.xgb = XGBClassifier()


    def get_best_params_for_random_forest(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception


        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_distributions = {
                                        'n_estimators': [40,55,75,80,100,120],
                                        'criterion': ['gini', 'entropy'],
                                        'max_depth': [int(x) for x in np.linspace(10, 100,8)],
                                        'min_samples_split': [2,3,4,5,6,7],
                                        'min_samples_leaf': [5,6,7,8,9],
                                        'max_leaf_nodes': [29,30,32,33]
                                        }

            #Creating an object of the Grid Search class
            self.random_search = RandomizedSearchCV(estimator=self.clf, param_distributions = self.param_distributions, cv=5, verbose=3)
            #finding the best parameters
            self.random_search.fit(train_x, train_y)

            #extracting the best parameters
            self.n_estimators = self.random_search.best_params_['n_estimators']
            self.min_samples_split = self.random_search.best_params_['min_samples_split']
            self.min_samples_leaf = self.random_search.best_params_['min_samples_leaf']
            self.max_leaf_nodes = self.random_search.best_params_['max_leaf_nodes']
            self.max_depth = self.random_search.best_params_['max_depth']
            self.criterion = self.random_search.best_params_['criterion']


            #creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_split=self.n_estimators,
                                           min_samples_leaf=self.min_samples_leaf, max_leaf_nodes=self.max_leaf_nodes,
                                           max_depth=self.max_depth, criterion=self.criterion)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.random_search.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_KNN(self, train_x, train_y):
        """
                                                Method Name: get_best_params_for_KNN
                                                Description: get the parameters for KNN Algorithm which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception

               """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_Ensembled_KNN method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_knn = {
                'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
                'leaf_size' : [10,17,24,28,30,35],
                'n_neighbors':[4,5,8,10,11],
                'p':[1,2]
            }

            # Creating an object of the Grid Search class
            self.search = RandomizedSearchCV(self.knn, param_distributions=self.param_grid_knn, verbose=3,
                                     cv=5)
            # finding the best parameters
            self.search.fit(train_x, train_y)

            # extracting the best parameters
            self.algorithm = self.search.best_params_['algorithm']
            self.leaf_size = self.search.best_params_['leaf_size']
            self.n_neighbors = self.search.best_params_['n_neighbors']
            self.p  = self.search.best_params_['p']

            # creating a new model with the best parameters
            self.knn = KNeighborsClassifier(algorithm=self.algorithm, leaf_size=self.leaf_size, n_neighbors=self.n_neighbors,p=self.p,n_jobs=-1)
            # training the mew model
            self.knn.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'KNN best params: ' + str(
                                       self.search.best_params_) + '. Exited the KNN method of the Model_Finder class')
            return self.knn
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in knn method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'KNN Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()

    # def get_best_params_for_xgboost(self,train_x,train_y):
    #
    #     """
    #                                     Method Name: get_best_params_for_xgboost
    #                                     Description: get the parameters for XGBoost Algorithm which give the best accuracy.
    #                                                  Use Hyper Parameter Tuning.
    #                                     Output: The model with the best parameters
    #                                     On Failure: Raise Exception
    #
    #
    #     """
    #     self.logger_object.log(self.file_object,
    #                            'Entered the get_best_params_for_xgboost method of the Model_Finder class')
    #     try:
    #         # initializing with different combination of parameters
    #         self.param_grid_xgboost = {
    #                              "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    #                              "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    #                              "min_child_weight": [1, 3, 5, 7],
    #                              "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    #                              "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
    #                             }
    #
    #         self.random = RandomizedSearchCV(XGBClassifier(objective='binary:logistic'), param_distributions=self.param_grid_xgboost,
    #                                       verbose=3, cv=5)
    #         # finding the best parameters
    #         self.random.fit(train_x, train_y)
    #
    #         # extracting the best parameters
    #
    #         self.learning_rate = self.random.best_params_["learning_rate"]
    #         self.max_depth = self.random.best_params_["max_depth"]
    #         self.min_child_weight = self.random.best_params_["min_child_weight"]
    #         self.gamma = self.random.best_params_["gamma"]
    #         self.colsample_bytree = self.random.best_params_["colsample_bytree"]
    #
    #
    #         # creating a new model with the best parameters
    #         self.xgb = XGBClassifier(max_depth=self.max_depth, learning_rate=self.learning_rate,
    #                                  min_child_weight=self.min_child_weight, gamma=self.gamma,
    #                             colsample_bytree=self.colsample_bytree)
    #
    #
    #         # training the mew model
    #         self.xgb.fit(train_x, train_y)
    #         self.logger_object.log(self.file_object,
    #                                f'XGBoost best params: {self.random.best_params_} \t Exited the get_best_params_for_xgboost method of the Model_Finder class')
    #
    #         return self.xgb
    #
    #     except Exception as e:
    #         self.logger_object.log(self.file_object,
    #                                'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
    #                                    e))
    #         self.logger_object.log(self.file_object,
    #                                'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
    #         raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception


                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for KNN
        try:
            self.knn = self.get_best_params_for_KNN(train_x,train_y)
            self.y_pred_knn = self.knn.predict(test_x)
            self.prediction_knn = self.knn.predict_proba(test_x) # Predictions using the KNN Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.knn_score = accuracy_score(test_y, self.y_pred_knn)
                self.logger_object.log(self.file_object, 'Accuracy for KNN:' + str(self.knn_score))  # Log AUC
            else:
                self.knn_score = roc_auc_score(test_y, self.prediction_knn, multi_class='ovr') # AUC for KNN
                self.logger_object.log(self.file_object, 'AUC for KNN:' + str(self.knn_score)) # Log AUC

            # create best model for Random Forest
            self.random_forest = self.get_best_params_for_random_forest(train_x,train_y)
            self.y_pred_rf = self.random_forest.predict(test_x)
            self.prediction_random_forest = self.random_forest.predict_proba(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(test_y, self.y_pred_rf)
                self.logger_object.log(self.file_object, 'Accuracy for RandomForest:' + str(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest,multi_class='ovr') # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for RandomForest:' + str(self.random_forest_score))

            #comparing the two models
            if(self.random_forest_score <  self.knn_score):
                return 'KNN', self.knn
            else:
                return 'RandomForest', self.random_forest

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

    #
    # def get_best_model(self,train_x,train_y,test_x,test_y):
    #     """
    #                                             Method Name: get_best_model
    #                                             Description: Find out the Model which has the best AUC score.
    #                                             Output: The best model name and the model object
    #                                             On Failure: Raise Exception
    #
    #
    #                                     """
    #     self.logger_object.log(self.file_object,
    #                            'Entered the get_best_model method of the Model_Finder class')
    #
    #     try:
    #         self.xgb = self.get_best_params_for_xgboost(train_x,train_y)
    #         self.prediction_xgb = self.xgb.predict_proba(test_x) # Predictions using the KNN Model
    #
    #         if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
    #             self.xgb_score = accuracy_score(test_y, self.prediction_xgb)
    #             self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgb_score))  # Log AUC
    #         else:
    #             self.xgb_score = roc_auc_score(test_y, self.prediction_xgb, multi_class='ovr') # AUC for KNN
    #             self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgb_score)) # Log AUC
    #
    #         # create best model for Random Forest
    #         self.random_forest = self.get_best_params_for_random_forest(train_x,train_y)
    #         self.prediction_random_forest = self.random_forest.predict_proba(test_x) # prediction using the Random Forest Algorithm
    #
    #         if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
    #             self.random_forest_score = accuracy_score((test_y),self.prediction_random_forest)
    #             self.logger_object.log(self.file_object, 'Accuracy for RandomForest:' + str(self.random_forest_score))
    #         else:
    #             self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest,multi_class='ovr') # AUC for Random Forest
    #             self.logger_object.log(self.file_object, 'AUC for RandomForest:' + str(self.random_forest_score))
    #
    #         #comparing the two models
    #         if(self.random_forest_score <  self.xgb_score):
    #             return 'XGBoost', self.xgb
    #         else:
    #             return 'RandomForest', self.random_forest
    #
    #     except Exception as e:
    #         self.logger_object.log(self.file_object,
    #                                'Exception occurred in get_best_model method of the Model_Finder class. Exception message:  ' + str(
    #                                    e))
    #         self.logger_object.log(self.file_object,
    #                                'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
    #         raise Exception()

