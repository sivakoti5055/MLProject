import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from src.utils import saveObject
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocess_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj = DataTransformationConfig()

    def get_data_transformation_object(self,train_data):
        logging.info('Hi')
        try:
            numerical_features = train_data.select_dtypes(include=['int64','float64']).columns.to_list()
            categorical_features = train_data.select_dtypes(include=['object']).columns.to_list()

            logging.info('numerical and categorical features are successfull')

            numerical_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaling',StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps =[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoding',OneHotEncoder()),
                    ('scaling',StandardScaler(with_mean=False))
                ]
            )

            logging.info('numerical and categorical pipeline are successfull')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',numerical_pipeline,numerical_features),
                    ('cat pipeline',categorical_pipeline,categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    
    def initiate_data_transformer(self,train_data,test_data):
        # def __init__(self):
        #     self.preprocessing_obj = get_data_transformation_object(train_data)

        try:
            logging.info('Entered')
            train_arr = pd.read_csv(train_data)
            test_arr = pd.read_csv(test_data)

            input_train_arr = train_arr.drop(['math score'],axis=1)
            train_target = train_arr['math score']

            input_test_arr = test_arr.drop(['math score'],axis=1)
            test_target = test_arr['math score']

            logging.info('split X,Y done')

            preprocessing_obj = self.get_data_transformation_object(input_train_arr)
            
            train_data_preprocess = preprocessing_obj.fit_transform(input_train_arr)
            complete_train_data = np.c_[train_data_preprocess,np.array(train_target)]

            logging.info('train_data completely success full')

            test_data_preprocess = preprocessing_obj.transform(input_test_arr)
            complete_test_data = np.c_[test_data_preprocess,np.array(test_target)]

            logging.info('test data success full')

            saveObject(
                filepath= self.preprocessor_obj.preprocess_path,
                obj = preprocessing_obj

            )

            logging.info('success full')

            return(
                complete_train_data,
                complete_test_data,
                self.preprocessor_obj.preprocess_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)



