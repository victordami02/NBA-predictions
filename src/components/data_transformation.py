import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            # Load your dataset
            df = pd.read_csv("notebook/Data/nbaallelo.csv")  # Replace 'your_dataset.csv' with your actual data file path

            # Data transformation: Map 'W' to 1 and 'L' to 0 in the 'game_result' column
            df['game_result'] = df['game_result'].map({'W': 1, 'L': 0})
            df['game_location'] = df['game_location'].map({'H': 1, 'A': 0})
            df['date_game'] = pd.to_datetime(df['date_game'])

            # Define columns to be removed
            removed_columns = ['gameorder', 'game_id', 'lg_id', 'year_id', 'date_game',
                            'seasongame', 'team_id', 'fran_id', 'pts', 'opp_id', 'opp_fran', 'opp_pts', 'game_result']

            # Select columns that are not in the 'removed_columns' list
            selected_columns = df.columns[~df.columns.isin(removed_columns)]

            # Create a preprocessing pipeline
            preprocessor = Pipeline([
                ('scaler', MinMaxScaler()),  # Apply Min-Max scaling
            ])


            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)    
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="game_result"

            input_feature_train_df=train_df[['elo_i', 'elo_n', 'win_equiv', 'opp_elo_n', 'forecast']]
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df[['elo_i', 'elo_n', 'win_equiv', 'opp_elo_n', 'forecast']]
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)    
        


        




