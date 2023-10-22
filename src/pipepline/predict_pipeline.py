import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        elo_i: float,
        elo_n: float,
        win_equiv: float,
        opp_elo_n: float,
        forecast: float):

        self.elo_i = elo_i

        self.elo_n = elo_n

        self.win_equiv = win_equiv

        self.opp_elo_n = opp_elo_n

        self.forecast = forecast



    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "elo_i": [self.elo_i],
                "elo_n": [self.elo_n],
                "win_equiv": [self.win_equiv],
                "opp_elo_n": [self.opp_elo_n],
                "forecast": [self.forecast],
                
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)