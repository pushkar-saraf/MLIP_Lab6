import pytest
import pandas as pd
import numpy as np
from prediction_demo import data_preparation,data_split,train_model,eval_model

@pytest.fixture
def housing_data_sample():
    return pd.DataFrame(
      data ={
      'price':[13300000,12250000],
      'area':[7420,8960],
    	'bedrooms':[4,4],	
      'bathrooms':[2,4],	
      'stories':[3,4],	
      'mainroad':["yes","yes"],	
      'guestroom':["no","no"],	
      'basement':["no","no"],	
      'hotwaterheating':["no","no"],	
      'airconditioning':["yes","yes"],	
      'parking':[2,3],
      'prefarea':["yes","no"],	
      'furnishingstatus':["furnished","unfurnished"]}
    )

def test_data_preparation(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    # Target and datapoints has same length
    assert feature_df.shape[0]==len(target_series)

    #Feature only has numerical values
    assert feature_df.shape[1] == feature_df.select_dtypes(include=(np.number,np.bool_)).shape[1]

@pytest.fixture
def feature_target_sample(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    return (feature_df, target_series)


def test_data_split(feature_target_sample):
    return_tuple = data_split(*feature_target_sample)

    assert len(return_tuple) == 4, "data_split should return four elements (X_train, X_test, y_train, y_test)"

    X_train, X_test, y_train, y_test = return_tuple

    assert X_train.shape[0] > 0, "X_train should not be empty"
    assert X_test.shape[0] > 0, "X_test should not be empty"
    assert y_train.shape[0] > 0, "y_train should not be empty"
    assert y_test.shape[0] > 0, "y_test should not be empty"

    assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch between X_train and X_test"

    total_samples = feature_target_sample[0].shape[0]
    assert X_train.shape[0] + X_test.shape[0] == total_samples, "Train-test split does not sum up to total samples"
