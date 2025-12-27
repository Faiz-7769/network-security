from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

#configuration data_validation config
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp #for data drift report
import pandas as pd
import os,sys
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def validate_no_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            schema_columns = self.schema_config["columns"]
            expected_columns = len(schema_columns)
            actual_columns = len(dataframe.columns)

            logging.info(f"Required number of columns as per schema: {expected_columns}")
            logging.info(f"Dataframe has columns: {actual_columns}")

            if actual_columns == expected_columns:
                return True

            logging.error(
                f"Column count mismatch. Expected: {expected_columns}, Found: {actual_columns}"
            )
            return False

        except Exception as e:
            raise NetworkSecurityException(e, sys)

        
    def detect_data_drift(self,base_df,curretn_df,threshold=0.05)->bool:
        try:
            status =True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = curretn_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found =False
                else:
                    is_found = True
                    status = False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_value":is_found
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            #create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content = report)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self.schema_config["numerical_columns"]

            missing_columns = []
            non_numerical_columns = []

            for col in numerical_columns:
                # 1. Check column existence
                if col not in dataframe.columns:
                    missing_columns.append(col)
                    continue

                # 2. Check if column is numeric
                if not pd.api.types.is_numeric_dtype(dataframe[col]):
                    non_numerical_columns.append(col)

            if missing_columns:
                logging.error(f"Missing numerical columns: {missing_columns}")

            if non_numerical_columns:
                logging.error(
                    f"Columns expected to be numeric but are not: {non_numerical_columns}"
                )

            return len(missing_columns) == 0 and len(non_numerical_columns) == 0

        except Exception as e:
            raise NetworkSecurityException(e, sys)

        
    def initiate_data_validation(self)-> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # read the data
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            #validate no. of columns

            status = self.validate_no_of_columns(dataframe=train_dataframe)
            if not status:
                raise Exception("Train dataframe column count mismatch")

            status = self.validate_no_of_columns(dataframe=test_dataframe)
            if not status:
                raise Exception("Test dataframe column count mismatch")

            # validate numerical columns
            status = self.validate_numerical_columns(train_dataframe)
            if not status:
                raise Exception("Train dataframe numerical column validation failed")

            status = self.validate_numerical_columns(test_dataframe)
            if not status:
                raise Exception("Test dataframe numerical column validation failed")


            #lets check data drift
            status = self.detect_data_drift(base_df=train_dataframe,curretn_df=test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path,index=False,header =True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path,index=False,header =True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)