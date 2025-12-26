from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ =='__main__':
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_inegstion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_inegstion_config)
        logging.info("initiate the data ingestion")
        artifact = data_ingestion.initiate_data_ingestion()
        print(artifact)

    except Exception as e:
        raise NetworkSecurityException(e,sys)