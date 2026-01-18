# from pathlib import Path
# import mlflow
# import dataclasses

# @dataclass
# class TrainingConfig:
#     """Centralized configuration for training runs"""
#     experiment_name: str = "MNIST_Model_Versioning"
#     model_name: str = "mnist_classifier"
#     tracking_uri: str = "./mlruns"
    
#     # Training hyperparameters
#     batch_size: int = 32
#     learning_rate: float = 0.001
#     epochs: int = 5
#     optimizer_type: str = "adam"  # adam, sgd
#     dropout_rate: float = 0.2
#     hidden_dim: int = 128
    
#     # Data
#     data_dir: str = "./data"
    
#     # Device
#     device: str = "cuda"
    
#     def __post_init__(self):
#         """Create directories if they don't exist"""
#         Path(self.data_dir).mkdir(exist_ok=True)
#         Path(self.tracking_uri).mkdir(exist_ok=True)

# class MLFlowManager:
#     """Handles MLflow experiment tracking and model registry"""
    
#     def __init__(self, config: TrainingConfig):
#         self.config = config
#         # Set tracking URI (local filesystem)
#         mlflow.set_tracking_uri(f"file:{config.tracking_uri}")
#         # Create experiment if it doesn't exist
#         mlflow.set_experiment(config.experiment_name)
    
#     def log_training_run(
#         self,
#         model: nn.Module,
#         config: TrainingConfig,
#         history: Dict,
#         run_name: str
#     ) -> str:
#         """Log complete training run to MLflow"""
        
#         with mlflow.start_run(run_name=run_name):
#             # ========== Log Parameters ==========
#             params = {
#                 "batch_size": config.batch_size,
#                 "learning_rate": config.learning_rate,
#                 "epochs": config.epochs,
#                 "optimizer": config.optimizer_type,
#                 "dropout_rate": config.dropout_rate,
#                 "hidden_dim": config.hidden_dim,
#                 "device": config.device,
#             }
#             mlflow.log_params(params)
#             logger.info(f"Logged {len(params)} parameters")
            
#             # ========== Log Metrics ==========
#             final_test_accuracy = history["test_accuracy"][-1]
#             final_test_loss = history["test_loss"][-1]
            
#             mlflow.log_metrics({
#                 "final_accuracy": final_test_accuracy,
#                 "final_loss": final_test_loss,
#                 "best_accuracy": max(history["test_accuracy"])
#             })
#             logger.info(f"Final Test Accuracy: {final_test_accuracy:.2f}%")
            
#             # ========== Log Model ==========
#             mlflow.pytorch.log_model(
#                 model,
#                 artifact_path="model",
#                 code_paths=[__file__]  # Log this script as artifact
#             )
#             logger.info("Model logged to MLflow")
            
#             # ========== Log Training History ==========
#             history_json = json.dumps(history, indent=2)
#             with open("training_history.json", "w") as f:
#                 f.write(history_json)
#             mlflow.log_artifact("training_history.json")
#             os.remove("training_history.json")
            
#             # ========== Log Config ==========
#             config_dict = asdict(config)
#             config_json = json.dumps(config_dict, indent=2)
#             with open("config.json", "w") as f:
#                 f.write(config_json)
#             mlflow.log_artifact("config.json")
#             os.remove("config.json")
            
#             logger.info("Artifacts logged to MLflow")
            
#             # Get run ID
#             run_id = mlflow.active_run().info.run_id
            
#         return run_id
    
#     def register_model(self, run_id: str, model_version_description: str = ""):
#         """Register trained model to MLflow Model Registry"""
        
#         model_uri = f"runs:/{run_id}/model"
        
#         result = mlflow.register_model(
#             model_uri=model_uri,
#             name=self.config.model_name,
#             await_registration_completion=True
#         )
        
#         logger.info(f"Model registered: {self.config.model_name} v{result.version}")
        
#         # Add description
#         client = mlflow.tracking.MlflowClient()
#         client.update_model_version(
#             name=self.config.model_name,
#             version=result.version,
#             description=model_version_description or f"Training run {run_id}"
#         )
        
#         return result.version
    
#     def transition_model_stage(self, version: int, stage: str):
#         """Transition model to different stage: Staging, Production, Archived"""
#         client = mlflow.tracking.MlflowClient()
#         client.transition_model_version_stage(
#             name=self.config.model_name,
#             version=str(version),
#             stage=stage
#         )
#         logger.info(f"Model {self.config.model_name} v{version} transitioned to {stage}")
    
#     def compare_versions(self) -> Dict:
#         """Compare all registered versions of the model"""
#         client = mlflow.tracking.MlflowClient()
        
#         try:
#             model = client.get_registered_model(self.config.model_name)
            
#             comparison = {}
#             for version in model.latest_versions:
#                 run = client.get_run(version.run_id)
#                 comparison[f"v{version.version}"] = {
#                     "stage": version.current_stage,
#                     "accuracy": run.data.metrics.get("final_accuracy", "N/A"),
#                     "loss": run.data.metrics.get("final_loss", "N/A"),
#                     "created_at": version.creation_timestamp
#                 }
            
#             return comparison
#         except Exception as e:
#             logger.warning(f"Could not fetch registered model info: {e}")
#             return {}