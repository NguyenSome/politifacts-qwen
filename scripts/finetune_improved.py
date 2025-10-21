"""
Improved fine-tuning script for Qwen models with industry best practices.

This module provides a robust, production-ready implementation for fine-tuning
Qwen language models using LoRA (Low-Rank Adaptation) for fact-checking tasks.
"""

import logging
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import f1_score, cohen_kappa_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader

import paths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore", message="The following generation flags are not valid")


class Config:
    """Configuration management with validation."""
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize configuration from YAML file."""
        self.config_path = Path(config_path)
        self._load_config()
        self._validate_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _validate_config(self) -> None:
        """Validate required configuration keys."""
        required_keys = ['model_name', 'max_length', 'data_path']
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate MLflow configuration if enabled
        if self.data.get('mlflow', {}).get('enable', False):
            mlflow_keys = ['tracking_uri', 'exp_name', 'run_name']
            for key in mlflow_keys:
                if key not in self.data['mlflow']:
                    raise ValueError(f"Missing required MLflow configuration key: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.data.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.data[key]


class ModelManager:
    """Manages model and tokenizer initialization and configuration."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._dtype = self._get_optimal_dtype()
    
    def _get_optimal_dtype(self) -> torch.dtype:
        """Determine optimal data type based on hardware capabilities."""
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif torch.cuda.is_available():
            return torch.float16
        else:
            return torch.float32
    
    def initialize_model(self) -> Tuple[Any, Any]:
        """Initialize model and tokenizer with LoRA configuration."""
        try:
            logger.info(f"Loading model: {self.config['model_name']}")
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'],
                trust_remote_code=True,
                torch_dtype=self._dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.get('lora_r', 8),
                lora_alpha=self.config.get('lora_alpha', 32),
                lora_dropout=self.config.get('lora_dropout', 0.05),
                bias="none",
                target_modules=self.config.get(
                    'target_modules',
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                ),
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_name'],
                use_fast=True,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model and tokenizer initialized successfully")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise


class DataProcessor:
    """Handles data loading, preprocessing, and tokenization."""
    
    def __init__(self, config: Config, tokenizer: Any):
        self.config = config
        self.tokenizer = tokenizer
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> str:
        """Create the prompt template for fact-checking."""
        return (
            "Classify the following statement with one label only, "
            "chosen from: pants-fire, false, mostly-false, half-true, mostly-true, true.\n\n"
            "Statement: {statement}\n"
            "Answer with only the label, nothing else:\n"
        )
    
    def tokenize_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize a single example with proper padding and attention masking.
        
        Args:
            example: Dictionary containing 'statement' and 'verdict' keys
            
        Returns:
            Tokenized example with input_ids, labels, attention_mask, and span indices
        """
        try:
            # Create prompt and target
            prompt = self.prompt_template.format(statement=example["statement"])
            target = " " + str(example["verdict"]).strip()
            
            # Tokenize prompt and target
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
            
            max_length = self.config["max_length"]
            
            # Handle length constraints
            if len(prompt_ids) + len(target_ids) > max_length:
                keep_prompt = max(0, max_length - len(target_ids))
                prompt_ids = prompt_ids[-keep_prompt:]
                if keep_prompt == 0:
                    target_ids = target_ids[:max_length]
            
            # Combine input
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids[:]
            
            # Pad sequences
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [self.tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len
            
            # Create attention mask
            attention_mask = [1] * (max_length - pad_len) + [0] * pad_len
            
            # Store span indices for evaluation
            y_start = len(prompt_ids)
            y_end = len(prompt_ids) + len(target_ids)
            
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "y_start": y_start,
                "y_end": y_end
            }
            
        except Exception as e:
            logger.error(f"Error tokenizing example: {e}")
            raise
    
    def load_and_process_data(self, data_path: Union[str, Path]) -> DatasetDict:
        """
        Load and process the training data.
        
        Args:
            data_path: Path to the training data file
            
        Returns:
            Processed dataset dictionary with train/validation splits
        """
        try:
            logger.info(f"Loading data from: {data_path}")
            
            # Load dataset
            dataset = load_dataset("json", data_files=str(data_path))
            
            # Create train/validation split
            split_dataset = dataset["train"].train_test_split(
                test_size=self.config.get('validation_split', 0.15),
                seed=self.config.get('seed', 42)
            )
            
            # Create DatasetDict
            data = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
            })
            
            # Tokenize all examples
            logger.info("Tokenizing dataset...")
            tokenized_data = data.map(
                self.tokenize_example,
                remove_columns=data["train"].column_names,
                desc="Tokenizing"
            )
            
            logger.info(f"Dataset processed: {len(tokenized_data['train'])} train, {len(tokenized_data['validation'])} validation")
            return tokenized_data
            
        except Exception as e:
            logger.error(f"Error loading and processing data: {e}")
            raise


class MetricsCalculator:
    """Handles metric calculation for model evaluation."""
    
    def __init__(self, tokenizer: Any, eval_dataset: Dataset):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
    
    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Compute accuracy metrics for the evaluation predictions.
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary containing accuracy metrics
        """
        try:
            preds, labels = eval_pred
            
            # Handle different prediction formats
            if preds.ndim == 3:
                pred_ids = preds.argmax(-1)
            else:
                pred_ids = preds
            
            correct = 0
            total = len(pred_ids)
            
            for i in range(total):
                # Get span indices
                ys = int(self.eval_dataset[i]["y_start"])
                ye = int(self.eval_dataset[i]["y_end"])
                
                # Extract predicted and true text
                pred_text = self.tokenizer.decode(
                    pred_ids[i, ys:ye], 
                    skip_special_tokens=True
                ).strip().lower()
                
                true_ids = np.array(labels[i, ys:ye])
                true_ids = true_ids[true_ids != -100]
                true_text = self.tokenizer.decode(
                    true_ids, 
                    skip_special_tokens=True
                ).strip().lower()
                
                # Check for exact match
                if pred_text == true_text and len(true_text) > 0:
                    correct += 1
            
            accuracy = correct / max(1, total)
            return {"accuracy": accuracy}
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {"accuracy": 0.0}


class TrainingManager:
    """Manages the training process and MLflow integration."""
    
    def __init__(self, config: Config):
        self.config = config
        self.trainer = None
    
    def setup_mlflow(self) -> None:
        """Setup MLflow tracking if enabled."""
        if self.config.get('mlflow', {}).get('enable', False):
            try:
                mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
                mlflow.set_experiment(self.config['mlflow']['exp_name'])
                logger.info("MLflow tracking enabled")
            except Exception as e:
                logger.warning(f"Failed to setup MLflow: {e}")
    
    def train_model(
        self, 
        model: Any, 
        tokenizer: Any, 
        train_data: Dataset, 
        eval_data: Dataset
    ) -> Any:
        """
        Train the model with the provided data.
        
        Args:
            model: The model to train
            tokenizer: The tokenizer
            train_data: Training dataset
            eval_data: Evaluation dataset
            
        Returns:
            Trained trainer object
        """
        try:
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=self.config.get('output_dir', './outputs'),
                num_train_epochs=self.config.get('num_epochs', 1),
                per_device_train_batch_size=self.config.get('train_batch_size', 2),
                per_device_eval_batch_size=self.config.get('eval_batch_size', 2),
                learning_rate=self.config.get('learning_rate', 5e-5),
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_steps=self.config.get('logging_steps', 10),
                save_total_limit=self.config.get('save_total_limit', 2),
                bf16=self.config.get('use_bf16', False),
                fp16=self.config.get('use_fp16', False),
                report_to=["mlflow"] if self.config.get('mlflow', {}).get('enable', False) else [],
                run_name=self.config.get('mlflow', {}).get('run_name', 'default_run'),
                remove_unused_columns=True,
                seed=self.config.get('seed', 42),
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                evaluation_strategy="epoch",
            )
            
            # Setup metrics calculator
            metrics_calc = MetricsCalculator(tokenizer, eval_data)
            
            # Create trainer
            self.trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=eval_data,
                tokenizer=tokenizer,
                compute_metrics=metrics_calc.compute_metrics,
            )
            
            # Add early stopping callback
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.config.get('early_stopping_patience', 3)
            )
            self.trainer.add_callback(early_stopping)
            
            # Start MLflow run if enabled
            if self.config.get('mlflow', {}).get('enable', False):
                with mlflow.start_run(run_name=self.config['mlflow']['run_name']):
                    logger.info("Starting training with MLflow tracking...")
                    self.trainer.train()
                    metrics = self.trainer.evaluate()
                    mlflow.log_metrics({f"final_{k}": v for k, v in metrics.items()})
            else:
                logger.info("Starting training...")
                self.trainer.train()
            
            logger.info("Training completed successfully")
            return self.trainer
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Main training pipeline."""
    try:
        # Load configuration
        config = Config(paths.CONFIG_DIR / "eval.yaml")
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Initialize components
        model_manager = ModelManager(config)
        training_manager = TrainingManager(config)
        
        # Setup MLflow if enabled
        training_manager.setup_mlflow()
        
        # Initialize model and tokenizer
        model, tokenizer = model_manager.initialize_model()
        
        # Process data
        data_processor = DataProcessor(config, tokenizer)
        data_path = paths.DATA_DIR / "train.json"
        processed_data = data_processor.load_and_process_data(data_path)
        
        # Train model
        trainer = training_manager.train_model(
            model=model,
            tokenizer=tokenizer,
            train_data=processed_data["train"],
            eval_data=processed_data["validation"]
        )
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
