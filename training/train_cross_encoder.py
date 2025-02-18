"""
This script trains a cross encoder. A cross encoder 
is a kind of BERT model that takes a text pair
as input and outputs a label. Here, it learns to predict the labels: "yes": 0, "no": 1 across
query Q and document D pairs, where f(Q,D) = 0 iff D is relevant for Q.
We can either use the predicted labels, or relevance probability to re-rank
query results in search systems. It is trained using CE loss (vs. triplet loss
for bi-encoders).

Usage:
python training/train_cross_encoder.py
"""

import logging
import math
from datetime import datetime

import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Comment out if using an Apple M-series chip
# torch.mps.set_per_process_memory_fraction(0.0)

from  preprocessing.split_data import *

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEF1Evaluator, CESoftmaxAccuracyEvaluator
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.readers import InputExample


#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)
#### /print debug information to stdout

# Load Cranfield and generate splits
logger.info("Read Cranfield dataset")

merged_data             = load_cranfield()
train_data, test_data   = split_data(merged_data, test_queries=5)
train_data              = negative_mining(train_data)
size                    = math.floor(train_data.shape[0] * 0.25)
dev_ids                 = list(train_data.sample(int(size), random_state=42).query_id.values)
df_train                = train_data[train_data.query_id.isin(dev_ids)]
df_val                  = train_data[~train_data.query_id.isin(dev_ids)]

logger.info("Serializing Cranfield splits")

merged_data.to_json('data/cran-full.json',            index=False, orient='records', lines=True)
train_data.to_json('data/cran-train.json',            index=False, orient='records', lines=True)
test_data.to_json('data/cran-test.json',              index=False, orient='records', lines=True)
df_train.to_json('data/train-cross-enc-cran.json',    index=False, orient='records', lines=True)
df_val.to_json('data/validation-cross-enc-cran.json', index=False, orient='records', lines=True)

logger.info(f"Training on {df_train.shape} samples")
logger.info(f"Validation on {df_val.shape} samples")

# Read in the traininga and validation data
logger.info("Read train and validation datasets")

label2int = {"no": 0, "yes": 1}
train_samples = []
dev_samples = []

for _, row in df_train.iterrows():
        label_id = label2int[row["relevant"]]
        train_samples.append(InputExample(texts=[row["query"], row["text"][0:5000]], label=label_id))

for _, row in df_train.iterrows():
        label_id = label2int[row["relevant"]]
        dev_samples.append(InputExample(texts=[row["query"], row["text"][0:5000]], label=label_id))

# Training hyperparameters
train_batch_size = 24
num_epochs = 4
model_save_path = "models/training_cranfield-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define our CrossEncoder model. We use distilroberta-base as basis and setup it up to predict 2 labels
model = CrossEncoder("distilroberta-base", num_labels=len(label2int))

# We wrap train_samples, which is a list of InputExample, in a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# During training, we use CESoftmaxAccuracyEvaluator and CEF1Evaluator to measure the performance on the dev set
accuracy_evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_samples, name="Cran-dev")
#f1_evaluator = CEF1Evaluator.from_input_examples(dev_samples, name="Cran-dev")
evaluator = SequentialEvaluator([accuracy_evaluator])

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logger.info(f"Warmup-steps: {warmup_steps}")

# Train the model
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=100,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
)