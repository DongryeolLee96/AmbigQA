# CAmbigQA Baseline Models

We revise a few lines in QAData.py, QGData.py, and run.py for supporting our experiments.

We also add new py files like CQ_utils.py for our auto-conversion.


Run 'run_cq_inference.sh' file to generate the CQ by BART model.
Run 'run_cq_training.sh' file to train the CQ generation BART model.

Run 'Run run_cambigqa_inference.sh' file to inference the reader (BART) model for checking the clarification.
Run 'run_cambigqa_train.sh' file to train the reader (BART) model on CAmbigNQ dataset.
