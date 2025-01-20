# ToxicBench

This is the repository containig the code for the ToxicBench L46 project. It is organized to be run in any standard Python environment (and should run within Lightning.ai),you might need to install some dependencies manually, though (No package manager comes with this e.g. uv). A lot of boilerplate code and scaffolding has been moved to `metrics.py`, `upload_results.py`, `finetune.py`, and `dataset_processing`. The respecitve workflows for our two experimetns are as follows. For experiment 1, where we look at existing work:
1. `Evaluation_Existing_Work.ipynb` collects and processes all datasets, collects and modifies all models, and evaluates them, stores the results locally as pickles. Runs for a few hours, but partial progress is saved.
2. `Upload_Results.ipynb` transforms, aggregates and uploads these results to hugginface
3. `Generate_Plots.ipynb` pulls the data locally and creates seaborn plots
For the second experiment:
1. `Finetune_Ablation_Study.ipynb` finetunes DistilBERT-base model on the repsective dataset
2. `Evaluation_Finetune_mixed_dataset.ipynb` creates the mixed dataset and finetunes our classifier on this, evaluates all teh other similar to 1. of the first experiment. 
3. `Upload_Results.ipynb` and`Generate_Plots.ipynb` same as above
