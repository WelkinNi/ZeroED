# ZeroED
**ZeroED** is a hybrid zero-shot error detection system for tabular data, within the reasoning ability of Large language model (LLM).  This is the respitory for the paper "ZeroED: Hybrid Zero-Shot Error Detection with Large Language Model Reasoning". 

### Initialization
To set up the environment and get started with ZeroED, follow these steps:

   ```
   # Create a new environment with Python 3.10
   conda create -n zeroed python=3.10
   conda activate zeroed

   # Clone the ZeroED repository
   git clone https://github.com/WelkinNi/ZeroED.git
   cd ZeroC
   ```

### Dataset
All the data used in this work can be found in `datasets` folder.

### Required configuaration 
`API related`: set in utility.py, including model_name, base_url, api_key

`model config`: set in run_config.yaml

Raha is from its original repository: https://github.com/BigDaMa/raha

