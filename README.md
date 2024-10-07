# ![Icon](assets/icon.png) Better than Your Teacher: LLM Agents that Learn from Privileged AI Feedback

<!-- Paper link: []() -->

## Installation

### Create virtual environment

To set up the project, clone the repository and create a virtual environment:

```bash
cd leap-llm
pyenv virtualenv leap-llm
pyenv activate leap-llm
```

### Set up environment keys
Ensure you have a `.env` file with your OpenAI API key and organization ID:

```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_ORGANIZATION=your_openai_organization_id
```

### Set up external dependencies

To set up external environments like AlfWorld or WebShop, [go to external environment instructions](#external-environment-instructions).

### Install package

Install the required packages:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data generation

### Generate raw logs

#### AlfWorld logs

The script below will run human-collected logs for every AlfWorld game. This may take a while because loading an AlfWorld game is slow.
```bash
python scripts/dataproc/collect_logs_alfworld.py --config configs/training_alfworld.yaml
```

#### WebShop logs

The script below will read precollected logs from WebShop.
```bash
python scripts/dataproc/collect_logs_webshop.py --config configs/training_webshop.yaml
```

### Reasoning logs

To generate logs annotated with reasoning, use the following command:
```bash
python scripts/dataproc/annotate_reason.py --config configs/training_{environment}.yaml
```

### Extract Privileged State

#### Extract privileged state for AlfWorld
```bash
python scripts/dataproc/extract_privileged_state_from_logs.py --config configs/training_alfworld.yaml
```

#### Extract privileged state for WebShop
```bash
python scripts/dataproc/extract_privileged_state_webshop.py --config configs/training_webshop.yaml
```

## Iterative Training

Let's go through all the steps of a typical training iteration.

### 1. Roll out previous iteration model (skip for iter0)
First, take the previous model, roll it out in the environment, and collect trajectories in `data/{environment}/corrections/{iter_id-1}/rollout`:
```bash
python scripts/eval/eval_{environment}.py --training_config configs/training_{environment}.yaml --iter {iter_id-1}
```

### 2. Generate corrections on rollout trajectory (skip for iter0)
Next, invoke privileged on the rollouts to generate corrections in `data/{environment}/corrections/{iter_id-1}/correction`:
```bash
python scripts/dataproc/correct_student_trajectory.py --config configs/training_{environment}.yaml --iter {iter_id-1}
```

### 3. Create training data

To generate training data for the current iteration, run the following command:
```bash
python scripts/dataproc/create_training_data.py --config configs/training_{environment}.yaml --train_method {train_method} --iter {iter_id}
```

### 4. Train model

For SFT, run the following script corresponding to the correct environment and iteration:
```bash
bash bash/train-sft-alfworld-iterx.sh {iter_id}
```

For DPO, run the following script corresponding to the correct environment and iteration:
```bash
bash bash/train-dpo-alfworld-iterx.sh {iter_id}
```

## Evaluation

### Alfworld evaluation
Configure the agents you want to evaluate in `configs/eval_alfworld.yaml` and run the following script:
```bash
python scripts/eval/eval_alfworld.py --eval_config configs/eval_alfworld.yaml
```
It will create a folder in `data/eval/alfworld/` with the current datetime where logs and summary.csv will be saved.

### WebShop evaluation
First, ensure you are running the WebShop server in another terminal tab:
```bash
bash bash/run_webshop_server.sh
```

Configure the agents you want to evaluate in `configs/eval_webshop.yaml` and run the following script:
```bash
python scripts/eval/eval_webshop.py --eval_config configs/eval_webshop.yaml
```
It will create a folder in `data/eval/webshop/` with the current datetime where  logs and summary.csv will be saved.

## External environment instructions

### Setup AlfWorld
Clone AlfWorld from [AlfWorld github repository](https://github.com/alfworld/alfworld). Follow the instructions in its README to get the game files.

Create an env_assets folder and copy over data to `env_assets/alfworld`. Set the following environment variable:
```bash
export ALFWORLD_DATA=</path/to/env_assets/alfworld>
```

### Setup Webshop
Clone our fork of WebShop:
```bash
git clone https://github.com/sanjibanc/WebShop.git
```

Create a conda environment and activate it:
```bash
conda create -n leap_llm python=3.10
conda activate leap_llm
```

#### Install Pyserini

Follow the installation instructions for Pyserini [here](https://github.com/castorini/pyserini/blob/master/docs/installation.md)

If you are on Mac:
```bash
conda install wget -y
conda install -c conda-forge openjdk=21 maven -y
conda install -c conda-forge lightgbm nmslib -y
conda install -c pytorch faiss-cpu pytorch -y

pip install pyserini
```
If you are on Linux:
```bash
conda install -c conda-forge openjdk=21
pip install torch faiss-cpu
pip install pyserini
```

#### Install WebShop dependencies

Install requirements.txt from WebShop along with other packages,
```bash
pip install -r requirements.txt
conda install -c pytorch faiss-cpu;
python -m spacy download en_core_web_lg
```

#### Download data and run search engine

Run the following commands to download data and set up the search engine:
```bash
mkdir -p data;
cd data;
gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib; # items_shuffle_1000 - product scraped info
gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu; # items_ins_v2_1000 - product attributes
gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB; # items_shuffle
gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi; # items_ins_v2
gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O # items_human_ins
cd ..

cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
./run_indexing.sh
cd ..
```

<!-- #### For live testing on the web browser

Run the WebShop server:
```bash
bash bash/run_webshop_server.sh
```

If you installed everything correctly as above, you should see a website in [http://localhost:3000/ABC](http://localhost:3000/ABC) -->

## Contact

This project is is actively being developed. For any questions or issues, please contact us at sanjibanc@cornell.edu or paloma.sodhi@gmail.com