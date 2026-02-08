# Causal Inference with Generative Artificial Intelligence: Application to Texts as Treatments

This repository contains the code for replication of the results in the paper "Causal Inference with Generative Artificial Intelligence: Application to Texts as Treatments" by Imai and Nakamura ([arxiv link](https://arxiv.org/pdf/2410.00903)).

This repository includes the code to reproduce all the figures and results in the manuscript. Note that we update the proposed algorithm with additional functionalities and released it as a python package. If you want to use the recent version of the package, please refer to [this website](https://gpi-pack.github.io/).

## About this repository

This github repository includes four subdirectories: `src`, `data`, `simulation`, and `analysis`. Each folder contains README file about the detailed explanation of each subdirectory.

- The sub-directory `create_data` includes the python code to create / repeat the texts and extract the hidden representations. For the entire analysis, we use the checkpoint [LLaMA3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and extract the hidden states of the last layer and the last token.

- The sub-directory `src` includes (1) the python code for our proposed methodology and (2) the existing methodology based on BERT ([Pryzant et al. 2021](https://aclanthology.org/2021.naacl-main.323/) and [Gui and Veitch 2022](https://arxiv.org/abs/2210.00079)). For the code of the existing methodology (under `src/TI_estimator`), we directly used the code provided by [this github repository](https://github.com/gl-ybnbxb/TI-estimator) of Gui and Veitch (2022).

- The sub-directory `data` includes the data used to replicate the original analysis, including (1) simulation studies (Section 4), (2) empirical analysis on candidate profile experiment (Section 5), and (3) additional empirical analysis on Hong Kong experiment (Appendix). In order to replicate the original analysis, you need to use LLM ([LLaMA3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)) and ask them to produce the texts and extract the hidden states. All simulation data are available in `data/simulation`. For the empirical analysis, we used the data of the published articles. We shared all the data we used in this subdirectory. See `data/candidate_profile` for the candidate profile experiment and `data/hongkong` for the Hong Kong experiment.

- The subdirectory `simulation` includes the codes used for the simulation studies.

- The subdirectory `analysis` includes the codes used for the two empirical applications: (1) candidate profile experiment (Section 5) and (2) Hong Kong experiment (Appendix).

- The file `sample.env` is how you should set up `.env` file, in which you store your own huggingface token. This is necessary if you use LLaMa3 to generate the data. See the preface of `create_data/create_text.py` for the detailed explanation.

For both simulation and analysis, you need to use GPU (Graphical Processing Units). In all the analysis, we use the FASRC Cluster with 16 cores, 4 GPU Units (A100 offered by `gpu` partition in FASRC). See [here](https://docs.rc.fas.harvard.edu/kb/running-jobs/) for the detailed information of the partition we used. We used python3.10.9 for all the analyses. The python modules for the exact replication (unless specified in .py file) are as follows:

```txt
numpy==1.26.4
scipy==1.11.3
matplotlib==3.8.1
pandas==2.1.1
scikit-learn==1.3.1
torch==2.3.1
transformers==4.18.0
tokenizers==0.12.1
tqdm==4.64.1
geomloss==0.2.6
imbalanced-learn==0.12.4
dcor==0.6
regex==2023.8.8
huggingface-hub==0.33.2
sacremoses==0.1.1
pyyaml==6.0.1
packaging==23.1
filelock==3.12.4
requests==2.28.2
joblib==1.3.2
threadpoolctl==3.2.0
numba==0.59.0
llvmlite==0.42.0
sympy==1.12
networkx==3.1
jinja2==3.1.2
fsspec==2023.6.0
typing-extensions==4.12.2
python-dateutil==2.8.2
pytz==2023.3.post1
tzdata==2023.3
pillow==10.0.1
cycler==0.12.1
kiwisolver==1.4.5
pyparsing==3.1.1
fonttools==4.44.3
six==1.16.0
MarkupSafe==2.1.3
click==8.1.7
mpmath==1.3.0
idna==3.4
charset-normalizer==2.1.1
urllib3==1.26.14
certifi==2022.12.7
```

## Replication Workflow

To replicate the original paper, please follow the following procedure:

1. run the code in `create_data` to obtain the internal representation of LLMs. To do this, you need to supply your huggingface token in .env file. See the preface of `create_text.py` for the more detailed instruction. It should take roughly 12 hours for each dataset (simulation studies, candidate profile experiment, and Hong Kong experiment). As this is the most computationally expensive process, we uploaded the extracted hidden states in our [Dataverse repository](https://doi.org/10.7910/DVN/2DZQ6T).

2. run the code in `simulation` to replicate the results in Section 4 (simulation studies) and `analysis` to replicate the results in Section 5 (empirical application, candidate profile experiment) and Appendix (Hong Kong experiment). For the simulation studies, the proposed method (GPI) takes approximately 72 hours in total for all iterations, and the BERT-based method requires an additional 72 hours for each confounding setting (weak, moderate, and strong). The empirical application takes approximately 1 hour on a machine equipped with an A100 GPU.

3. run the code in `visualization` to create all the figures in the main text.