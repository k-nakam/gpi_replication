# Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments

This repository contains the code for replication of the results in the paper "Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments" by Imai and Nakamura ([arxiv link](https://arxiv.org/pdf/2410.00903)).

This repository includes the source code of the proposed algorithm. Note that we update the proposed algorithm with additional functionalities and released it as a python package. If you want to use the recent version of the package, please refer to [this website](https://gpi-pack.github.io/).

## About this repository

This github repository includes four subdirectories: `src`, `data`, `simulation`, and `analysis`. Each folder contains README file about the detailed explanation of each subdirectory.

- The sub-directory `create_data` includes the python code to create / repeat the texts and extract the hidden representations. For the entire analysis, we use the checkpoint [LLaMA3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and extract the hidden states of the last layer and the last token.

- The sub-directory `src` includes (1) the python code for our proposed methodology and (2) the existing methodology based on BERT ([Pryzant et al. 2021](https://aclanthology.org/2021.naacl-main.323/) and [Gui and Veitch 2022](https://arxiv.org/abs/2210.00079)). For the code of the existing methodology (under `src/TI_estimator`), we directly used the code provided by [this github repository](https://github.com/gl-ybnbxb/TI-estimator) of Gui and Veitch.

- The sub-directory `data` includes the data used to replicate the original analysis, including (1) simulation studies (Section 4), (2) empirical analysis on candidate profile experiment (Section 5), and (3) additional empirical analysis on Hong Kong experiment (Appendix). In order to replicate the original analysis, you need to use LLM ([LLaMA3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)) and ask them to produce the texts and extract the hidden states (we cannot share the extracted hidden states as they are too large). For the empirical analysis, we used the data of the published articles. See [Fong and Grimmer (2016)](https://aclanthology.org/P16-1151/) for candidate profile experiment (we obtained the data by contacting the author) and [Fong and Grimmer (2023)](https://onlinelibrary.wiley.com/doi/abs/10.1111/ajps.12649) for Hong Kong experiment (replication data is available from [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MVDWCS)). 

- The subdirectory `simulation` includes the codes used for the simulation studies.

- The subdirectory `analysis` includes the codes used for the two empirical applications: (1) candidate profile experiment (Section 5) and (2) Hong Kong experiment (Appendix).

- The file `sample.env` is how you should set up `.env` file, in which you store your own huggingface token. This is necessary when you use LLaMa3 to generate the data. See the preface of `create_data/create_text.py` for the detailed explanation.

For both simulation and analysis, you need to use GPU (Graphical Processing Units). In all the analysis, we use the FASRC Cluster with 16 cores, 4 GPU Units (A100 offered by `gpu` partition in FASRC). See [here](https://docs.rc.fas.harvard.edu/kb/running-jobs/) for the detailed information of the partition we used. We used python3.10.9 for all the analyses.

## Replication Workflow

To replicate the original paper, please follow the following procedure:

1. run the code in `create_data` to obtain the internal representation of LLMs. To do this, you need to supply your huggingface token in .env file. See the preface of `create_text.py` for the more detailed instruction.

2. run the code in `simulation` to replicate the results in Section 4 (simulation studies) and `analysis` to replicate the results in Section 5 (empirical application, candidate profile experiment) and Appendix (Hong Kong experiment).