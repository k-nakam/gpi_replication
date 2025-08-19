# About this folder

This folder contains data files needed to replicate the simulations and empirical analysis. This folder contains the following files:

1. `data/candidate_profile/experiment.csv`: the data used to replicate the results in Candidate Profile Experiment (Section 5). Here, `X` is the texts, `P` is the prompts (as we used text reuse, `X` = `P`), `T` is the treatment feature of interest (military experience), and `resp` is the outcome (feeling thermometer).

2. `data/hongkong/HKData.csv`: the data used to replicate the results in Hong Kong experiment (wave1 in December 2019). This data is from [the replication archive of the original paper](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MVDWCS). See [their codebook](https://dataverse.harvard.edu/file.xhtml?fileId=4460776&version=1.1) for the explanation of variables.

3. `data/hongkong/HKData.csv`: the data used to replicate the results in Hong Kong experiment (wave2 in October 2020). This data is from [the replication archive of the original paper](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MVDWCS). See [their codebook](https://dataverse.harvard.edu/file.xhtml?fileId=4460776&version=1.1) for the explanation of variables.

4. `data/simulation/prompt_create.csv`: the data used to replicate the results in simulation studies (text create). We used [LLaMA3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) based on the prompt to create the new texts and extract the hidden states.

5. `data/simulation/prompt_reuse.csv`: the data used to replicate the results in simulation studies (text reuse). We used [LLaMA3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) based on the prompt to regenerate the input texts and extract the hidden states.

6. `data/simulation/text_processed.pkl`: Pickled file containing the data for simulation (for the version compatibility issue of the pickle file, we also release `data/simulation/text_processed.csv`, which contains the same data frame).


## Caution

Due to data size, we are not able to upload the extracted hidden states. The researchers who want to replicate the empirical analysis thus need to generate the internal representations. See `create_data` folder for more details.