# About this folder

This folder contains the file needed to replicate the empirical analysis. There is only one file.

1. `analysis/empirical_application.py`: Python code to run the empirical application.

# How to Use

For each application, you need to run the following script:

- For candidate profile experiment (Section 5):

```bash
python analysis/empirical_application.py \
    --application "candidate_profile" \
    --data_dir "data/candidate_profile/experiment" \
    --hidden_dir "data/candidate_profile/hidden_candidate"
```

- For Hong Kong experiment Wave 1 (Appendix):

```bash
python analysis/empirical_application.py \
    --application "hongkong1" \
    --data_dir "data/hongkong/HKData" \
    --K "5" \
    --hidden_dir "data/hongkong/hidden_HK"
```


- For Hong Kong experiment Wave 2 (Appendix):

```bash
python analysis/empirical_application.py \
    --application "hongkong2" \
    --data_dir "data/hongkong/HKRepData" \
    --K "5" \
    --hidden_dir "data/hongkong/hidden_HKrep"
```