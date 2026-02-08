# About this folder

This folder contains the python codes for (1) GPI method (proposed) and (2) BERT-based method (proposed by Gui and Veitch 2022). The folder contains the following python scripts and the following functions:

- `TarNet.py`: main code for the treatment effect estimation for GPI method
    - `save_filename`: the function to find the filename without overwriting (used to save propensity score in each simulation without overwriting)
    - `TarNet`: the wrapper class for neural network models used as a nuisance function. This class handles the training, validation, and prediction of the neural network models.
    - `estimate_k`: the main function for the treatment effect estimation with cross-fitting. It performs k-fold cross-fitting using the TarNet and the specified propensity score model as nuisance functions. This function is used for empirical applications, which we performed cross-fitting
    - `estimate_and_loss`: the main function for treatment effect estimation with sample-splitting. It performs the statistical inference only using the test dataset. This is the main function used for the simulation studies, as the cross-fitting is computationally infeasible for many iterations.

- `TNutil.py`: the scripts that contains all the utility functions for `TarNet.py`
    - `TarNetBase`: the pytorch class of neural network architecture predicting both outcome head simultaneously.
    - `dml_score`: the function that calculates the influence function for ATE
    - `estimate_psi_split`: the utility function that is used for propensity score estimation without double dipping. This function splits the data and perform the estimation of propensity score in each fold.
    - `load_hiddens`: python function to load the hidden states saved as .pt file and create the python object for treatment effect estimation.
    - `TarNet_loss`: the function that calculates the loss function for the proposed neural network architecture (`TarNetBase`). The neural network and loss function are both used in `TarNet` class in `TarNet.py` file.

- `TI_estimator/Qmod.py`: main code for the treatment effect estimation proposed by Gui and Veitch 2022

- `TI_estimator/TIutil.py`: the scripts that contains all the utility functions for `TI_estimator/Qmod.py`
    - For both `Qmod.py` and `TIutil.py` we used the original scripts used in Gui and Veitch (2022). See their [Github repository](https://github.com/gl-ybnbxb/TI-estimator) for details.