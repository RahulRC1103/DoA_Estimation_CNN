# DoA Estimation CNN

The purpose of this project was to assess the performance of neural network based models for direction of arrival estimation when compared to conventional models (signal processing based). This folder contains a Jupyter notebook, a txt file containing all the libraries required, a python script with custom functions and a Data folder.

The Jupyter notebook "SSL_Project.ipynb" generates the dataset (Room Impulse Responses) and saves them under Data folder. The files are then used to evaluate the performance of the models. By default, the Data folder contains 2 files. The two files are for plotting in Section 6 in the notebook.

- vary_RT.npz: Model performance when varying the Reverberation Time 
- vary_SNR.npz: Model performance when varying the Signal to Noise Ratio
