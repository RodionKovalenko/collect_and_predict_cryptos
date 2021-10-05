# collect_and_predict_cryptos-
Collections data from uphold API written in Rust and predicts the price with Convolutions LSTM written in python


1. Install Rust and C++ Libraries (perhaps through Visual Studio Code on windows). 
2. Install python and libraries
pandas
numpy
pathlib
keras
matplotlib
math
sklearn
datetime
os
scipy


Usage:

Collect crypto data every 30 minutes from uphold api. The list of currencies can be seen extended in src/uphold_api/cryptocurrency_api.rs

The default save path is cryptocurrency_rates_history.json in src folder.

For predictions run 
		python lstm-conv-dense.py

Or use any other framework like Spyder to run the following script.



After script is run, two folders will be created:
	- predictions
	- saved_models


The currency for prediction can be specified in the script lstm-conv-dense.py line 142. 











