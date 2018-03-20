# dl-project

* Debleena Sengupta (004945991)
* Fan Hung (804319873)
* Jayanth (405026111)
  

We split our efforts into 3 tracks, one for primarily testing CNNs,
one for primarily testing RNNs (alone), and one for primarily testing
a combination of the two layer types. We created alternate branches for
our different tests. After some discussion of the general structure of
each started these three paths concurrently, and then gradually converged
to our better models.

This is a summary of the notable branches in the repo:

master:
	This master branch contains the best model we found with only CNN layers.
	the main code is in the jupyter notebook "Project Workbook - Best Result.ipynb."
	This model produces  65-69% accuracy on test data.To check the model’s accuracy, simply run all the cells in order and the last cell will report the test accuracy.
	We optimized this model to train on  patient 1 as well as over all patients. We also used this model to check the how the accuracy varies over different time steps. In order to see those results, we modified the cell with the comment “Creating train, val, test sets”. In this cell, we modified the “time_steps” variable.We then run the notebook and save the final output and graphed the results shown in Figure 1 of our report.

debleena_branch:
	This branch contains the initial implementation of CNN layers by themselves on the raw data.

fans_branch:
	This branch contains the initial implementation of Vanilla RNN layers on top of CNN layers.

jayanth_lstm:
	This branch contains the initial implementation of 3 layered stacked lstm layers by themselves on the raw data.

fans_branch_lstm:
	This branch was made to test the LSTM atop an initial CNN.
	We ended up seeing that replacing the recurrent layers with fully connected ones acheived better accuracy across the board.

optimizing_cnn:
	This branch contains some code for optimizing a shorter CNN over different training parameters, and also
	This branch contains our implementation of the deep network from the paper at our first reference
	(link: https://arxiv.org/pdf/1703.05051.pdf) the network is called "Large_CNN"

rnn_on_new_cnn:
	After implementing the model (right up above) with the deeper cnn, we also tried adding rnn layers to this model

fans_branch_rnn_backtrack:
	After training other cnns we tried going back and changing the RNN atop CNN implementation

fans_branch_no_rnn_sanity_check:
	This branch removed the recurrent layers from the CNNs and showed better accuracy, confirming that our recurrent layers
	weren't really helping in our setup

To include the dataset (which is private!!!) create a soft link
in this repository called "project_datasets"
(as in "ln -s <path to your project_datasets> project_datasets")

link to doc: https://docs.google.com/document/d/1hKi0a5zU24qeTtkO4PjkS_J2gP3mFFG1YGfTcLORkxw/edit
link to data description: http://www.bbci.de/competition/iv/desc_2a.pdf
