# dl-project
After implementing the deeper CNN from the paper (reference [1] in our report)
(Large_CNN), we tried the RNN atop CNN structure with this model.
Because our time series was longer than that in the original paper, initially, when
we implemented our version of the paper's model, we increased pooling kernel sizes and
strides. In some of this testing, we revert to the original smaller sizes and strides,
and instead use RNN layers to compensate for the longer signal.
In this branch, we also tried adding the rnn layers between layers.
It didn't really help much.

The code for these changes is under the cnn folder, in the "rnn.py" file

To include the dataset (which is private!!!) create a soft link
in this repository called "project_datasets"
(as in "ln -s <path to your project_datasets> project_datasets")

link to doc: https://docs.google.com/document/d/1hKi0a5zU24qeTtkO4PjkS_J2gP3mFFG1YGfTcLORkxw/edit
link to data description: http://www.bbci.de/competition/iv/desc_2a.pdf
