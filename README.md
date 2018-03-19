# dl-project
This branch was made to test the LSTM atop our initial CNN.
We ended up seeing that replacing the recurrent layers with
fully connected ones acheived better accuracy across the board.
Maybe we didn't use a large enough set of recurrent layers,
but training was getting very slow

To include the dataset (which is private!!!) create a soft link
in this repository called "project_datasets"
(as in "ln -s <path to your project_datasets> project_datasets")

link to doc: https://docs.google.com/document/d/1hKi0a5zU24qeTtkO4PjkS_J2gP3mFFG1YGfTcLORkxw/edit
link to data description: http://www.bbci.de/competition/iv/desc_2a.pdf
