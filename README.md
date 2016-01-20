PySLDANormal
==========

PySLDANormal is a supervised Latent Dirichlet Allocation topic modeling package for regression.

Please download the latest version from our [GitHub repository](https://github.com/kzhai/PySLDANormal).

Please send any bugs or problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------

This package depends on many external python libraries, such as numpy, scipy and nltk.

Launch and Execute
----------

Assume the PySLDANormal package is downloaded under directory ```$PROJECT_SPACE/src/```, i.e., 

	$PROJECT_SPACE/src/PySLDANormal

To prepare the example dataset,

	tar zxvf review.scale.stem.tar.gz

To launch PySLDANormal, first redirect to the directory of PySLDANormal source code,

	cd $PROJECT_SPACE/src/PySLDANormal

and run the following command on example dataset,

	python -m launch_train --input_directory=./review.scale.stem/ --output_directory=./ --number_of_topics=10 --training_iterations=50
	
The generic argument to run PySLDANormal is

	python -m launch_train --input_directory=$INPUT_DIRECTORY/$CORPUS_NAME --output_directory=$OUTPUT_DIRECTORY --number_of_topics=$NUMBER_OF_TOPICS --training_iterations=$NUMBER_OF_ITERATIONS

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$CORPUS_NAME```.

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -m launch_train --help
