#include "multilayer_perceptron.hpp"
#include "perceptron.hpp"
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <iostream>

using namespace std;

multilayer_perceptron::multilayer_perceptron(int inp_sz, int hidden_sz, int out_sz,
        long double etha, long double alpha, long double w_range, long double a) {
	// Set rand seed so we can set random weights
	srand(time(NULL));

	// Create perceptrons in the hidden layer
	for (int i = 0; i < hidden_sz; ++i) {
		perceptron aux(inp_sz, etha, alpha, w_range, a);
		hidden_layer.push_back(aux);
	}

	// Create perceptrons in the output layer
	for (int i = 0; i < out_sz; ++i) {
		perceptron aux(hidden_sz, etha, alpha, w_range, a);
		output_layer.push_back(aux);
	}
}

void multilayer_perceptron::propagate_forward(const vector<long double> &inp) {
	int hidden_sz = hidden_layer.size();
	int output_sz = output_layer.size();
	vector<long double> hidden_y(hidden_sz);

	// Calculate the output of the hidden layer and save it, using the input
	for (int i = 0; i < hidden_sz; ++i) {
		hidden_layer[i].sigma(inp);
		hidden_y[i] = hidden_layer[i].get_y();
	}

	// Calculate the output of the output layer using the hidden layer output
	for (int i = 0; i < output_sz; ++i) {
		output_layer[i].sigma(hidden_y);
	}
}

void multilayer_perceptron::propagate_backward(const vector<long double> &d) {
	int hidden_sz = hidden_layer.size();
	int output_sz = output_layer.size();

	// Calculate local gradient of output gradient with the desired output
	for (int i = 0; i < output_sz; ++i) {
		output_layer[i].compute_output_gradient(d[i]);
	}

	// Calculate local gradient of hidden layer with the local gradients
	// of the
	for (int i = 0; i < hidden_sz; ++i) {
		long double sum_grad_w = 0;
		for (int j = 0; j < output_sz; ++j) {
			sum_grad_w +=
			    output_layer[j].get_local_gradient() * output_layer[j].get_w(i);
		}
		hidden_layer[i].compute_hidden_gradient(sum_grad_w);
	}
}

void multilayer_perceptron::update_weights(const vector<long double> &inp) {
	int hidden_sz = hidden_layer.size();
	int output_sz = output_layer.size();
	vector<long double> hidden_y(hidden_sz);

	// Train the hidden layer and save their output to train the output layer
	for (int i = 0; i < hidden_sz; ++i) {
		hidden_layer[i].train(inp);
		hidden_y[i] = hidden_layer[i].get_y();
	}

	// Train the output layer with the hidden layer output
	for (int i = 0; i < output_sz; ++i) {
		output_layer[i].train(hidden_y);
	}
}

error_data multilayer_perceptron::train(input_data train_data, input_data test_data,
                                        int num_epochs) {
	vector<long double> train_error, test_error;
	int n = train_data.size();
	int m = test_data.size();
	int out_sz = output_layer.size();

	for (int k = 0; k <= num_epochs; ++k) {
		long double avg_error_train = 0, avg_error_test = 0;

		// Get output with testing data and compute average error
		for (int i = 0; i < m; ++i) {
			propagate_forward(test_data[i].first);
			for (int j = 0; j < out_sz; ++j) {
				long double e = test_data[i].second[j] - hidden_layer[j].get_y();
				avg_error_test += e * e;
			}
		}
		avg_error_test = avg_error_test / (2.0 * m);
		test_error.push_back(avg_error_test);

		// Get output with training data, compute average error and train the
		// perceptrons
		for (int i = 0; i < n; ++i) {
			propagate_forward(train_data[i].first);
			for (int j = 0; j < out_sz; ++j) {
				long double e = train_data[i].second[j] - hidden_layer[j].get_y();
				avg_error_train += e * e;
			}

			propagate_backward(train_data[i].second);
			update_weights(train_data[i].first);
		}
		avg_error_train = avg_error_train / (2.0 * n);
		train_error.push_back(avg_error_train);

		printf("Cycle number %d complete.\n", k);
	}

	return {train_error, test_error};
}

long double multilayer_perceptron::test1(input_data test_data) {
	int out_sz = output_layer.size();
	int n = test_data.size();
	long double percentage = 0;

	for (int i = 0; i < n; ++i) {
		// Get the output with the test data
		propagate_forward(test_data[i].first);

		bool good = true;
		for (int j = 0; j < out_sz; ++j) {
			long double y = output_layer[j].get_y();
			long double d = test_data[i].second[j];

			if ( y >= 0.9 ) {
				// if output is >= 0.9 but desired output is 0, is bad
				if ( d <= 0.1 ) {
					good = false;
				}
			}
			else if ( y <= 0.1 ) {
				// if output is <= 0.1 but desired output is 1, is bad
				if ( d >= 0.9 ) {
					good = false;
				}
			}
			else {
				// if 0.1 < output < 0.9, is bad
				good = false;
			}
		}

		if ( good ) {
			percentage++;
		}
	}

	return percentage / n;
}

long double multilayer_perceptron::test2(input_data test_data) {
	int out_sz = output_layer.size();
	int n = test_data.size();
	long double percentage = 0;

	for (int i = 0; i < n; ++i) {
		propagate_forward(test_data[i].first);

		long double mx = -1;
		int mxi;
		for (int j = 0; j < out_sz; ++j) {
			long double y = output_layer[j].get_y();
			// Get the index of the neuron with max output
			if ( mx < y ) {
				mx = y;
				mxi = j;
			}
		}

		// If the number that the neurn represents is 1 in the desired output
		// sum a good classification
		if ( test_data[i].second[mxi] > 0.9 ) {
			percentage++;
		}
	}

	return percentage / n;
}