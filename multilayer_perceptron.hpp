#ifndef _MULTILAYER_PERCEPTRON_HPP
#define _MULTILAYER_PERCEPTRON_HPP

#include "perceptron.hpp"
#include <vector>

using namespace std;

// vector represents error in each ephoc, one for training data and other
// for testing data
typedef pair<vector<long double>, vector<long double>> error_data;
typedef vector<pair<vector<long double>, vector<long double>>> input_data;

class multilayer_perceptron{
    vector<perceptron> hidden_layer;
    vector<perceptron> output_layer;
public:
    multilayer_perceptron(int inp_sz, int hidden_sz, int out_sz,
        long double etha, long double alpha, long double w_range,
        long double a);
    
    void propagate_forward(const vector<long double> &inp);

    void propagate_backward(const vector<long double> &id);

    void update_weights(const vector<long double> &inp);

    error_data train(input_data train_data, input_data test_data, int num_epochs);

    long double test1(input_data test_data);

    long double test2(input_data test_data);
};








#endif
