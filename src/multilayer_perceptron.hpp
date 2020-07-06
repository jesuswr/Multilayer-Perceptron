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
    vector<perceptron> hidden_layer; // vector of perceptrons in hidden layer
    vector<perceptron> output_layer; // vector of perceptrons in output layer
public:
    /*
        Class constructor
        
        Receives:
            inp_sz: size of the input coming to the MLP

            hidden_sz: size of the hidden layer

            out_sz: size of the output layer

            etha: learning rate value

            alpha: momentum constant value

            w_range: range of the weights

            a: constant of the sigmoid function
    */
    multilayer_perceptron(int inp_sz, int hidden_sz, int out_sz,
        long double etha, long double alpha, long double w_range,
        long double a);
    
    /*
        propagate_forward
        Gets the input of the MLP and it does the forward propagation part
        of the algorithm
        
        Receives:
            inp: vector of long double representing the input of the MLP
    */
    void propagate_forward(const vector<long double> &inp);



    /*
        propagate_backward
        Gets the desired output of the MLP and it does the 
        backward propagation part of the algorithm
        
        Receives:
            d: vector of long double representing the desired output of the MLP
    */
    void propagate_backward(const vector<long double> &d);

    /*
        update_weights
        Gets the input of the MLP and it updates the weights, before using this
        it's important to use the forward and backward propagation functions
        
        Receives:
            inp: vector of long double representing the input of the MLP
    */
    void update_weights(const vector<long double> &inp);

    /*
        train
        Receives the training data, testing data and number of epochs and it 
        trains the network with the backpropagation algorithm, also returns 
        the average error per epoch

        Receives:
            train_data and test_data: vector of pairs of input data and wanted output

            num_epochs: number of epochs to train

        Returns:
            pair of vectors of long doubles, one has the average errors for the
            training data and the other of the testing data
    */
    error_data train(input_data train_data, input_data test_data, int num_epochs);


    /*
        test1
        Calculates the percentage of well classified data, the standard for this
        is that the output is one is the perceptron gives >= 0.9 and 0 if it gives
        <= 0.1

        receives:
            test_data: data to test the perceptron

        returns:
            percentage of well classified data/100
    */
    long double test1(input_data test_data);

    /*
        test2
        Calculates the percentage of well classified data, the standard for this
        is that the output is one is the perceptron gives the maximum value between
        the perceptrons and 0 otherwise

        receives:
            test_data: data to test the perceptron

        returns:
            percentage of well classified data/100
    */
    long double test2(input_data test_data);
};








#endif
