#ifndef _PERCEPTRON_HPP
#define _PERCEPTRON_HPP

#include <vector>

using namespace std;

class perceptron {
    vector<long double> w; // weights coming to the perceptron
    vector<long double> delta_w; // last delta applied to the weights
    long double local_gradient; //  value of the local gradient
    long double y; // sigma( v )
    long double v; // sum ( w * inp )
    long double etha; // learning rate
    long double alpha; // momentum constant
    long double a; // sigmoid function parameter

public:
    /*
        Class constructor

        Receives:
            inp_sz: Size of the input of the perceptron

            etha: learning rate value

            alpha: momentum constant value

            w_range: range of the weights [-w_range, w_range]

            a: constant of the sigmoid function
    */
    perceptron(int inp_sz, long double etha, long double alpha,
               long double w_range, long double a);

    /*
        *
        Basic dot product implementation

        Receives:
            inp: vector of long doubles

        Returns:
            value of the dot product of weights and input
    */
    long double operator*(const vector<long double> &inp);

    /*
        act_func
        Activation function of the perceptron, in this case sigmoid

        Receives:
            x: value to evaluate

        Returns:
            f(x) where f is sigmoid in this case
    */
    long double act_func(long double x);

    /*
        sigma
        Sets the value of y and v, v is the dot product of weights and the input
        and y is f(v) where f is the activation function

        Receives:
            inp: vector of long double representing the input of the perceptron
    */
    void sigma(const vector<long double> &inp);

    /*
        compute_output_gradient
        Given the desired output, calculates the
        local gradient of the perceptron in the output layer

        Receives:
            d: long double representing the desired output
    */
    void compute_output_gradient(long double d);

    /*
        compute_hidden_gradient
        Given the sum for each k of gradient_k*w_k_j, where k is a perceptron
        in the next layer and w_k_j is the weight going from the current perceptron
        to k, calculates the local gradient of the perceptron in the hidden layer

        Receives:
            sum_gradient_w: long double representing the sum of gradients*weights
    */
    void compute_hidden_gradient(long double sum_gradient_w);

    /*
        train
        Given the input of the perceptron, it trains it, the values of
        the local gradient and y need to be computed before calling this

        Receives:
            inp_y: input coming to the perceptron
    */
    void train(const vector<long double> &inp_y);

    /*
        get_y
        Returns the output of the perceptron

        Returns:
            output of the perceptron
    */
    long double get_y();

    /*
        get_w
        Returns the weight incoming from perceptron j to this perceptron

        Receives:
            j: index of the incoming weight

        Returns:
            weight coming from j
    */
    long double get_w(int j);

    /*
        get_local_gradient
        Returns the local gradient of the perceptron

        Returns:
            local gradient of the perceptron
    */
    long double get_local_gradient();
};

#endif