#ifndef _PERCEPTRON_HPP
#define _PERCEPTRON_HPP

#include <vector>

using namespace std;

class perceptron{
    vector<long double> w;
    vector<long double> delta_w;
    long double local_gradient; 
    long double y; // sigma( v )
    long double v; // sum ( w * inp )
    long double etha; // learning rate
    long double alpha; // momentum constant
    long double a; // sigmoid function parameter

public:
    perceptron(int inp_sz, long double etha, long double alpha,
        long double w_range, long double a);

    long double operator*(const vector<long double> &inp);

    long double act_func(long double x);

    void sigma(const vector<long double> &inp);

    void compute_output_gradient(long double d);

    void compute_hidden_gradient(long double sum_gradient_w);

    void train(const vector<long double> &inp_y);

    long double get_y();

    long double get_w(int j);

    long double get_local_gradient();
};

#endif