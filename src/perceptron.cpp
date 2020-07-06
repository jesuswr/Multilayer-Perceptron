#include <vector>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "perceptron.hpp"

using namespace std;

perceptron::perceptron(int inp_sz, long double etha, long double alpha,
                       long double w_range, long double a = 1) : etha(etha), alpha(alpha), a(a) {
    w.resize(inp_sz + 1);
    delta_w.resize(inp_sz + 1);
    for (int i = 0; i <= inp_sz; ++i) {
        // assing random value in the given range to w and 0 to delta_w
        long double rnd = (long double)rand() * 2 / RAND_MAX - 1;
        w[i] = (long double)rnd * w_range;
        delta_w[i] = 0;
    }
}

long double perceptron::operator*(const vector<long double> &inp) {
    long double acum = 0;
    int n = inp.size();
    for (int i = 0; i < n; ++i) {
        acum += w[i] * inp[i];
    }
    // Add the bias value
    acum += w[n];
    return acum;
}

long double perceptron::act_func(long double x) {
    // sigmoid function
    return ( 1.0 / ( 1.0 + exp(-a * x) ) );
}

void perceptron::sigma(const vector<long double> &inp) {
    v = (*this) * inp;
    y = act_func(v);
}

void perceptron::compute_output_gradient(long double d) {
    local_gradient = a * y * (1.0 - y) * (d - y) ;
}

void perceptron::compute_hidden_gradient(long double sum_gradient_w) {
    local_gradient = a * y * (1.0 - y) * sum_gradient_w;
}

void perceptron::train(const vector<long double> &inp_y) {
    int n = inp_y.size();
    for (int i = 0; i < n; ++i) {
        delta_w[i] = alpha * delta_w[i] + etha * local_gradient * inp_y[i];
        w[i] += delta_w[i];
    }
    // update bias
    delta_w[n] = alpha * delta_w[n] + etha * local_gradient;
    w[n] += delta_w[n];
}

long double perceptron::get_y() {
    return y;
}

long double perceptron::get_w(int j) {
    return w[j];
}

long double perceptron::get_local_gradient() {
    return local_gradient;
}