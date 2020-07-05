#include "multilayer_perceptron.hpp"
#include "perceptron.hpp"
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <iostream>

using namespace std;

multilayer_perceptron::multilayer_perceptron(int inp_sz, int hidden_sz, int out_sz,
        long double etha, long double alpha, long double w_range, long double a){
    srand(time(NULL));

    for(int i = 0; i < hidden_sz; ++i){
        perceptron aux(inp_sz, etha, alpha, w_range, a);
        hidden_layer.push_back(aux);
    }

    for(int i = 0; i < out_sz; ++i){
        perceptron aux(hidden_sz, etha, alpha, w_range, a);
        output_layer.push_back(aux);
    }
}

void multilayer_perceptron::propagate_forward(const vector<long double> &inp){
    int hidden_sz = hidden_layer.size();
    int output_sz = output_layer.size();
    vector<long double> hidden_y(hidden_sz);

    for(int i = 0; i < hidden_sz; ++i){
        hidden_layer[i].sigma(inp);
        hidden_y[i] = hidden_layer[i].get_y();
    }

    for(int i = 0; i < output_sz; ++i){
        output_layer[i].sigma(hidden_y);
    }
}

void multilayer_perceptron::propagate_backward(const vector<long double> &d){
    int hidden_sz = hidden_layer.size();
    int output_sz = output_layer.size();

    for(int i = 0; i < output_sz; ++i){
        output_layer[i].compute_output_gradient(d[i]);
    }

    for(int i = 0; i < hidden_sz; i++){
        long double sum_grad_w = 0;
        for(int j = 0; j < output_sz; ++j){
            sum_grad_w += 
                output_layer[j].get_local_gradient()*output_layer[j].get_w(i);
        }
        hidden_layer[i].compute_hidden_gradient(sum_grad_w);
    }
}

void multilayer_perceptron::update_weights(const vector<long double> &inp){
    int hidden_sz = hidden_layer.size();
    int output_sz = output_layer.size();
    vector<long double> hidden_y(hidden_sz);

    for(int i = 0; i < hidden_sz; i++){
        hidden_layer[i].train(inp);
        hidden_y[i] = hidden_layer[i].get_y();
    }
    
    for(int i = 0; i < output_sz; ++i){
        output_layer[i].train(hidden_y);
    }
}

error_data multilayer_perceptron::train(input_data train_data, input_data test_data,
        int num_epochs){
    vector<long double> train_error, test_error;
    int n = train_data.size();
    int m = test_data.size();
    int out_sz = output_layer.size();

    for(int k = 0; k <= num_epochs; k++){
        long double avg_error_train = 0, avg_error_test = 0;

        for(int i = 0; i < m; ++i){
            propagate_forward(test_data[i].first);
            for(int j = 0; j < out_sz; ++j){
                long double e = test_data[i].second[j] - hidden_layer[j].get_y();
                avg_error_test += e*e;
            }
        }
        avg_error_test = avg_error_test/(2.0*m);
        test_error.push_back(avg_error_test);

        for(int i = 0; i < n; ++i){
            propagate_forward(train_data[i].first);
            for(int j = 0; j < out_sz; ++j){
                long double e = train_data[i].second[j] - hidden_layer[j].get_y();
                avg_error_train += e*e;
            }
            
            propagate_backward(train_data[i].second);
            update_weights(train_data[i].first);
        }
        avg_error_train = avg_error_train/(2.0*n);
        train_error.push_back(avg_error_train);

        printf("Cycle number %d complete.\n", k);
    }

    return {train_error, test_error};
}

long double multilayer_perceptron::test1(input_data test_data){
    int out_sz = output_layer.size();
    int n = test_data.size();
    long double porcentage = 0;

    for(int i = 0; i < n; i++){
        propagate_forward(test_data[i].first);
        
        bool good = true;
        for(int j = 0; j < out_sz; j++){
            long double y = output_layer[j].get_y();
            long double d = test_data[i].second[j];
           
            if ( y >= 0.9 ){
                if ( d <= 0.1 ){
                    good = false;
                }
            }
            else if ( y <= 0.1 ){
                if ( d >= 0.9 ){
                    good = false;
                }
            }
            else{
                good = false;
            }
        }
        
        if ( good ){
            porcentage++;
        }
    }

    return porcentage/n;
}

long double multilayer_perceptron::test2(input_data test_data){
    int out_sz = output_layer.size();
    int n = test_data.size();
    long double porcentage = 0;

    for(int i = 0; i < n; i++){
        propagate_forward(test_data[i].first);
        
        long double mx = -1;
        int mxi;
        for(int j = 0; j < out_sz; j++){
            long double y = output_layer[j].get_y();

            if ( mx < y ){
                mx = y;
                mxi = j;
            }
        }
        
        if ( test_data[i].second[mxi] > 0.9 ){
            porcentage++;
        }
    }

    return porcentage/n;
}