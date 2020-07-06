// Question 3: MLP with alpha as 0, 0.25 and 0.5

#include <iostream>
#include <vector>
#include "multilayer_perceptron.hpp"

using namespace std;

const int NUM_PIXELS = 784, OUTPUT_SIZE = 10, EPOCHS = 15;
const int HIDDEN_SIZE = 100;
const long double WEIGHT_RANGE = 0.05;
const long double ETHA = 0.1, A = 1;
const long double ALPHA[] = {0, 0.25, 0.5};

int main(){
    FILE *input = fopen("mnist_train.csv", "r");

    input_data train_data;
    int ans;
    while(fscanf(input, "%d", &ans) != EOF){
        vector<long double> aux_ans(OUTPUT_SIZE,0);
        aux_ans[ans] = 1.0;

        vector<long double> pixels(NUM_PIXELS);
        for (int i = 0; i < NUM_PIXELS; ++i){
            int aux;
            fscanf(input, ",%d", &aux);
            pixels[i] = (long double)aux/255;
        }

        train_data.push_back({pixels, aux_ans});
    }
    fclose(input);

    input = fopen("mnist_test.csv", "r");
    input_data test_data;
    while(fscanf(input, "%d", &ans) != EOF){
        vector<long double> aux_ans(OUTPUT_SIZE,0);
        aux_ans[ans] = 1.0;

        vector<long double> pixels(NUM_PIXELS);
        for (int i = 0; i < NUM_PIXELS; ++i){
            int aux;
            fscanf(input, ",%d", &aux);
            pixels[i] = (long double)aux/255;
        }

        test_data.push_back({pixels, aux_ans});
    }
    fclose(input);

    multilayer_perceptron network1(NUM_PIXELS, HIDDEN_SIZE, OUTPUT_SIZE,
        ETHA, ALPHA[0], WEIGHT_RANGE, A);
    multilayer_perceptron network2(NUM_PIXELS, HIDDEN_SIZE, OUTPUT_SIZE,
        ETHA, ALPHA[1], WEIGHT_RANGE, A);
    multilayer_perceptron network3(NUM_PIXELS, HIDDEN_SIZE, OUTPUT_SIZE,
        ETHA, ALPHA[2], WEIGHT_RANGE, A);

    printf("Training network 1...\n");
    error_data error_info1 = network1.train(train_data, test_data, EPOCHS);
    printf("Training network 2...\n");
    error_data error_info2 = network2.train(train_data, test_data, EPOCHS);
    printf("Training network 3...\n");
    error_data error_info3 = network3.train(train_data, test_data, EPOCHS);


    printf("Calculating Percentage of good clasifications of network 1...\n");
    long double pct_new_standard1 = network1.test1(test_data);
    long double pct_old_standard1 = network1.test2(test_data);
    printf("Calculating Percentage of good clasifications of network 2...\n");
    long double pct_new_standard2 = network2.test1(test_data);
    long double pct_old_standard2 = network2.test2(test_data);
    printf("Calculating Percentage of good clasifications of network 3...\n");
    long double pct_new_standard3 = network3.test1(test_data);
    long double pct_old_standard3 = network3.test2(test_data);

    printf("Average error for epoch with training data in network 1:\n");
    for(auto x : error_info1.first )
        printf("%Lf, ", x);
    printf("\n");
    printf("Average error for epoch with testing data in network 1:\n");
    for(auto x : error_info1.second )
        printf("%Lf, ", x);
    printf("\n");

    printf("Average error for epoch with training data in network 2:\n");
    for(auto x : error_info2.first )
        printf("%Lf, ", x);
    printf("\n");
    printf("Average error for epoch with testing data in network 2:\n");
    for(auto x : error_info2.second )
        printf("%Lf, ", x);
    printf("\n");

    printf("Average error for epoch with training data in network 3:\n");
    for(auto x : error_info3.first )
        printf("%Lf, ", x);
    printf("\n");
    printf("Average error for epoch with testing data in network 3:\n");
    for(auto x : error_info3.second )
        printf("%Lf, ", x);
    printf("\n");

    printf("Percentage with new standard in network 1: %c%Lf\n", '%', 100*pct_new_standard1);
    printf("Percentage with old standard in network 1: %c%Lf\n", '%', 100*pct_old_standard1);
    
    printf("Percentage with new standard in network 2: %c%Lf\n", '%', 100*pct_new_standard2);
    printf("Percentage with old standard in network 2: %c%Lf\n", '%', 100*pct_old_standard2);

    printf("Percentage with new standard in network 3: %c%Lf\n", '%', 100*pct_new_standard3);
    printf("Percentage with old standard in network 3: %c%Lf\n", '%', 100*pct_old_standard3);

    return 0;
}