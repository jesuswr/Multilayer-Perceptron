// Question 3: MLP with half and a quarter of training data

#include <iostream>
#include <vector>
#include "multilayer_perceptron.hpp"
#include <algorithm>

using namespace std;

const int NUM_PIXELS = 784, OUTPUT_SIZE = 10, EPOCHS = 15;
const int HIDDEN_SIZE = 100;
const long double WEIGHT_RANGE = 0.05;
const long double ETHA = 0.1, A = 1;
const long double ALPHA = 0.9;

int main() {
	FILE *input = fopen("mnist_train.csv", "r");

	input_data train_data;
	int ans;
	while (fscanf(input, "%d", &ans) != EOF) {
		vector<long double> aux_ans(OUTPUT_SIZE, 0);
		aux_ans[ans] = 1.0;

		vector<long double> pixels(NUM_PIXELS);
		for (int i = 0; i < NUM_PIXELS; ++i) {
			int aux;
			fscanf(input, ",%d", &aux);
			pixels[i] = (long double)aux / 255;
		}

		train_data.push_back({pixels, aux_ans});
	}
	fclose(input);

	input = fopen("mnist_test.csv", "r");
	input_data test_data;
	while (fscanf(input, "%d", &ans) != EOF) {
		vector<long double> aux_ans(OUTPUT_SIZE, 0);
		aux_ans[ans] = 1.0;

		vector<long double> pixels(NUM_PIXELS);
		for (int i = 0; i < NUM_PIXELS; ++i) {
			int aux;
			fscanf(input, ",%d", &aux);
			pixels[i] = (long double)aux / 255;
		}

		test_data.push_back({pixels, aux_ans});
	}
	fclose(input);

	input_data train_data1(train_data.size() / 4), train_data2(train_data.size() / 2);

	random_shuffle(train_data.begin(), train_data.end());
	for (unsigned int i = 0; i < train_data.size() / 4; ++i) {
		train_data1[i] = train_data[i];
	}

	random_shuffle(train_data.begin(), train_data.end());
	for (unsigned int i = 0; i < train_data.size() / 2; ++i) {
		train_data2[i] = train_data[i];
	}
	train_data.clear();

	multilayer_perceptron network1(NUM_PIXELS, HIDDEN_SIZE, OUTPUT_SIZE,
	                               ETHA, ALPHA, WEIGHT_RANGE, A);
	multilayer_perceptron network2(NUM_PIXELS, HIDDEN_SIZE, OUTPUT_SIZE,
	                               ETHA, ALPHA, WEIGHT_RANGE, A);

	printf("Training network 1...\n");
	error_data error_info1 = network1.train(train_data1, test_data, EPOCHS);
	printf("Training network 2...\n");
	error_data error_info2 = network2.train(train_data2, test_data, EPOCHS);


	printf("Calculating Percentage of good clasifications of network 1...\n");
	long double pct_new_standard1 = network1.test1(test_data);
	long double pct_old_standard1 = network1.test2(test_data);
	printf("Calculating Percentage of good clasifications of network 2...\n");
	long double pct_new_standard2 = network2.test1(test_data);
	long double pct_old_standard2 = network2.test2(test_data);

	printf("Average error for epoch with training data in network 1:\n");
	for (auto x : error_info1.first )
		printf("%Lf, ", x);
	printf("\n");
	printf("Average error for epoch with testing data in network 1:\n");
	for (auto x : error_info1.second )
		printf("%Lf, ", x);
	printf("\n");

	printf("Average error for epoch with training data in network 2:\n");
	for (auto x : error_info2.first )
		printf("%Lf, ", x);
	printf("\n");
	printf("Average error for epoch with testing data in network 2:\n");
	for (auto x : error_info2.second )
		printf("%Lf, ", x);
	printf("\n");


	printf("Percentage with new standard in network 1: %c%Lf\n", '%', 100 * pct_new_standard1);
	printf("Percentage with old standard in network 1: %c%Lf\n", '%', 100 * pct_old_standard1);

	printf("Percentage with new standard in network 2: %c%Lf\n", '%', 100 * pct_new_standard2);
	printf("Percentage with old standard in network 2: %c%Lf\n", '%', 100 * pct_old_standard2);

	return 0;
}