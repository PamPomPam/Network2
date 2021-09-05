
#include <iostream>
#include <time.h>
#include <algorithm>
#include "Matrix.h"
#include "Algo.h"
using namespace std;



void dense_ffw(Activation* input, Activation* output, Matrix* weights, Vector* biases) {
	int input_dim = input->size;
	int output_dim = output->size;
	if (weights->rows != output_dim ||
		weights->cols != input_dim ||
		biases->rows != output_dim) {
		cerr << "Cannot dense_ffw!!!" << endl;
		throw std::invalid_argument("");
	}
	double summer;
	for (int j = 0; j < output_dim; j++) {
		summer = 0;
		for (int i = 0; i < input_dim; i++) {
			summer += weights->values[j * input_dim + i] * input->values[i];
		}
		output->values[j] = summer + biases->values[j];
	}
}

void dense_backprop(Activation* output, Activation* input, Matrix* weights, Matrix* w_updates, Vector* b_updates) {
	int input_dim = input->size;
	int output_dim = output->size;
	if (weights->rows != output_dim ||
		weights->cols != input_dim ||
		w_updates->rows != output_dim ||
		w_updates->cols != input_dim ||
		b_updates->rows != output_dim) {
		cerr << "Cannot dense_backprop!!!" << endl;
		throw std::invalid_argument("");
	}
	float summer;
	for (int j = 0; j < output_dim; j++) {
		b_updates->values[j] += output->values[j];
	}
	for (int i = 0; i < input_dim; i++) {
		summer = 0;
		for (int j = 0; j < output_dim; j++) {
			w_updates->values[input_dim * j + i] += output->values[j] * input->values[i];
			summer += weights->values[input_dim * j + i] * output->values[j];
		}
		input->values[i] = summer;
	}
}

void dense_update(Matrix* weights, Vector* biases, Matrix* w_updates, Vector* b_updates, float alpha) {
	int input_dim = weights->cols;
	int output_dim = weights->rows;
	for (int j = 0; j < output_dim; j++) {
		biases->values[j] -= alpha * b_updates->values[j];
		b_updates->values[j] = 0;
		for (int i = 0; i < input_dim; i++) {
			weights->values[j * input_dim + i] -= alpha * w_updates->values[j * input_dim + i];
			w_updates->values[j * input_dim + i] = 0;
		}
	}
}

void sigm_ffw(Activation* input, Activation* output) {
	int input_dim = input->size;
	int output_dim = output->size;
	if (input_dim != output_dim) {
		cerr << "Cannot sigm_ffw!!!" << endl;
		throw std::invalid_argument("");
	}
	for (int j = 0; j < input_dim; j++) {
		output->values[j] = sigmoid(input->values[j]);
	}
}

void sigm_backprop(Activation* output, Activation* input) {
	int input_dim = input->size;
	int output_dim = output->size;
	if (input_dim != output_dim) {
		cerr << "Cannot dense_ffw!!!" << endl;
		throw std::invalid_argument("");
	}
	for (int j = 0; j < output_dim; j++) {
		input->values[j] = output->values[j] * d_sigmoid(input->values[j]);
	}
}

void relu_ffw(Activation* input, Activation* output) {
	int input_dim = input->size;
	int output_dim = output->size;
	if (input_dim != output_dim) {
		cerr << "Cannot relu_ffw!!!" << endl;
		throw std::invalid_argument("");
	}
	for (int j = 0; j < input_dim; j++) {
		output->values[j] = max(input->values[j], 0.0f);
	}
}

void relu_backprop(Activation* output, Activation* input) {
	int input_dim = input->size;
	int output_dim = output->size;
	if (input_dim != output_dim) {
		cerr << "Cannot relu_backprop!!!" << endl;
		throw std::invalid_argument("");
	}
	for (int j = 0; j < output_dim; j++) {
		input->values[j] = output->values[j] * (input->values[j] > 0);
	}
}



void conv_ffw1(Activation* input, Activation* output, Matrix* weights, Vector* biases, int filter_size, int n_filters, int filter_stride, int padding) {
	cerr << "new" << endl;
	int input_cols = input->cols;
	int input_depth = input->depth;
	int input_rows = input->size / input_depth / input_cols;

	int output_cols = output->cols;
	int output_depth = output->depth;
	int output_rows = output->size / output_depth / output_cols;

	int weights_cols = weights->cols;
	int weights_rows = weights->rows;
	int weights_rows_ = filter_size;
	int weights_cols_ = filter_size;

	Matrix temp(output_cols, output_rows);
	temp.make_zero();

	int n;
	int d;
	int j;
	int i;
	int i_;
	int j_;

	int inpt_index = 0;
	int outpt_index = 0;
	int weights_index = 0;
	int temp_index = weights_index; // ? right

	for (n = 0; n < n_filters; n++) {
		for (d = 0; d < input_depth; d++) {

			for (j = 0; j < output_rows; j++) {
				for (i = 0; i < output_cols; i++) {
					// actual filter
					for (j_ = 0; j_ < filter_size; j_++) {
						for (i_ = 0; i_ < filter_size; i_++) {
							cerr << weights_index << endl;
							cerr << inpt_index << endl;
							cerr << outpt_index << endl;
							cerr << "----" << endl;
							float a = input->values[inpt_index];
							float b = input->values[weights_index];
							float c = input->values[weights_index];
							temp.values[outpt_index] += weights->values[weights_index] * input->values[inpt_index];

							inpt_index += 1;
							weights_index += 1;
						}
						inpt_index -= filter_size;
						weights_index -= filter_size;

						inpt_index += input_cols;
						weights_index += filter_size;
						
					}
					temp.values[weights_index] += biases->values[n];

					inpt_index -= filter_size * input_cols;
					weights_index -= filter_size * filter_size; // 0


					inpt_index += filter_stride;
					outpt_index += 1;
				}
				inpt_index -= output_cols * filter_stride;
				outpt_index -= output_cols;


				inpt_index += filter_stride * input_cols;
				outpt_index += output_cols;
			}
			inpt_index -= output_rows * filter_stride * input_cols;
			outpt_index -= output_rows * output_cols;

			inpt_index += input_cols * input_rows;
			weights_index += filter_size * filter_size;
			outpt_index += output_cols * output_rows;
		}
		inpt_index -= input_depth * input_cols * input_rows;
		weights_index -= input_depth * filter_size * filter_size;
		outpt_index -= input_depth * output_cols * output_rows;

		//cerr << output_cols * output_rows << endl;

		//copy(temp.values, temp.values + output_cols * output_rows, &output->values[n*output_cols * output_rows]);

		temp.make_zero();

		weights_index += weights_cols;
		outpt_index += output_cols * output_rows * input_depth;


	}
}

void conv_ffw2(Activation* input, Activation* output, Matrix* weights, Vector* biases, int filter_size, int n_filters, int filter_stride, int padding) {
	int input_cols = input->cols;
	int input_depth = input->depth;
	int input_rows = input->size / input_depth / input_cols;

	int output_cols = output->cols;
	int output_depth = output->depth;
	int output_rows = output->size / output_depth / output_cols;

	int weights_cols = weights->cols;
	int weights_rows = weights->rows;
	int weights_rows_ = filter_size;
	int weights_cols_ = filter_size;

	float sum;
	int n;
	int d;
	int j;
	int i;
	int i_;
	int j_;

	int inpt_index = 0;
	int outpt_index = 0;
	int weights_index = 0;

	for (n = 0; n < n_filters; n++) {
		for (d = 0; d < input_depth; d++) {

			for (j = 0; j < output_rows; j++) {
				for (i = 0; i < output_cols; i++) {
					// actual filter
					for (int j_ = 0; j_ < filter_size; j_++) {

						inpt_index += input_cols;
						weights_index += filter_size;

					}



					inpt_index += filter_stride;
					outpt_index += 1;
				}
				inpt_index -= output_cols * filter_stride;
				outpt_index -= output_cols;


				inpt_index += filter_stride * input_cols;
				outpt_index += output_cols;
			}
			inpt_index -= output_rows * filter_stride * input_cols;
			outpt_index -= output_rows * output_cols;

			inpt_index += input_cols * input_rows;
			weights_index += weights_rows * weights_cols;
			outpt_index += output_cols * output_rows;
		}
		inpt_index -= input_depth * input_cols * input_rows;
		weights_index -= input_depth * filter_size * filter_size;
		outpt_index -= input_depth * output_cols * output_rows;

		weights_index += weights_cols;
		outpt_index += output_cols * output_rows * input_depth;
	}
}


void pool_ffw(Activation* input, Activation* output, int pool_size, int pool_stride) {

	int input_depth = input->depth;

	int input_cols = input->cols;
	int input_rows = input->size / input_depth / input_cols;

	int output_cols = output->cols;
	int output_rows = output->size / input_depth / output_cols;

	float best;
	float val;

	int d;
	int j;
	int i;
	int i_;
	int j_;

	int inpt_index = 0;
	int outpt_index = 0;

	for (d = 0; d < input_depth; d++) {
		for (j = 0; j < output_rows; j++) {
			for (i = 0; i < output_cols; i++) {
				best = -10000;
				for (j_ = 0; j_ < pool_size; j_++) {
					for (i_ = 0; i_ < pool_size; i_++) {
						best = max(best, input->values[inpt_index + i_]);
					}
					inpt_index += input_cols;
				}
				output->values[outpt_index++] = best;
				inpt_index += pool_stride - input_cols * pool_size;
			}
			inpt_index += pool_stride * (input_cols - output_cols);
		}
		inpt_index += input_cols * (input_rows - pool_stride * output_rows);
	}
}
