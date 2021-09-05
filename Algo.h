#pragma once

#include "Matrix.h"

void dense_ffw(Activation* input, Activation* output, Matrix* weights, Vector* biases);
void dense_backprop(Activation* output, Activation* input, Matrix* weights, Matrix* w_updates, Vector* b_updates);
void dense_update(Matrix* weights, Vector* biases, Matrix* w_updates, Vector* b_updates, float alpha);

void sigm_ffw(Activation* input, Activation* output);
void sigm_backprop(Activation* output, Activation* input);

void relu_ffw(Activation* input, Activation* output);
void relu_backprop(Activation* output, Activation* input);


void conv_ffw1(Activation* input, Activation* output, Matrix* weights, Vector* biases, int filter_size, int n_filters, int filter_stride, int padding);

void conv_ffw2(Activation* input, Activation* output, Matrix* weights, Vector* biases, int filter_size, int n_filters, int filter_stride, int padding);


void pool_ffw(Activation* input, Activation* output, int pool_size, int pool_stride);

inline float sigmoid(float x) { return 1 / (1 + exp(-x)); }
inline float d_sigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }
inline int random60000() { return rand() % 300 + (rand() % 200) * 300; }

