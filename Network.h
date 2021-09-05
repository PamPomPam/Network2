#pragma once
#pragma once
#include "Matrix.h"
#include <vector>
#include <string>

using namespace std;



enum class layertype { DENSE, SIGM, RELU, CONV, POOL, NONE };
layertype trans(string s);

struct Layer {
	Activation* input = nullptr;
	Activation activation; // will change into desired at backprop
	layertype tp = layertype::NONE;

	Matrix weights; // purely for tests
	Vector biases; // purely for tests

	virtual void FFW() = 0;
	virtual void Backprop() = 0;
	virtual void Update(float alpha) = 0;
};

struct Dense : Layer {
	Matrix weights;
	Vector biases;

	Matrix w_updates;
	Vector b_updates;

	Dense(Activation* input_, int output_dim_, float mean = 0, float stddev = 1);
	void FFW() override;
	void Backprop() override;
	void Update(float alpha) override;
};

struct Sigm : Layer {
	Sigm(Activation* input_);
	void FFW() override;
	void Backprop() override;
	void Update(float alpha) override;
};

struct Relu : Layer {
	Relu(Activation* input_);
	void FFW() override;
	void Backprop() override;
	void Update(float alpha) override;
};

struct Conv : Layer {
	int n_filters;
	int filter_size; // all filters are squares
	int filter_stride;
	int padding;

	Matrix weights; // contains all weights, so size = filter_size**2  * n_filters
	Vector biases;
	Matrix w_updates;
	Vector b_updates;

	Conv(Activation* input_, int filter_stride_, int filter_size_, int n_filters_, int padding_ = 2, float mean = 0, float stddev = 1);
	void FFW() override;
	void Backprop() override;
	void Update(float alpha) override;
};

struct Pool : Layer {
	int pool_size;
	int pool_stride;

	Pool(Activation* input_, int pool_stride_, int pool_size_);
	void FFW() override;
	void Backprop() override;
	void Update(float alpha) override;
};


struct Network {
	vector<Layer*> layers;
	int n_layers;
	int input_size;
	int input_cols;
	int input_depth;
	Activation input;
	int output_dim;
	float* output_values;

	Network(string design, int input_size_, int input_cols_=-1, int input_depth_=-1);
	~Network();
	int GetResult();
	void Test_accuracy(uint8_t* testdata, uint8_t* testlabels, int sz);
	void ApplyUpdates(float alpha);
	void Change_updates(int label);
	void Batch_update(int batch_size, float eta, uint8_t* traindata, uint8_t* trainlabels);
};
