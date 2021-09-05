#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

#include "Algo.h"
#include "Network.h"
using namespace std;


layertype trans(string s) {
	if (s == "dense") {
		return layertype::DENSE;
	}
	else if (s == "sigm") {
		return layertype::SIGM;
	}
	else if (s == "relu") {
		return layertype::RELU;
	}
	else if (s == "conv") {
		return layertype::CONV;
	}
	else if (s == "sigm") {
		return layertype::POOL;
	}
	else {
		throw invalid_argument("no layertype");
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

Dense::Dense(Activation* input_, int output_dim_, float mean, float stddev) {
	input = input_;
	tp = layertype::DENSE;

	weights = Matrix(output_dim_, input->size, mean, stddev);
	biases = Vector(output_dim_, mean, stddev);

	w_updates = Matrix(output_dim_, input->size);
	b_updates = Vector(output_dim_);
	w_updates.make_zero();
	b_updates.make_zero();
	/*Matrix w1 = Matrix("-0.050908   2.134407    0.187322    0.505207 n\
	1.008708 -1.645438   0.528933    0.583465 n\
	0.344574 -0.785388   1.078312    0.938099 n\
	-0.324690 -0.263037 -2.295243 -0.580762");
	Vector b1 = Vector("0.595911 n\
	-0.898004 n\
	0.179778 n\
	-1.4780832");
	Matrix w2 = Matrix("1.211050    -0.019860   0.794359    -0.567335 n\
		0.670572 -1.499298   0.496915 -1.080994");
	Vector b2 = Vector("-0.597926 n\
			-0.472514");
	if (output_dim_ == 2) {
		weights = w2;
		biases = b2;
	}
	else {
		weights = w1;
		biases = b1;
	}*/

	activation.update(output_dim_);
}
void Dense::FFW() {
	dense_ffw(input, &activation, &weights, &biases);
}
void Dense::Backprop() {
	dense_backprop(&activation, input, &weights, &w_updates, &b_updates);
}
void Dense::Update(float alpha) {
	dense_update(&weights, &biases, &w_updates, &b_updates, alpha);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Sigm::Sigm(Activation* input_) {
	input = input_;
	tp = layertype::SIGM;
	activation.update(input->size, input->cols, input->depth);
}
void Sigm::FFW() {
	sigm_ffw(input, &activation);
}
void Sigm::Backprop() {
	sigm_backprop(&activation, input);
}
void Sigm::Update(float alpha) {
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Relu::Relu(Activation* input_) {
	input = input_;
	tp = layertype::RELU;
	activation.update(input->size, input->cols, input->depth);
}
void Relu::FFW() {
	relu_ffw(input, &activation);
}
void Relu::Backprop() {
	relu_backprop(&activation, input);
}
void Relu::Update(float alpha) {
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Conv::Conv(Activation* input_, int filter_stride_, int filter_size_, int n_filters_, int padding_, float mean, float stddev) :
	filter_stride(filter_stride_), filter_size(filter_size_), n_filters(n_filters), padding(padding_) {
	tp = layertype::CONV;
	input = input_;

	int input_rows = (input->size / input->cols) / input->depth;
	int output_rows = (input_rows + 2 * padding - filter_size) / filter_stride;
	int output_cols = (input->cols + 2 * padding - filter_size) / filter_stride;
	int output_depth = input->depth * n_filters;

	if ((input_rows + 2 * padding - filter_size) % filter_stride != 0 ||
		(input->cols + 2 * padding - filter_size) % filter_stride != 0 ||
		(input->size % input->cols) != 0 ||
		(input->size / input->cols) % input->depth != 0 ) {
		cerr << "Invalid size/stride/padding combination" << endl;
		throw invalid_argument("");
	}

	weights = Matrix(n_filters, filter_size * filter_size * input->depth, mean, stddev);
	biases = Vector(n_filters, mean, stddev);
	w_updates = Matrix(n_filters, filter_size * n_filters * input->depth);
	b_updates = Vector(n_filters);
	w_updates.make_zero();
	b_updates.make_zero();

	activation.update(output_rows, output_cols, output_depth);
}
void Conv::FFW() {
	//conv_ffw(input, &activation, &weights, &biases, filter_size, n_filters, filter_stride, padding);
}
void Conv::Backprop() {

}
void Conv::Update(float alpha) {

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Pool::Pool(Activation* input_, int pool_stride_, int pool_size_) {
	tp = layertype::POOL;
	pool_stride = pool_stride_;
	pool_size = pool_size_;
	input = input_;
	int input_rows = (input->size / input->cols) / input->depth;

	int output_rows = (input_rows - pool_size) / pool_stride;
	int output_cols = (input->cols - pool_size) / pool_stride;
	int output_depth = input->depth;
	if ((input_rows - pool_size) % pool_stride != 0 ||
		(input->cols - pool_size) % pool_stride != 0 ||
		(input->size % input->cols) != 0 ||
		(input->size / input->cols) % input->depth != 0) {
		cerr << "Invalid size/stride/padding combination" << endl;
		throw invalid_argument("");
	}
	activation.update(output_rows, output_cols, output_depth);
}
void Pool::FFW() {
	pool_ffw(input, &activation, pool_size, pool_stride);
}
void Pool::Backprop() {

}
void Pool::Update(float alpha) {

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Network::Network(string design, int input_size_, int input_cols_, int input_depth_) :
	input(input_size_, input_cols_, input_depth_), input_size(input_size_), input_cols(input_cols_), input_depth(input_depth_)  {
	istringstream iss(design);
	string temp;
	n_layers = 0;

	Activation* prev = &input;

	while (iss >> temp) {
		n_layers++;
		switch (trans(temp)) {
		case layertype::DENSE: {
			iss >> temp;
			Dense* p_tempobj = new Dense(prev, stoi(temp));
			layers.emplace_back(p_tempobj);
			prev = &layers.back()->activation;
			break;
		}
		case layertype::SIGM: {
			Sigm* p_tempobj2 = new Sigm(prev);
			layers.emplace_back(p_tempobj2);
			prev = &layers.back()->activation;
			break;
		}
		case layertype::RELU: {
			Relu* p_tempobj3 = new Relu(prev);
			layers.emplace_back(p_tempobj3);
			prev = &layers.back()->activation;
			break;
		}
		case layertype::CONV: {
			iss >> temp;
			int filter_stride = stoi(temp);
			iss >> temp;
			int filter_size = stoi(temp);
			iss >> temp;
			int n_filters = stoi(temp);
			Conv* p_tempobj = new Conv(prev, filter_stride, filter_size, n_filters);
			layers.emplace_back(p_tempobj);
			prev = &layers.back()->activation;
			break;
		}
		case layertype::POOL: {
			iss >> temp;
			int pool_stride = stoi(temp);
			iss >> temp;
			int pool_size = stoi(temp);
			Pool* p_tempobj = new Pool(prev, pool_stride, pool_size);
			layers.emplace_back(p_tempobj);
			prev = &layers.back()->activation;
			break;
		}
		}
	}
	output_dim = layers.back()->activation.size;
	output_values = layers.back()->activation.values;
}
Network::~Network() {
	for (auto& layer : layers) {
		delete layer;
	}
}
int Network::GetResult() {
	for (auto& layer : layers) { 
		layer->FFW();
	}
	output_values = layers.back()->activation.values;
	return max_element(output_values, output_values + output_dim) - output_values;
}
void Network::Test_accuracy(uint8_t* testdata, uint8_t* testlabels, int sz) {
	int correct = 0;
	for (int i = 0; i < sz; i++) {
		if (i % 1000 == 0) {
			cerr << i << endl;
		}
		input.update(&testdata[i * input_size]);
		correct += (GetResult() == testlabels[i]);
	}
	double percentage = correct / (sz / 100.0);
	cout << "Got " << correct << " correct, this is " << percentage << " percent" << endl;
}
void Network::ApplyUpdates(float alpha) {
	for (auto& layer : layers) { layer->Update(alpha); }
}
void Network::Change_updates(int label) {
	for (auto& layer : layers) { layer->FFW(); }
	layers.back()->activation.values[label] -= 1;
	for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer) { (*layer)->Backprop(); }
}
void Network::Batch_update(int batch_size, float eta, uint8_t* traindata, uint8_t* trainlabels) {
	Matrix temp;
	int adress;
	for (int i = 0; i < batch_size; i++) {
		adress = random60000();
		input.update(&traindata[adress * input_size]);
		Change_updates(trainlabels[adress]);
	}
	ApplyUpdates(eta / batch_size);
}


