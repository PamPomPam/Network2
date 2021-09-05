
#include "Matrix.h"
#include <stdexcept>
#include <string>
#include <time.h>
#include <sstream>

unsigned seed = time(NULL);
std::default_random_engine generator(seed);


Activation::Activation() : size(-1), cols(-1), depth(-1), values(nullptr) {}
Activation::Activation(int size_, int cols_, int depth_) : size(size_), cols(cols_), depth(depth_) {
    values = new float[size];
}
Activation::~Activation() {
    cout << size << endl;
    delete[] values;
}
Activation::Activation(const Activation& that) {
    cout << "Activations cannot be copied" << endl;
    throw invalid_argument("");
}
Activation& Activation::operator=(const Activation& that) {
    cout << "Activations cannot be copied" << endl;
    throw invalid_argument("");
}
void Activation::update(uint8_t arr[]) { // for mnist data, assumes size, cols and depth are already right
    delete[] values;
    values = new float[size];
    for (unsigned int i = 0; i < size; i++) {
        values[i] = arr[i] / 256.0;
    }
}
void Activation::update(int size_, int cols_, int depth_) {
    size = size_;
    cols = cols_;
    depth = depth_;
    values = new float[size];
}
void Activation::update(const Activation& that) {
    values = new float[that.size];
    memcpy(values, that.values, that.size * sizeof(that.values[0]));
}
void Activation::print() {
    cerr << "---------------" << endl;
    string str;
    for (int j = 0; j < size; j++) {
        str = std::to_string(values[j]);
        cerr << str << endl;
    }
    cerr << "---------------" << endl << endl;
}
void Activation::printout() {
    cout << "---------------" << endl;
    string str;
    for (int j = 0; j < size; j++) {
        str = std::to_string(values[j]);
        cout << str << endl;
    }
    cout << "---------------" << endl << endl;
}
void Activation::print_image() {
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            if (values[28 * y + x] > 0.5) {
                cerr << 'x';
            }
            else {
                cerr << ' ';
            }
        }
        cerr << endl;
    }
}
void Activation::shape() {
    cerr << "shape: " << size << endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Vector::Vector() {
    rows = 0;
    values = nullptr;
}
Vector::Vector(int rows_) : rows(rows_) {
    values = new float[rows];
}
Vector::Vector(int rows_, float mean, float stddev) : rows(rows_) {
    values = new float[rows];
    initialize(mean, stddev);
}
Vector::Vector(string input) {
    istringstream iss(input);
    vector<double> temp;
    string temporary;

    rows = 0;
    while (getline(iss, temporary, 'n')) {
        rows++;
        istringstream iss2(temporary);
        while (iss2 >> temporary) {
            temp.emplace_back(stod(temporary));
        }
    }
    values = new float[rows];
    for (unsigned int i = 0; i < rows; i++) {
        values[i] = temp[i];
    }
}
Vector::~Vector() {
    delete[] values;
}
Vector::Vector(const Vector& that) : rows(that.rows) {
    values = new float[that.rows];
    memcpy(values, that.values, that.rows* sizeof(that.values[0]));
}
Vector& Vector::operator=(const Vector& that)
{
    delete[] values;
    values = new float[that.rows];
    memcpy(values, that.values, that.rows * sizeof(that.values[0]));
    rows = that.rows;
    return *this;
}
void Vector::print() {
    cerr << "---------------" << endl;
    string str;
    for (int j= 0; j < rows; j++) {
        str = std::to_string(at(j));
        cerr << str << endl;
    }
    cerr << "---------------" << endl << endl;
}
void Vector::printout() {
    cout << "---------------" << endl;
    string str;
    for (int j = 0; j < rows; j++) {
        str = std::to_string(at(j));
        cout << str << endl;
    }
    cout << "---------------" << endl << endl;
}
void Vector::shape() {
    cerr << "shape: " << rows << endl;
}
void Vector::initialize(float mean, float stddev) {
    std::normal_distribution<float> distribution(mean, stddev);
    for (int j = 0; j < rows; j++) {
        values[j] = distribution(generator);
    }
}
void Vector::make_zero() {
    memset(values, 0, rows * sizeof(values[0]));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Matrix::Matrix() {
    rows = 0;
    cols = 0;
    values = nullptr;
}
Matrix::Matrix(int rows_, int cols_) : rows(rows_), cols(cols_) {
    values = new float[rows * cols];
    //memset(values, 0, sizeof(values));
}
Matrix::Matrix(int rows_, int cols_, float mean, float stddev) : rows(rows_), cols(cols_) {
    values = new float[rows * cols];
    initialize(mean, stddev);
}
Matrix::Matrix(string input) {
    istringstream iss(input);
    vector<double> temp;
    string temporary;

    rows = 0;
    while (getline(iss, temporary, 'n')) {
        rows++;
        istringstream iss2(temporary);
        while (iss2 >> temporary) {
            temp.emplace_back(stod(temporary));
        }
    }
    cols = temp.size() / rows;
    values = new float[rows * cols];
    for (unsigned int i = 0; i < cols * rows; i++) {
        values[i] = temp[i];
    }
}
Matrix::~Matrix() {
    delete[] values;
}
Matrix::Matrix(const Matrix& that) : rows(that.rows), cols(that.cols) {
    values = new float[that.rows * that.cols];
    memcpy(values, that.values, that.rows * that.cols * sizeof(that.values[0]));
}
Matrix& Matrix::operator=(const Matrix& that)
{
    delete[] values;
    values = new float[that.rows * that.cols];
    memcpy(values, that.values, that.rows * that.cols * sizeof(that.values[0]));
    cols = that.cols;
    rows = that.rows;
    return *this;
}
void Matrix::print() {
    cerr << "---------------" << endl;
    string str;
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            str = std::to_string(at(y, x));
            cerr << str;
            for (unsigned int i = 0; i < (12 - str.length()); i++) { cerr << ' '; }
        }
        cerr << endl;
    }
    cerr << endl;
}
void Matrix::printout() {
    cout << "---------------" << endl;
    string str;
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            str = std::to_string(at(y, x));
            cout << str;
            for (unsigned int i = 0; i < (12 - str.length()); i++) { cerr << ' '; }
        }
        cout << endl;
    }
    cout << endl;
}
void Matrix::shape() {
    cerr << "shape: " << rows << 'x' << cols << endl;
}
void Matrix::initialize(float mean, float stddev) {
    std::normal_distribution<float> distribution(mean, stddev);
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            values[y * cols + x] = distribution(generator);
        }
    }
}
void Matrix::make_zero() {
    memset(values, 0, cols * rows * sizeof(values[0]));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Cube::Cube() {
    rows = -1;
    cols = -1;
    depth = -1;
    values = nullptr;
}
Cube::Cube(int rows_, int cols_, int depth_) : rows(rows_), cols(cols_), depth(depth_) {
    values = new float[rows * cols * depth];
    //memset(values, 0, sizeof(values));
}
Cube::Cube(int rows_, int cols_, int depth_, float mean, float stddev) : rows(rows_), cols(cols_), depth(depth_) {
    values = new float[rows * cols * depth];
    initialize(mean, stddev);
}
Cube::~Cube() {
    delete[] values;
}
Cube::Cube(const Cube& that) : rows(that.rows), cols(that.cols), depth(that.depth) {
    values = new float[that.rows * that.cols * that.depth];
    memcpy(values, that.values, that.rows * that.cols * that.depth * sizeof(that.values[0]));
}
Cube& Cube::operator=(const Cube& that)
{
    delete[] values;
    values = new float[that.rows * that.cols * that.depth];
    memcpy(values, that.values, that.rows * that.cols * that.depth * sizeof(that.values[0]));
    cols = that.cols;
    rows = that.rows;
    depth = that.depth;
    return *this;
}
void Cube::print() {
    cerr << "---------------" << endl;
    string str;
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            for (int d = 0; d < depth; d++) {
                str = std::to_string(at(j, i, d));
                cerr << str;
                for (unsigned int _ = 0; _ < (12 - str.length()); _++) { cerr << ' '; }
            }
            cerr << endl;
        }
        cerr << endl << endl;
    }
    cerr << "---------------" << endl;
}
void Cube::printout() {
    cout << "---------------" << endl;
    string str;
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            for (int d = 0; d < depth; d++) {
                str = std::to_string(at(j, i, d));
                cout << str;
                for (unsigned int _ = 0; _ < (12 - str.length()); _++) { cout << ' '; }
            }
            cout << endl;
        }
        cout << endl << endl;
    }
    cout << "---------------" << endl;
}
void Cube::shape() {
    cerr << "shape: " << rows << 'x' << cols << 'x' << depth << endl;
}
void Cube::initialize(float mean, float stddev) {
    std::normal_distribution<float> distribution(mean, stddev);
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            for (int d = 0; d < depth; d++) {
                at(j, i, d) = distribution(generator);
            }
        }
    }
}
void Cube::make_zero() {
    memset(values, 0, cols * rows * depth * sizeof(values[0]));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Tesseract::Tesseract() {
    rows = -1;
    cols = -1;
    depth = -1;
    values = nullptr;
}
Tesseract::Tesseract(int rows_, int cols_, int depth_, int n_depth_) : rows(rows_), cols(cols_), depth(depth_), n_depth(n_depth_) {
    values = new float[rows * cols * depth * n_depth];
    //memset(values, 0, sizeof(values));
}
Tesseract::Tesseract(int rows_, int cols_, int depth_, int n_depth_, float mean, float stddev) : rows(rows_), cols(cols_), depth(depth_), n_depth(n_depth_) {
    values = new float[rows * cols * depth * n_depth];
    initialize(mean, stddev);
}
Tesseract::~Tesseract() {
    delete[] values;
}
Tesseract::Tesseract(const Tesseract& that) : rows(that.rows), cols(that.cols), depth(that.depth), n_depth(that.n_depth) {
    values = new float[that.rows * that.cols * that.depth * n_depth];
    memcpy(values, that.values, that.rows * that.cols * that.depth * that.n_depth * sizeof(that.values[0]));
}
Tesseract& Tesseract::operator=(const Tesseract& that)
{
    delete[] values;
    values = new float[that.rows * that.cols * that.depth * n_depth];
    memcpy(values, that.values, that.rows * that.cols * that.depth * that.n_depth * sizeof(that.values[0]));
    cols = that.cols;
    rows = that.rows;
    depth = that.depth;
    n_depth = that.n_depth;
    return *this;
}
/*void Tesseract::print() {
    cerr << "---------------" << endl;
    string str;
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            for (int d = 0; d < depth; d++) {
                str = std::to_string(at(j, i, d));
                cerr << str;
                for (unsigned int _ = 0; _ < (12 - str.length()); _++) { cerr << ' '; }
            }
            cerr << endl;
        }
        cerr << endl << endl;
    }
    cerr << "---------------" << endl;
}
void Tesseract::printout() {
    cout << "---------------" << endl;
    string str;
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            for (int d = 0; d < depth; d++) {
                str = std::to_string(at(j, i, d));
                cout << str;
                for (unsigned int _ = 0; _ < (12 - str.length()); _++) { cout << ' '; }
            }
            cout << endl;
        }
        cout << endl << endl;
    }
    cout << "---------------" << endl;
}*/
void Tesseract::shape() {
    cerr << "shape: " << rows << 'x' << cols << 'x' << depth << 'x'<< n_depth << endl;
}
void Tesseract::initialize(float mean, float stddev) {
    std::normal_distribution<float> distribution(mean, stddev);
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            for (int d = 0; d < depth; d++) {
                for (int n = 0; n < n_depth; n++) {
                    at(j, i, d, n) = distribution(generator);
                }         
            }
        }
    }
}
void Tesseract::make_zero() {
    memset(values, 0, cols * rows * depth * n_depth * sizeof(values[0]));
}
