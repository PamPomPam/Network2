#pragma once

#include <random>
#include <iostream>
#include <string>
#include <math.h>
#include <time.h>
#include <cstring>

using namespace std;


struct Activation {
    float* values;
    int size;

    int cols;
    int depth;

    Activation();
    Activation(int size_, int cols_ = -1, int depth_ = -1);
    ~Activation();
    Activation(const Activation& that);
    Activation& operator=(const Activation& that);
    void update(uint8_t arr[]);
    void update(int size_, int cols_ = -1, int depth_ = -1);
    void update(const Activation& that);

    void print();
    void printout();
    void print_image();
    void shape();

};



struct Vector {
    float* values;
    int rows;
    inline float& at(int j) {
        return values[j];
    }
    Vector();
    Vector(int rows_);
    Vector(int rows_, float mean, float stddev);
    Vector(string input);
    ~Vector();

    Vector(const Vector& that);
    Vector& operator=(const Vector& that);

    void print();
    void printout();
    void shape();

    void initialize(float mean, float stddev);
    void make_zero();
};


struct Matrix {

    float* values;
    int rows;
    int cols;

    inline float& at(int row, int col) {return values[row * cols + col];}

    Matrix();
    Matrix(int rows_, int cols_);
    Matrix(int rows_, int cols_, float mean, float stddev);
    Matrix(string input);
    ~Matrix();
    Matrix(const Matrix& that);
    Matrix& operator=(const Matrix& that);

    void print();
    void printout();
    void shape();
    
    void initialize(float mean, float stddev);
    void make_zero();
};

struct Cube {
    float* values;
    int rows;
    int cols;
    int depth;

    inline float& at(int j, int i, int d) {return values[d * cols * rows + j * cols + i];}

    Cube();
    Cube(int rows_, int cols_, int depth_);
    Cube(int rows_, int cols_, int depth_, float mean, float stddev);
    ~Cube();

    Cube(const Cube& that);
    Cube& operator=(const Cube& that);

    void print();
    void printout();
    void shape();

    void initialize(float mean, float stddev);
    void make_zero();
};

struct Tesseract {
    float* values;
    int rows;
    int cols;
    int depth;
    int n_depth;

    inline float& at(int j, int i, int d, int n) { return values[n * depth * cols * rows + d * cols * rows + j * cols + i]; }

    Tesseract();
    Tesseract(int rows_, int cols_, int depth_, int n_depth_);
    Tesseract(int rows_, int cols_, int depth_, int n_depth_, float mean, float stddev);
    ~Tesseract();

    Tesseract(const Tesseract& that);
    Tesseract& operator=(const Tesseract& that);

    //void print();
    //void printout();
    void shape();

    void initialize(float mean, float stddev);
    void make_zero();
};
