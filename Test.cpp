
#include "Matrix.h"
#include "Network.h"
#include "Algo.h"
#include <time.h>
#include <algorithm>
#include <fstream>
#include <cstring>

using namespace std;


void MNIST_test() {
    //std::cerr.setstate(std::ios_base::failbit);
    clock_t t;
    t = clock();
    srand(time(NULL));
    fstream f1;
    uint8_t* test_labels = new uint8_t[10000];
    uint8_t* train_labels = new uint8_t[60000];
    uint8_t* test_data = new uint8_t[10000 * 784];
    uint8_t* train_data = new uint8_t[60000 * 784];
    int magic_number;
    int nr_items;
    int row_size;
    int col_size;

    f1.open("C:\\Users\\jonel\\OneDrive\\Documenten\\Programmas\\newnn\\t10k-labels-idx1-ubyte", ios::in | ios::binary);
    f1.read((char*)&magic_number, sizeof(magic_number)); // 2049
    f1.read((char*)&nr_items, sizeof(nr_items)); // 10000
    for (int i = 0; i < 10000; ++i) {
        f1.read((char*)&(test_labels[i]), 1);
    }
    f1.close();

    f1.open("C:\\Users\\jonel\\OneDrive\\Documenten\\Programmas\\newnn\\train-labels-idx1-ubyte", ios::in | ios::binary);
    f1.read((char*)&magic_number, sizeof(magic_number)); // 2049
    f1.read((char*)&nr_items, sizeof(nr_items)); // 60000
    for (int i = 0; i < 60000; ++i) {
        f1.read((char*)&(train_labels[i]), 1);
    }
    f1.close();


    f1.open("C:\\Users\\jonel\\OneDrive\\Documenten\\Programmas\\newnn\\t10k-images-idx3-ubyte", ios::in | ios::binary);
    f1.read((char*)&magic_number, sizeof(magic_number)); // 2051
    f1.read((char*)&nr_items, sizeof(nr_items)); // 10000
    f1.read((char*)&col_size, sizeof(col_size)); // 28
    f1.read((char*)&row_size, sizeof(row_size)); // 28
    for (int i = 0; i < 10000; ++i) {
        f1.read((char*)&(test_data[784 * i]), 784);
    }
    f1.close();

    f1.open("C:\\Users\\jonel\\OneDrive\\Documenten\\Programmas\\newnn\\train-images-idx3-ubyte", ios::in | ios::binary);
    f1.read((char*)&magic_number, sizeof(magic_number)); // 2051
    f1.read((char*)&nr_items, sizeof(nr_items)); // 60000
    f1.read((char*)&col_size, sizeof(col_size)); // 28
    f1.read((char*)&row_size, sizeof(row_size)); // 28
    for (int i = 0; i < 60000; ++i) {
        f1.read((char*)&(train_data[784 * i]), 784);
    }
    f1.close();


    double time_taken;
    Network mynet("dense 30 sigm dense 10 sigm", 784);
    mynet.Test_accuracy(test_data, test_labels, 1);
    for (int i = 0; i < 30; i++) {
        for (int j = 0; j < 6000; j++) {
            mynet.Batch_update(10, 2, train_data, train_labels);
        }
        mynet.Test_accuracy(test_data, test_labels, 10000);
        time_taken = ((double)(clock() - t)) / CLOCKS_PER_SEC;
        cerr << "time taken: " << time_taken << endl;
    }

    delete[] test_labels;
    delete[] train_labels;
    delete[] test_data;
    delete[] train_data;
    time_taken = ((double)(clock() - t)) / CLOCKS_PER_SEC;
    cerr << "time taken: " << time_taken << endl;
    //std::cerr.clear();
}



void pool_ffw_test() {
    const int length = 4 * 2 * 6;
    Activation inpt(length, 4, 2);
    float stack_array[length] = { 9, 6, 3, 9,\
    8, 8, 5, 8,\
    7, 9, 2, 0,\
    6, 9, 1, 8,\
    0, 5, 0, 7,\
    4, 2, 3, 7,\
    0, 7, 3, 4,\
    4, 4, 9, 8,\
    7, 5, 8, 3,\
    9, 4, 6, 8,\
    8, 4, 6, 6,\
    2, 1, 6, 4};
    std::copy(stack_array, stack_array + length, &inpt.values[0]);


    const int length2 = 3 * 3 * 2 * 2;
    float weights[length2] = { 1 , 0, 0 , 0, 0, 2, 0, 0, 0, \
        0, 3, 0, 1, 0, 0, 1, 0, 0, \
        2, 0, 0, 0, 0, 4, 0, 0, 0, \
        0, 6, 0, 2, 0, 0, 2, 0, 0 };
    Matrix weigs(2, 3 * 3 * 2);
    std::copy(weights, weights + length2, &weigs.values[0]);

    const int length3 = 2;
    float a[length3] = { 200000, 100000};
    Vector b(2);
    std::copy(a, a + length3, &b.values[0]);

    Activation output(2 * 4 * 2, 2, 2);


    int filter_size = 3;
    int filter_stride = 1;
    int n_filters = 2;

    conv_ffw1(&inpt, &output, &weigs, &b, filter_size, n_filters, filter_stride, 0);
    conv_ffw2(&inpt, &output, &weigs, &b, filter_size, n_filters, filter_stride, 0);
    output.print();
    cerr << "finished" << endl;
    
}

int main() {
    //Hardcode_test();
    //MNIST_test();
    pool_ffw_test();
}