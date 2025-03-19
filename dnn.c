#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define INPUT_SIZE 16
#define HIDDEN1_SIZE 32
#define HIDDEN2_SIZE 16
#define OUTPUT_SIZE 10


typedef struct {
    float w1[INPUT_SIZE][HIDDEN1_SIZE];
    float b1[HIDDEN1_SIZE];
    float w2[HIDDEN1_SIZE][HIDDEN2_SIZE];
    float b2[HIDDEN2_SIZE];
    float w3[HIDDEN2_SIZE][OUTPUT_SIZE];
    float b3[OUTPUT_SIZE];
} DNN;

void layer_forward(float *input, float *weights, float *bias, float *output, int in_size, int out_size);
void relu(float *x, int size);
void softmax(float *x, int size);
void init_model(DNN *model);

int main() {
    DNN model;
    float input[INPUT_SIZE] = {0};
    float hidden1[HIDDEN1_SIZE];
    float hidden2[HIDDEN2_SIZE];
    float output[OUTPUT_SIZE];

    init_model(&model);
    input[0] = 1.0;
    input[1] = 0.5;

    layer_forward(input, (float *)model.w1, model.b1, hidden1, INPUT_SIZE, HIDDEN1_SIZE);
    relu(hidden1, HIDDEN1_SIZE);

    layer_forward(hidden1, (float *)model.w2, model.b2, hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE);
    relu(hidden2, HIDDEN2_SIZE);

    layer_forward(hidden2, (float *)model.w3, model.b3, output, HIDDEN2_SIZE, OUTPUT_SIZE);
    softmax(output, OUTPUT_SIZE);

    printf("Output probabilities:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Class %d: %.4f\n", i, output[i]);
    }

    return 0;
}

void layer_forward(float *input, float *weights, float *bias, float *output, int in_size, int out_size) {
    for (int i = 0; i < out_size; i++) {
        float sum = 0.0;
        for (int j = 0; j < in_size; j++) {
            sum += input[j] * weights[j * out_size + i];
        }
        output[i] = sum + bias[i];
    }
}

void relu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}

void softmax(float *x, int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void init_model(DNN *model) {
    srand(42);

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            model->w1[i][j] = (float)rand() / RAND_MAX - 0.5;
        }
    }
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        model->b1[i] = 0.1;
    }

    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            model->w2[i][j] = (float)rand() / RAND_MAX - 0.5;
        }
    }
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        model->b2[i] = 0.1;
    }

    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            model->w3[i][j] = (float)rand() / RAND_MAX - 0.5;
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        model->b3[i] = 0.1;
    }
}
