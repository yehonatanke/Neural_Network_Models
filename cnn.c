#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 28
#define KERNEL_SIZE 3
#define OUTPUT_SIZE 10
#define FEATURE_MAPS 4


typedef struct {
    float kernel[FEATURE_MAPS][KERNEL_SIZE][KERNEL_SIZE];
    float bias[FEATURE_MAPS];
    float fc_weights[FEATURE_MAPS * 26 * 26][OUTPUT_SIZE];
    float fc_bias[OUTPUT_SIZE];
} CNN;

void conv_layer(float input[INPUT_SIZE][INPUT_SIZE], float output[FEATURE_MAPS][26][26], CNN *model);
void relu(float *x, int size);
void fully_connected(float input[FEATURE_MAPS * 26 * 26], float output[OUTPUT_SIZE], CNN *model);
void softmax(float *x, int size);
void init_model(CNN *model);

int main() {
    CNN model;
    float input[INPUT_SIZE][INPUT_SIZE] = {0};
    float conv_output[FEATURE_MAPS][26][26];
    float fc_input[FEATURE_MAPS * 26 * 26];
    float output[OUTPUT_SIZE];

    init_model(&model);
    input[14][14] = 1.0;

    conv_layer(input, conv_output, &model);
    for (int f = 0; f < FEATURE_MAPS; f++) {
        relu((float *)conv_output[f], 26 * 26);
    }

    for (int f = 0; f < FEATURE_MAPS; f++) {
        for (int i = 0; i < 26; i++) {
            for (int j = 0; j < 26; j++) {
                fc_input[f * 26 * 26 + i * 26 + j] = conv_output[f][i][j];
            }
        }
    }

    fully_connected(fc_input, output, &model);
    softmax(output, OUTPUT_SIZE);

    printf("Output probabilities:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Class %d: %.4f\n", i, output[i]);
    }

    return 0;
}

void conv_layer(float input[INPUT_SIZE][INPUT_SIZE], float output[FEATURE_MAPS][26][26], CNN *model) {
    int out_size = INPUT_SIZE - KERNEL_SIZE + 1;

    for (int f = 0; f < FEATURE_MAPS; f++) {
        for (int i = 0; i < out_size; i++) {
            for (int j = 0; j < out_size; j++) {
                float sum = 0.0;
                for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                    for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                        sum += input[i + ki][j + kj] * model->kernel[f][ki][kj];
                    }
                }
                output[f][i][j] = sum + model->bias[f];
            }
        }
    }
}

void relu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}

void fully_connected(float input[FEATURE_MAPS * 26 * 26], float output[OUTPUT_SIZE], CNN *model) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = 0.0;
        for (int j = 0; j < FEATURE_MAPS * 26 * 26; j++) {
            sum += input[j] * model->fc_weights[j][i];
        }
        output[i] = sum + model->fc_bias[i];
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

void init_model(CNN *model) {
    srand(42);

    for (int f = 0; f < FEATURE_MAPS; f++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                model->kernel[f][i][j] = (float)rand() / RAND_MAX - 0.5;
            }
        }
        model->bias[f] = 0.1;
    }

    for (int i = 0; i < FEATURE_MAPS * 26 * 26; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            model->fc_weights[i][j] = (float)rand() / RAND_MAX - 0.5;
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        model->fc_bias[i] = 0.1;
    }
}
