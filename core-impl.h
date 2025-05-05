#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <compute-gradients.h>

// Error checking macros
#define CHECK_CUDA_ERROR(call)                                                \
    do                                                                        \
    {                                                                         \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CHECK_CUBLAS_ERROR(call)                                                \
    do                                                                          \
    {                                                                           \
        cublasStatus_t status = call;                                           \
        if (status != CUBLAS_STATUS_SUCCESS)                                    \
        {                                                                       \
            fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, \
                    status);                                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Class definition for our CUDA SGD implementation
class CudaSGD
{
private:
    // Data dimensions and model parameters
    int numSamples;
    int numFeatures;
    float initialLearningRate;
    float exponent;
    int batchSize;
    int maxEpochs;

    // Device data pointers
    float *d_features;
    float *d_features_transposed;
    float *d_labels;
    float *d_weights;
    float *d_gradients;

    // Host data
    float *h_weights;

    // CUDA handles and streams
    cublasHandle_t cublasHandle;

public:
    CudaSGD(int numSamples, int numFeatures, float initialLearningRate, int batchSize, int maxEpochs, float exponent = 0.25f);
    ~CudaSGD();

    void loadData(const float *h_features, const float *h_features_transposed, const float *h_labels);
    void getWeights(float *h_weights_out);

    // Training methods with different optimization strategies
    void trainWithCuBLAS();
    void trainWithCustomKernels();
    void trainWithMemoryCoalescing();
    void trainAsynchronous();

    void train(const std::string &method = "optimized");
    void benchmark();
};

CudaSGD::CudaSGD(int numSamples, int numFeatures, float initialLearningRate,
                 int batchSize, int maxEpochs, float exponent)
    : numSamples(numSamples), numFeatures(numFeatures),
      initialLearningRate(initialLearningRate), batchSize(batchSize),
      maxEpochs(maxEpochs), exponent(exponent)
{
    printf("Initializing CudaSGD with %d samples and %d features...\n", numSamples, numFeatures);
    // Allocate device memory for weights and gradients
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_weights, numFeatures * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_gradients, numFeatures * sizeof(float)));

    // Create cuBLAS handle
    printf("[INFO] Creating cuBLAS handle...\n");
    CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));

    // Initialize weights with a normal distribution
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.1f); // Mean 0, StdDev 0.1

    std::vector<float> initialWeights(numFeatures);
    for (int i = 0; i < numFeatures; ++i)
    {
        initialWeights[i] = distribution(generator);
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, initialWeights.data(), numFeatures * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize host weights buffer
    h_weights = new float[numFeatures];

    printf("[INFO] CudaSGD initialized with %d samples and %d features\n", numSamples, numFeatures);
    printf("[INFO] Learning rate: %.6f, Batch size: %d, Max epochs: %d\n",
           initialLearningRate, batchSize, maxEpochs);
}

CudaSGD::~CudaSGD()
{
    printf("[INFO] Cleaning up CudaSGD resources...\n");

    // Destroy cuBLAS handle
    CHECK_CUBLAS_ERROR(cublasDestroy(cublasHandle));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_features));
    CHECK_CUDA_ERROR(cudaFree(d_features_transposed));
    CHECK_CUDA_ERROR(cudaFree(d_labels));
    CHECK_CUDA_ERROR(cudaFree(d_weights));
    CHECK_CUDA_ERROR(cudaFree(d_gradients));

    // Free host memory
    delete[] h_weights;
    printf("[INFO] CudaSGD resources cleaned up successfully.\n");
}

void CudaSGD::loadData(const float *h_features, const float *h_features_transposed, const float *h_labels)
{
    printf("[INFO] Loading data to device...\n");
    // Allocate device memory for features and labels
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_features, numSamples * numFeatures * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_features_transposed, numSamples * numFeatures * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_labels, numSamples * sizeof(float)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_features, h_features,
                                numSamples * numFeatures * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_features_transposed, h_features_transposed,
                                numSamples * numFeatures * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels,
                                numSamples * sizeof(float),
                                cudaMemcpyHostToDevice));

    printf("[INFO] Data loaded to device successfully.\n");
}

void CudaSGD::getWeights(float *h_weights_out)
{
    printf("[INFO] Copying weights from device to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(h_weights_out, d_weights,
                                numFeatures * sizeof(float),
                                cudaMemcpyDeviceToHost));
    printf("[INFO] Weights copied to host successfully.\n");
}

void CudaSGD::trainWithCuBLAS()
{
    printf("[INFO] Starting training with cuBLAS...\n");

    // Allocate device memory
    float *d_predictions;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_predictions, batchSize * sizeof(float)));

    // Constants for cuBLAS operations
    float learningRate = initialLearningRate;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float neg_alpha = -1.0f;

    for (int epoch = 0; epoch < maxEpochs; epoch++)
    {
        printf("[INFO] Starting epoch %d of %d...\n", epoch + 1, maxEpochs);
        for (int batchStart = 0; batchStart < numSamples; batchStart += batchSize)
        {
            int currentBatchSize = std::min(batchSize, numSamples - batchStart);

            // Zero accumulated gradient
            CHECK_CUDA_ERROR(cudaMemset(d_gradients, 0, numFeatures * sizeof(float)));

            // 1. Forward pass: compute predictions (X_batch * w)
            CHECK_CUBLAS_ERROR(cublasSgemm(cublasHandle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           1, currentBatchSize, numFeatures,
                                           &alpha,
                                           d_weights, 1,
                                           d_features, numFeatures,
                                           &beta,
                                           d_predictions, 1););

            // 2. Compute errors (predictions - y_batch)
            CHECK_CUBLAS_ERROR(cublasSaxpy(cublasHandle, currentBatchSize, &neg_alpha, d_labels, 1, d_predictions, 1));

            // 3. Compute gradients (X_batch^T * errors)
            CHECK_CUBLAS_ERROR(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 1, numFeatures, currentBatchSize,
                                           &alpha, d_predictions, 1, d_features, numFeatures, &beta, d_gradients, 1));

            // 5. Update weights (w = w - lr * gradients)
            float scale = -learningRate / currentBatchSize;
            CHECK_CUBLAS_ERROR(cublasSaxpy(cublasHandle, numFeatures, &scale, d_gradients, 1, d_weights, 1));
        }

        // Update learning rate using inverse scaling
        learningRate = initialLearningRate / pow(epoch + 1, exponent);
    }

    // Cleanup
    printf("[INFO] Cleaning up resources...\n");
    CHECK_CUDA_ERROR(cudaFree(d_predictions));
    printf("[INFO] cuBLAS training completed successfully.\n");
}

void CudaSGD::trainWithCustomKernels()
{
    printf("[INFO] Starting training with custom kernels...\n");
    float *d_predictions;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_predictions, batchSize * sizeof(float)));
    float *d_mse;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_mse, sizeof(float)));

    float learningRate = initialLearningRate;
    int totalBatches = 0;
    const int gradBlockSize = 256;
    const int blockSize = 256;
    const int gridSize = (batchSize + blockSize - 1) / blockSize;

    for (int epoch = 0; epoch < maxEpochs; epoch++)
    {
        printf("[INFO] Starting epoch %d of %d...\n", epoch + 1, maxEpochs);

        // Reset MSE at the start of each epoch
        CHECK_CUDA_ERROR(cudaMemset(d_mse, 0, sizeof(float)));
        // Reset total batches for this epoch
        totalBatches = 0;

        for (int batchStart = 0; batchStart < numSamples; batchStart += batchSize)
        {
            int currentBatchSize = std::min(batchSize, numSamples - batchStart);

            // Reset gradients
            CHECK_CUDA_ERROR(cudaMemset(d_gradients, 0, numFeatures * sizeof(float)));

            // 1. Forward pass
            predictKernel<<<gridSize, blockSize>>>(d_features, d_weights, d_predictions,
                                                   numSamples, numFeatures, batchStart,
                                                   currentBatchSize);

            // 2. Compute gradients
            int sharedMemSize = (numFeatures + 1) * sizeof(float);
            computeGradientsKernel<<<numFeatures, gradBlockSize, sharedMemSize>>>(
                d_features, d_labels, d_predictions, d_gradients, d_mse,
                numSamples, numFeatures, batchStart, currentBatchSize);

            // 3. Update weights
            updateWeightsKernel<<<(numFeatures + 255) / 256, 256>>>(
                d_weights, d_gradients, learningRate, numFeatures);

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Increment total batches processed
            totalBatches++;
        }

        learningRate = initialLearningRate / pow(epoch + 1, exponent);

        // Copy MSE from device to host and print it
        float h_mse = 0.0f;
        CHECK_CUDA_ERROR(cudaMemcpy(&h_mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost));
        printf("[INFO] Completed epoch %d of %d. MSE: %.6f. Learning Rate: %.6f\n", epoch + 1, maxEpochs, h_mse / totalBatches, learningRate);
    }

    printf("[INFO] Cleaning up resources...\n");
    CHECK_CUDA_ERROR(cudaFree(d_predictions));
    CHECK_CUDA_ERROR(cudaFree(d_mse));
    printf("[INFO] Custom kernel training completed successfully.\n");
}

void CudaSGD::trainWithMemoryCoalescing()
{
    printf("[INFO] Starting training with memory coalescing...\n");
    float *d_predictions;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_predictions, batchSize * sizeof(float)));
    float *d_mse;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_mse, sizeof(float)));

    float learningRate = initialLearningRate;
    int totalBatches = 0;
    const int gradBlockSize = 256;
    const int blockSize = 256;
    const int gridSize = (batchSize + blockSize - 1) / blockSize;

    for (int epoch = 0; epoch < maxEpochs; epoch++)
    {
        printf("[INFO] Starting epoch %d of %d...\n", epoch + 1, maxEpochs);

        // Reset MSE at the start of each epoch
        CHECK_CUDA_ERROR(cudaMemset(d_mse, 0, sizeof(float)));
        // Reset total batches for this epoch
        totalBatches = 0;

        for (int batchStart = 0; batchStart < numSamples; batchStart += batchSize)
        {
            int currentBatchSize = std::min(batchSize, numSamples - batchStart);

            // Reset gradients
            CHECK_CUDA_ERROR(cudaMemset(d_gradients, 0, numFeatures * sizeof(float)));

            // Use transposed features for coalesced access
            predictCoalescedKernel<<<gridSize, blockSize, numFeatures * sizeof(float)>>>(
                d_features_transposed, d_weights, d_predictions,
                numSamples, numFeatures, batchStart, currentBatchSize);

            // Coalesced gradient computation
            int sharedMemSize = (numFeatures + 1) * sizeof(float);
            computeGradientsKernel<<<numFeatures, gradBlockSize, sharedMemSize>>>(
                d_features, d_labels, d_predictions, d_gradients, d_mse,
                numSamples, numFeatures, batchStart, currentBatchSize);

            updateWeightsKernel<<<(numFeatures + 255) / 256, 256>>>(
                d_weights, d_gradients, learningRate, numFeatures);

            totalBatches++;
        }

        // Copy MSE from device to host and print it
        float h_mse = 0.0f;
        CHECK_CUDA_ERROR(cudaMemcpy(&h_mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost));
        printf("[INFO] Completed epoch %d of %d. MSE: %.6f. Learning Rate: %.6f\n", epoch + 1, maxEpochs, h_mse / totalBatches, learningRate);

        // Update learning rate using inverse scaling
        learningRate = initialLearningRate / pow(epoch + 1, exponent);
    }

    printf("[INFO] Cleaning up resources...\n");
    CHECK_CUDA_ERROR(cudaFree(d_predictions));
    CHECK_CUDA_ERROR(cudaFree(d_mse));
    printf("[INFO] Memory coalescing training completed successfully.\n");
}

void CudaSGD::trainAsynchronous()
{
    printf("[INFO] Starting asynchronous training with Hogwild style...\n");

    // Allocate device memory for batch predictions
    float *d_predictions;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_predictions, batchSize * sizeof(float)));
    float *d_mse;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_mse, sizeof(float)));

    float learningRate = initialLearningRate;

    // Create multiple CUDA streams for asynchronous updates
    const int numStreams = 4; // Number of parallel streams
    cudaStream_t streams[numStreams];

    printf("[INFO] Creating %d CUDA streams...\n", numStreams);
    for (int i = 0; i < numStreams; i++)
    {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }
    printf("[INFO] CUDA streams created successfully.\n");

    // Set grid and block dimensions
    int totalBatches = 0;
    int blockSize = 256;
    int gridSize = (batchSize + blockSize - 1) / blockSize;
    int gradBlockSize = 256;

    for (int epoch = 0; epoch < maxEpochs; epoch++)
    {
        printf("[INFO] Starting epoch %d of %d...\n", epoch + 1, maxEpochs);

        // Reset MSE at the start of each epoch
        CHECK_CUDA_ERROR(cudaMemsetAsync(d_mse, 0, sizeof(float), streams[0]));
        // Reset total batches for this epoch
        totalBatches = 0;

        // Process batches asynchronously across multiple streams
        for (int streamIdx = 0; streamIdx < numStreams; streamIdx++)
        {
            for (int batchStart = streamIdx * batchSize;
                 batchStart < numSamples;
                 batchStart += numStreams * batchSize)
            {

                int currentBatchSize = std::min(batchSize, numSamples - batchStart);

                // 1. Forward pass: compute predictions
                predictKernel<<<gridSize, blockSize, 0, streams[streamIdx]>>>(
                    d_features, d_weights, d_predictions + streamIdx * batchSize,
                    numSamples, numFeatures, batchStart, currentBatchSize);

                // 2. Compute gradients
                float *d_stream_gradients;
                CHECK_CUDA_ERROR(cudaMalloc((void **)&d_stream_gradients, numFeatures * sizeof(float)));
                CHECK_CUDA_ERROR(cudaMemsetAsync(d_stream_gradients, 0, numFeatures * sizeof(float), streams[streamIdx]));

                int sharedMemSize = (numFeatures + 1) * sizeof(float);
                computeGradientsKernel<<<numFeatures, gradBlockSize, sharedMemSize, streams[streamIdx]>>>(
                    d_features, d_labels, d_predictions + streamIdx * batchSize, d_stream_gradients, d_mse,
                    numSamples, numFeatures, batchStart, currentBatchSize);

                // 3. Update weights (Hogwild style - no locks)
                updateWeightsKernel<<<(numFeatures + 255) / 256, 256, 0, streams[streamIdx]>>>(
                    d_weights, d_stream_gradients, learningRate, numFeatures);

                // Free stream-specific gradients
                CHECK_CUDA_ERROR(cudaFree(d_stream_gradients));

                totalBatches++;
            }
        }

        // Synchronize all streams at the end of each epoch
        for (int i = 0; i < numStreams; i++)
        {
            CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
        }

        // Update learning rate using inverse scaling
        learningRate = initialLearningRate / pow(epoch + 1, exponent);

        // Copy MSE from device to host and print it
        float h_mse = 0.0f;
        CHECK_CUDA_ERROR(cudaMemcpy(&h_mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost));
        printf("[INFO] Completed epoch %d of %d. MSE: %.6f. Learning Rate: %.6f\n", epoch + 1, maxEpochs, h_mse / totalBatches, learningRate);
    }

    // Cleanup
    printf("[INFO] Cleaning up resources...\n");
    CHECK_CUDA_ERROR(cudaFree(d_predictions));
    CHECK_CUDA_ERROR(cudaFree(d_mse));
    for (int i = 0; i < numStreams; i++)
    {
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }
    printf("[INFO] Asynchronous training completed successfully.\n");
}

void CudaSGD::train(const std::string &method)
{
    if (method == "cublas")
    {
        trainWithCuBLAS();
    }
    else if (method == "custom")
    {
        trainWithCustomKernels();
    }
    else if (method == "coalesced")
    {
        trainWithMemoryCoalescing();
    }
    else if (method == "async")
    {
        trainAsynchronous();
    }
}

void CudaSGD::benchmark()
{
    printf("[INFO] Benchmarking CudaSGD with %d samples and %d features...\n", numSamples, numFeatures);
    // warp
    std::vector<std::string> methods = {"async", "cublas", "custom", "coalesced"};

    for (const auto &method : methods)
    {
        // Reset weights
        CHECK_CUDA_ERROR(cudaMemset(d_weights, 0, numFeatures * sizeof(float)));

        auto start = std::chrono::high_resolution_clock::now();
        train(method);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;
        printf("[INFO] Training with %s completed in %.6f seconds.\n", method.c_str(), duration.count());
    }
}
