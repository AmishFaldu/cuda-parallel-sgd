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

__global__ void transposeKernel(const float *input, float *output, int rows, int cols)
{
    // Shared memory for the block
    __shared__ float tile[16][16];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Read data in a coalesced manner
    if (x < cols && y < rows)
    {
        // tile[threadIdx.y * blockDim.x + threadIdx.x] = input[y * cols + x];
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // Calculate transposed coordinates
    int out_x = blockIdx.y * blockDim.y + threadIdx.x;
    int out_y = blockIdx.x * blockDim.x + threadIdx.y;

    // Write data in a coalesced manner
    if (out_x < rows && out_y < cols)
    {
        output[out_y * rows + out_x] = tile[threadIdx.x][threadIdx.y];
        // output[out_y * rows + out_x] = tile[threadIdx.y * blockDim.x + threadIdx.x];
        // output[out_y * rows + out_x] = tile[threadIdx.x * blockDim.y + threadIdx.y];
    }
}

void transposeMatrix(const float *d_input, float *d_output, int rows, int cols)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    // dim3 gridDim(min((rows + blockDim.y - 1) / blockDim.y, 65535), min((cols + blockDim.x - 1) / blockDim.x, 65535));

    printf("Launching transpose with grid size: %d x %d and block size: %d x %d\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // size_t sharedMemSize = blockDim.x * blockDim.y * sizeof(float);

    transposeKernel<<<gridDim, blockDim>>>(d_input, d_output, rows, cols);
    // transposeKernel<<<gridDim, blockDim, sharedMemSize, stream>>>(d_input, d_output, rows, cols);

    // Synchronize the stream to ensure the kernel has completed
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}

// Class definition for our CUDA SGD implementation
class CudaSGD
{
private:
    // Data dimensions and model parameters
    int numSamples;
    int numFeatures;
    float learningRate;
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
    CudaSGD(int numSamples, int numFeatures, float learningRate, int batchSize, int maxEpochs);
    ~CudaSGD();

    void loadData(const float *h_features, const float *h_labels);
    void getWeights(float *h_weights_out);

    // Training methods with different optimization strategies
    void trainWithCuBLAS();
    void trainWithCustomKernels();
    void trainWithWarpOptimization();
    void trainWithMemoryCoalescing();
    void trainAsynchronous();

    void train(const std::string &method = "optimized");
    void benchmark();
};

CudaSGD::CudaSGD(int numSamples, int numFeatures, float learningRate,
                 int batchSize, int maxEpochs)
    : numSamples(numSamples), numFeatures(numFeatures),
      learningRate(learningRate), batchSize(batchSize),
      maxEpochs(maxEpochs)
{
    printf("Initializing CudaSGD with %d samples and %d features...\n", numSamples, numFeatures);
    // Allocate device memory for weights and gradients
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_weights, numFeatures * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_gradients, numFeatures * sizeof(float)));

    // Create cuBLAS handle
    printf("[INFO] Creating cuBLAS handle...\n");
    CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));

    // Initialize weights to zero
    std::vector<float> initialWeights(numFeatures, 0.5f);
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, initialWeights.data(), numFeatures * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize host weights buffer
    h_weights = new float[numFeatures];

    printf("[INFO] CudaSGD initialized with %d samples and %d features\n", numSamples, numFeatures);
    printf("[INFO] Learning rate: %.6f, Batch size: %d, Max epochs: %d\n",
           learningRate, batchSize, maxEpochs);
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

void CudaSGD::loadData(const float *h_features, const float *h_labels)
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
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels,
                                numSamples * sizeof(float),
                                cudaMemcpyHostToDevice));

    printf("[INFO] Data loaded to device successfully.\n");

    // Transpose features for coalesced memory access
    printf("[INFO] Transposing feature matrix...\n");
    transposeMatrix(d_features, d_features_transposed, numSamples, numFeatures);
    printf("[INFO] Feature matrix transposed successfully.\n");
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
    // int weightSize = inputSize;

    // Allocate device memory
    float *d_predictions;
    float *d_errors;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_predictions, batchSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_errors, batchSize * sizeof(float)));

    // Constants for cuBLAS operations
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
    }

    // Cleanup
    printf("[INFO] Cleaning up resources...\n");
    CHECK_CUDA_ERROR(cudaFree(d_predictions));
    CHECK_CUDA_ERROR(cudaFree(d_errors));
    printf("[INFO] cuBLAS training completed successfully.\n");
}

void CudaSGD::trainWithCustomKernels()
{
    printf("[INFO] Starting training with custom kernels...\n");
    float *d_predictions;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_predictions, batchSize * sizeof(float)));

    const int blockSize = 256;
    const int gridSize = (batchSize + blockSize - 1) / blockSize;

    for (int epoch = 0; epoch < maxEpochs; epoch++)
    {
        printf("[INFO] Starting epoch %d of %d...\n", epoch + 1, maxEpochs);
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
            int gradBlockSize = 256;
            size_t sharedMemSize = numFeatures * sizeof(float);
            computeGradientsKernel<<<numFeatures, gradBlockSize, sharedMemSize>>>(
                d_features, d_labels, d_predictions, d_gradients,
                numSamples, numFeatures, batchStart, currentBatchSize);

            // 3. Update weights
            updateWeightsKernel<<<(numFeatures + 255) / 256, 256>>>(
                d_weights, d_gradients, learningRate, numFeatures);
        }
    }

    printf("[INFO] Cleaning up resources...\n");
    CHECK_CUDA_ERROR(cudaFree(d_predictions));
    printf("[INFO] Custom kernel training completed successfully.\n");
}

void CudaSGD::trainWithWarpOptimization()
{
    printf("[INFO] Starting training with warp optimization...\n");
    float *d_predictions;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_predictions, batchSize * sizeof(float)));

    const int blockSize = 256;
    const int gridSize = (batchSize + blockSize - 1) / blockSize;

    for (int epoch = 0; epoch < maxEpochs; epoch++)
    {
        printf("[INFO] Starting epoch %d of %d...\n", epoch + 1, maxEpochs);
        for (int batchStart = 0; batchStart < numSamples; batchStart += batchSize)
        {
            int currentBatchSize = std::min(batchSize, numSamples - batchStart);

            CHECK_CUDA_ERROR(cudaMemset(d_gradients, 0, numFeatures * sizeof(float)));

            // Warp-optimized prediction
            predictKernel<<<gridSize, blockSize>>>(d_features, d_weights, d_predictions,
                                                   numSamples, numFeatures, batchStart,
                                                   currentBatchSize);

            // Warp-level gradient computation
            int gradBlockSize = 256;
            size_t sharedMemSize = (gradBlockSize / 32) * sizeof(float);
            computeGradientsWarpKernel<<<numFeatures, gradBlockSize, sharedMemSize>>>(
                d_features, d_labels, d_predictions, d_gradients,
                numSamples, numFeatures, batchStart, currentBatchSize);

            updateWeightsKernel<<<(numFeatures + 255) / 256, 256>>>(
                d_weights, d_gradients, learningRate, numFeatures);
        }
        printf("[INFO] Completed epoch %d of %d.\n", epoch + 1, maxEpochs);
    }

    printf("[INFO] Cleaning up resources...\n");
    CHECK_CUDA_ERROR(cudaFree(d_predictions));
    printf("[INFO] Warp optimization training completed successfully.\n");
}

void CudaSGD::trainWithMemoryCoalescing()
{
    printf("[INFO] Starting training with memory coalescing...\n");
    float *d_predictions;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_predictions, batchSize * sizeof(float)));

    const int blockSize = 256;
    const int gridSize = (batchSize + blockSize - 1) / blockSize;

    for (int epoch = 0; epoch < maxEpochs; epoch++)
    {
        printf("[INFO] Starting epoch %d of %d...\n", epoch + 1, maxEpochs);
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
            int gradBlockSize = 256;
            size_t sharedMemSize = numFeatures * sizeof(float);
            computeGradientsKernel<<<numFeatures, gradBlockSize, sharedMemSize>>>(
                d_features, d_labels, d_predictions, d_gradients,
                numSamples, numFeatures, batchStart, currentBatchSize);

            updateWeightsKernel<<<(numFeatures + 255) / 256, 256>>>(
                d_weights, d_gradients, learningRate, numFeatures);
        }
    }

    printf("[INFO] Cleaning up resources...\n");
    CHECK_CUDA_ERROR(cudaFree(d_predictions));
    printf("[INFO] Memory coalescing training completed successfully.\n");
}

void CudaSGD::trainAsynchronous()
{
    printf("[INFO] Starting asynchronous training with Hogwild style...\n");

    // Allocate device memory for batch predictions
    float *d_predictions;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_predictions, batchSize * sizeof(float)));

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
    int blockSize = 256;
    int gridSize = (batchSize + blockSize - 1) / blockSize;
    int gradBlockSize = 256;

    for (int epoch = 0; epoch < maxEpochs; epoch++)
    {
        printf("[INFO] Starting epoch %d of %d...\n", epoch + 1, maxEpochs);
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

                int sharedMemSize = numFeatures * sizeof(float);
                computeGradientsKernel<<<numFeatures, gradBlockSize, sharedMemSize, streams[streamIdx]>>>(
                    d_features, d_labels, d_predictions + streamIdx * batchSize, d_stream_gradients,
                    numSamples, numFeatures, batchStart, currentBatchSize);

                // 3. Update weights (Hogwild style - no locks)
                updateWeightsKernel<<<(numFeatures + 255) / 256, 256, 0, streams[streamIdx]>>>(
                    d_weights, d_stream_gradients, learningRate, numFeatures);

                // Free stream-specific gradients
                CHECK_CUDA_ERROR(cudaFree(d_stream_gradients));
            }
        }

        // Synchronize all streams at the end of each epoch
        for (int i = 0; i < numStreams; i++)
        {
            CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
        }
        printf("[INFO] Completed epoch %d of %d.\n", epoch + 1, maxEpochs);
    }

    // Cleanup
    printf("[INFO] Cleaning up resources...\n");
    CHECK_CUDA_ERROR(cudaFree(d_predictions));
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
    else if (method == "warp")
    {
        trainWithWarpOptimization();
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
    // coalesced
    std::vector<std::string> methods = {"async", "cublas", "custom", "warp", "coalesced"};

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
