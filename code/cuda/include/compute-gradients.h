// Memory-coalesced prediction kernel that uses shared memory
__global__ void predictCoalescedKernel(const float *features, const float *weights, float *predictions,
                                       int numSamples, int numFeatures, int batchOffset, int batchSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float sharedWeights[];

    // Collaboratively load weights into shared memory
    for (int i = threadIdx.x; i < numFeatures; i += blockDim.x)
    {
        sharedWeights[i] = weights[i];
    }

    __syncthreads();

    if (tid < batchSize)
    {
        int sampleIdx = batchOffset + tid;
        if (sampleIdx < numSamples)
        {
            float pred = 0.0f;

            // Coalesced memory access pattern
            for (int j = 0; j < numFeatures; j++)
            {
                pred += features[j * numSamples + sampleIdx] * sharedWeights[j];
            }

            predictions[tid] = pred;
        }
    }
}

// Gradient computation with shared memory
__global__ void computeGradientsKernel(const float *features, const float *labels,
                                       const float *predictions, float *gradients, float *mse,
                                       int numSamples, int numFeatures, int batchOffset,
                                       int batchSize)
{
    extern __shared__ float sharedMemory[];

    float *sharedGradients = sharedMemory;                   // Shared memory for gradients
    float *sharedErrors = &sharedMemory[numFeatures];        // Shared memory for squared errors

    int featureIdx = blockIdx.x;
    int localThreadIdx = threadIdx.x;

    // Initialize shared memory
    if (localThreadIdx < numFeatures && featureIdx < numFeatures)
    {
        sharedGradients[featureIdx] = 0.0f; // Initialize gradients for this feature
    }

    if (localThreadIdx == 0 && featureIdx == 0)
    {
        sharedErrors[0] = 0.0f;    // Only one thread initializes the error accumulator
    }
    __syncthreads();

    // Each thread computes partial gradient for one feature across samples
    for (int i = localThreadIdx; i < batchSize; i += blockDim.x)
    {
        int sampleIdx = batchOffset + i;
        if (sampleIdx < numSamples)
        {
            float error = predictions[i] - labels[sampleIdx];
            float gradient = error * features[sampleIdx * numFeatures + featureIdx];
            atomicAdd(&sharedGradients[featureIdx], gradient);
            // Accumulate squared error
            atomicAdd(&sharedErrors[0], error * error);
        }
    }

    __syncthreads();

    // Reduce shared memory results to global memory
    if (localThreadIdx == 0)
    {
        atomicAdd(&gradients[featureIdx], sharedGradients[featureIdx] / batchSize);

        // Compute and store the MSE
        if (featureIdx == 0) // Only one thread writes the MSE
        {
            atomicAdd(mse, sharedErrors[0] / batchSize);
        }
    }
}

// Prediction kernel
__global__ void predictKernel(const float *features, const float *weights, float *predictions,
                              int numSamples, int numFeatures, int batchOffset, int batchSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < batchSize)
    {
        int sampleIdx = batchOffset + tid;
        if (sampleIdx < numSamples)
        {
            float pred = 0.0f;
            for (int j = 0; j < numFeatures; j++)
            {
                pred += features[sampleIdx * numFeatures + j] * weights[j];
            }
            predictions[tid] = pred;
        }
    }
}

__global__ void updateWeightsKernel(float *weights, const float *gradients,
                                    float learningRate, int numFeatures)
{
    // Grid-stride loop for efficient memory access
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numFeatures;
         idx += gridDim.x * blockDim.x)
    {
        // Atomic update to prevent race conditions
        atomicAdd(&weights[idx], -learningRate * gradients[idx]);
    }
}
