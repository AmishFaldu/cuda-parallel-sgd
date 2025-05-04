__global__ void computeGradientsWarpKernel(const float *features, const float *labels,
                                           const float *predictions, float *gradients,
                                           int numSamples, int numFeatures, int batchOffset,
                                           int batchSize)
{
    extern __shared__ float sharedGradients[];

    int featureIdx = blockIdx.x;
    int localThreadIdx = threadIdx.x;
    int warpSize = 32;
    int warpIdx = localThreadIdx / warpSize;
    int laneIdx = localThreadIdx % warpSize;
    int numWarps = blockDim.x / warpSize;

    // Initialize shared memory
    if (localThreadIdx < numWarps)
    {
        sharedGradients[localThreadIdx] = 0.0f;
    }
    __syncthreads();

    // Compute partial gradients
    float gradientSum = 0.0f;

    for (int i = localThreadIdx; i < batchSize; i += blockDim.x)
    {
        int sampleIdx = batchOffset + i;
        if (sampleIdx < numSamples)
        {
            float error = predictions[i] - labels[sampleIdx];
            gradientSum += error * features[sampleIdx * numFeatures + featureIdx];
        }
    }

    // Warp-level reduction using shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        gradientSum += __shfl_down_sync(0xffffffff, gradientSum, offset);
    }

    // First thread in each warp writes to shared memory
    if (laneIdx == 0)
    {
        sharedGradients[warpIdx] = gradientSum;
    }

    __syncthreads();

    // Final reduction across warps (only in the first warp)
    if (warpIdx == 0)
    {
        gradientSum = (laneIdx < numWarps) ? sharedGradients[laneIdx] : 0.0f;

        // Another warp-level reduction for final sum
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            gradientSum += __shfl_down_sync(0xffffffff, gradientSum, offset);
        }

        // First thread writes final result
        if (laneIdx == 0)
        {
            atomicAdd(&gradients[featureIdx], gradientSum / batchSize);
        }
    }
}

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
                                       const float *predictions, float *gradients,
                                       int numSamples, int numFeatures, int batchOffset,
                                       int batchSize)
{
    extern __shared__ float sharedGradients[];

    int featureIdx = blockIdx.x;
    int localThreadIdx = threadIdx.x;

    // Initialize shared memory
    if (localThreadIdx < numFeatures)
    {
        sharedGradients[localThreadIdx] = 0.0f;
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
            // atomicAdd(&sharedGradients[featureIdx], gradient);
            if (featureIdx < numFeatures)
            {
                atomicAdd(&sharedGradients[featureIdx], gradient);
            }
        }
    }

    __syncthreads();

    // Reduce shared memory results to global memory
    if (localThreadIdx == 0)
    {
        atomicAdd(&gradients[featureIdx], sharedGradients[featureIdx] / batchSize);
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
