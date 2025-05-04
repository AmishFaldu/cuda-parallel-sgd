#include <core-impl.h>

int main(int argc, char **argv)
{
    printf("Starting CudaSGD Benchmark...\n");
    // Default parameters
    int numSamples = 1000000;
    int numFeatures = 10;
    float learningRate = 0.01f;
    int batchSize = 256;
    int maxEpochs = 10;

    // Create and initialize random data
    printf("Initializing variables...\n");
    float *h_features = new float[numSamples * numFeatures];
    float *h_labels = new float[numSamples];
    float *h_true_weights = new float[numFeatures];

    if (!h_features || !h_labels || !h_true_weights)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return -1;
    }

    std::mt19937 gen(42);
    std::normal_distribution<float> feat_dist(0.0f, 1.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.1f);

    // Generate true weights
    printf("Generating true weights...\n");
    for (int j = 0; j < numFeatures; j++)
    {
        h_true_weights[j] = feat_dist(gen);
    }
    // Print true weights for reference
    printf("True weights: ");
    for (int j = 0; j < numFeatures; j++)
    {
        printf("%.6f ", h_true_weights[j]);
    }
    printf("\n");

    // Generate synthetic data
    printf("Generating synthetic features and labels...\n");
    for (int i = 0; i < numSamples; i++)
    {
        float label = 0.0f;
        for (int j = 0; j < numFeatures; j++)
        {
            h_features[i * numFeatures + j] = feat_dist(gen);
            label += h_features[i * numFeatures + j] * h_true_weights[j];
        }
        h_labels[i] = label + noise_dist(gen);
    }

    // Create SGD instance
    printf("Creating CudaSGD instance...\n");
    CudaSGD sgd(numSamples, numFeatures, learningRate, batchSize, maxEpochs);

    // Load data
    printf("Loading data into CudaSGD...\n");
    sgd.loadData(h_features, h_labels);

    // Benchmark different implementations
    // printf("Benchmarking different implementations...\n");
    // sgd.benchmark();

    // Train using the optimized method
    printf("\nTraining with optimized method...\n");
    sgd.train("coalesced");

    // Get and evaluate the learned weights
    printf("Evaluating learned weights...\n");
    float *h_learned_weights = new float[numFeatures];
    sgd.getWeights(h_learned_weights);
    printf("Learned weights: ");
    for (int j = 0; j < numFeatures; j++)
    {
        printf("%.6f ", h_learned_weights[j]);
    }
    printf("\n");

    // Compute mean squared error
    float mse = 0.0f;
    for (int i = 0; i < numFeatures; i++)
    {
        float diff = h_learned_weights[i] - h_true_weights[i];
        mse += diff * diff;
    }
    mse /= numFeatures;

    printf("Training complete.\n");
    printf("Mean squared error between true and learned weights: %.8f\n", mse);

    // Clean up
    delete[] h_features;
    delete[] h_labels;
    delete[] h_true_weights;
    delete[] h_learned_weights;

    return 0;
}
