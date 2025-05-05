#include <load_data.h>
#include <core-impl.h>

int main(int argc, char **argv)
{
    printf("Starting CudaSGD Benchmark...\n");
    // Default parameters
    int numSamples = 8348650;
    int numFeatures = 6;
    float initialLearningRate = 0.01f;
    float exponent = 0.25f;
    int batchSize = 256;
    int maxEpochs = 18;

    // Create and initialize random data
    printf("Initializing variables...\n");
    float *h_features = new float[numSamples * numFeatures];
    float *h_features_transposed = new float[numFeatures * numSamples];
    float *h_labels = new float[numSamples];

    if (!h_features || !h_labels || !h_features_transposed)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return -1;
    }

    // Load data from CSV file
    loadData("data/train_truncated.csv", h_features, h_labels, numSamples, numFeatures);
    loadDataTransposed("data/train_truncated_transposed.csv", h_features_transposed, h_labels, numSamples, numFeatures);

    // Create SGD instance
    printf("Creating CudaSGD instance...\n");
    CudaSGD sgd(numSamples, numFeatures, initialLearningRate, batchSize, maxEpochs, exponent);

    // Load data
    printf("Loading data into CudaSGD...\n");
    sgd.loadData(h_features, h_features_transposed, h_labels);

    // Benchmark different implementations
    printf("Benchmarking different implementations...\n");
    sgd.benchmark();

    // Train using the optimized method
    // printf("Training with CudaSGD...\n");
    // sgd.train("coalesced");

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

    printf("Training complete.\n");

    // Clean up
    delete[] h_features;
    delete[] h_labels;
    delete[] h_learned_weights;

    return 0;
}
