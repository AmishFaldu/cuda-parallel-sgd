#include <load_data.h>
#include <core-impl.h>

int main(int argc, char **argv)
{
    printf("Starting CudaSGD Benchmark...\n");
    // Default parameters
    int numSamples = 8348650;
    int numFeatures = 6;
    float learningRate = 0.001f;
    int batchSize = 256;
    int maxEpochs = 10;

    // Create and initialize random data
    printf("Initializing variables...\n");
    float *h_features = new float[numSamples * numFeatures];
    float *h_labels = new float[numSamples];

    if (!h_features || !h_labels)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return -1;
    }

    // Load data from CSV file
    loadData("data/train_truncated.csv", h_features, h_labels, numSamples, numFeatures);

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
    // float mse = 0.0f;
    // for (int i = 0; i < numFeatures; i++)
    // {
    //     float diff = h_learned_weights[i] - h_true_weights[i];
    //     mse += diff * diff;
    // }
    // mse /= numFeatures;

    printf("Training complete.\n");
    // printf("Mean squared error between true and learned weights: %.8f\n", mse);

    // Clean up
    delete[] h_features;
    delete[] h_labels;
    delete[] h_learned_weights;

    return 0;
}
