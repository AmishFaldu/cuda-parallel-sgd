#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

int loadData(const std::string &csv_filename, float *h_features, float *h_labels, int numSamples, int numFeatures)
{
    std::cout << "Loading data from " << csv_filename << " with " << numSamples << " samples and " << numFeatures << " features." << std::endl;
    std::ifstream file(csv_filename);
    if (!file.is_open())
    {
        fprintf(stderr, "Could not open CSV file: %s\n", csv_filename.c_str());
        return -1;
    }

    std::string line;
    // Skip header
    std::getline(file, line);

    int row = 0;
    while (std::getline(file, line) && row < numSamples)
    {
        std::stringstream ss(line);
        std::string value;
        int col = 0;

        // Read each value, up to numFeatures+1 (last is label)
        while (std::getline(ss, value, ','))
        {
            if (col == numFeatures)
            {
                h_labels[row] = std::stof(value);
            }
            else
            {
                h_features[row * numFeatures + col] = std::stof(value);
            }
            ++col;
        }
        if (col != numFeatures + 1)
        {
            fprintf(stderr, "CSV row %d does not have expected %d columns\n", row, numFeatures + 1);
            return -1;
        }
        ++row;
    }
    if (row != numSamples)
    {
        fprintf(stderr, "CSV file has %d samples, expected %d\n", row, numSamples);
        return -1;
    }

    // Print the first few samples for verification
    std::cout << "First few samples loaded:" << std::endl;
    for (int i = 0; i < std::min(5, numSamples); ++i)
    {
        std::cout << "Label: " << h_labels[i] << ", Features: ";
        for (int j = 0; j < numFeatures; ++j)
        {
            std::cout << h_features[i * numFeatures + j] << (j < numFeatures - 1 ? ", " : "");
        }
        std::cout << std::endl;
    }

    // Close the file
    file.close();
    std::cout << "Data loaded successfully." << std::endl;
    return 0;
}

int loadDataTransposed(const std::string &csv_filename, float *h_features, float *h_labels, int numSamples, int numFeatures)
{
    std::cout << "Loading transposed data from " << csv_filename << " with " << numSamples << " samples and " << numFeatures << " features." << std::endl;
    std::ifstream file(csv_filename);
    if (!file.is_open())
    {
        fprintf(stderr, "Could not open CSV file: %s\n", csv_filename.c_str());
        return -1;
    }

    std::string line;
    // Skip header
    std::getline(file, line);

    int row = 0;
    while (std::getline(file, line) && row < numFeatures + 1)
    {
        std::stringstream ss(line);
        std::string value;
        int col = 0;

        while (std::getline(ss, value, ','))
        {
            if (row == numFeatures)
            {
                h_labels[col] = std::stof(value);
            }
            else
            {
                // Store transposed: feature at row, sample at col
                h_features[row * numSamples + col] = std::stof(value);
            }
            ++col;
        }

        if (col != numSamples)
        {
            fprintf(stderr, "CSV file has %d samples, expected %d\n", col, numSamples);
            return -1;
        }

        ++row;
    }
    if (row != numFeatures + 1)
    {
        fprintf(stderr, "CSV row %d does not have expected %d columns\n", row, numFeatures + 1);
        return -1;
    }

    // Print the first few samples for verification
    std::cout << "First few samples loaded:" << std::endl;
    for (int i = 0; i < std::min(5, numSamples); ++i)
    {
        std::cout << "Label: " << h_labels[i] << ", Features: ";
        for (int j = 0; j < numFeatures; ++j)
        {
            std::cout << h_features[j * numSamples + i] << " ";
        }
        std::cout << std::endl;
    }

    // Close the file
    file.close();
    std::cout << "Data loaded successfully." << std::endl;
    return 0;
}