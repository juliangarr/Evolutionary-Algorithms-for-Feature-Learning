#include "GreedyClassifier.h"
#include <cmath>
#include <limits>
#include <iostream>

using namespace std;

GreedyClassifier::GreedyClassifier(const vector<vector<DataInstance>>& datos, string type) : Classifier(datos, type) {}

tuple<vector<double>, double> GreedyClassifier::train(int fold) {
    return greedy(fold);
}

tuple<vector<double>, double> GreedyClassifier::greedy(int fold) {
    // Inicializar W a ceros
    vector<double> weights = vector<double>(numFeatures, 0.0);  

    // Update W
    for (size_t i = 0; i < trainingSet.size(); i++) {
        double minEnemyDistance = numeric_limits<double>::max();
        double minFriendDistance = numeric_limits<double>::max();
        DataInstance ee, ea;

        // Find the closest enemy and friend
        for (size_t j = 0; j < trainingSet.size(); j++) {
            if (i != j) {          
                double distance = euclideanDistance(trainingSet[i].features, trainingSet[j].features);
                // Friend case
                if (trainingSet[i].classLabel == trainingSet[j].classLabel) {
                    if (distance < minFriendDistance) {
                        minFriendDistance = distance;
                        ea = trainingSet[j];
                    }
                }
                // Enemy case
                else {
                    if (distance < minEnemyDistance) {
                        minEnemyDistance = distance;
                        ee = trainingSet[j];
                    }
                }
            }
        }

        // Check if there are no enemies or friends
        if (ee.features.empty() || ea.features.empty()) {
            continue;   // Ignorar esta iteración y pasar a la siguiente
        }

        // Update Weights
        for (size_t j = 0; j < weights.size(); j++) {
            weights[j] = weights[j] + abs(trainingSet[i].features[j] - ee.features[j]) - abs(trainingSet[i].features[j] - ea.features[j]);
        }
    }

    // Max of W: wm
    double wm = numeric_limits<double>::min();
    for (size_t i = 0; i < weights.size(); i++) {
        if (weights[i] > wm) {
            wm = weights[i];
        }
    }

    // Normalize Weights
    for (size_t i = 0; i < weights.size(); i++) {
        if (weights[i] < 0) {
            weights[i] = 0.0;
        } 
        else {
            weights[i] /= wm;
        }
    }

    // Calculate fitness
    double fitness = funcionObjetivo(weights, fold); 

    return make_tuple(weights, fitness);
}