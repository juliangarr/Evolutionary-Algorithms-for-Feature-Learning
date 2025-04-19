#include "OLD_BLClassifier.h"

#include "random.hpp"

using namespace std;
using Random = effolkronium::random_static;

OLD_BLClassifier::OLD_BLClassifier(const vector<vector<DataInstance>>& datos, string type) : Classifier(datos, type) {
    MAX_NEIGHBORS = numFeatures*20;
}

vector<double> OLD_BLClassifier::generaSolucionInicial() const{
    vector<double> weights(numFeatures);
    
    for (int i = 0; i < numFeatures; i++) {
        weights[i] = Random::get<double>(0.0, 1.0); // Random::get<double>(min, max) usa ya una distribución uniforme
    }

    return weights;
}

// Classification rate with LEAVE-ONE-OUT
double OLD_BLClassifier::calculateClassRateInTrain(const vector<double>& weights) const{
    double numCorrects = 0.0;
    int numTrainingInstances = trainingSet.size();

    for (int i = 0; i < numTrainingInstances; ++i) {
        double minDistance = numeric_limits<double>::max();
        string nearestNeighborClass;
        for (int j = 0; j < numTrainingInstances; ++j) {
            if (i != j) {
                double distance = weightedEuclideanDistance(trainingSet[i].features, trainingSet[j].features, weights);
                if (distance < minDistance && distance > 0.0) {
                    minDistance = distance;
                    nearestNeighborClass = trainingSet[j].classLabel;
                }
            }
        }
        if (nearestNeighborClass == trainingSet[i].classLabel) {
            numCorrects++;
        }
    }
    return ( (100.0 * numCorrects) / numTrainingInstances );
}

double OLD_BLClassifier::funcionObjetivoBL(const vector<double>& weights) const {
    return ( ALPHA * calculateClassRateInTrain(weights) + (1.0 - ALPHA) * calculateReductionRate(weights) );
}

tuple<vector<double>, double> OLD_BLClassifier::train(int fold) {
    return busquedaLocal(fold);
}

tuple<vector<double>, double> OLD_BLClassifier::busquedaLocal(int fold) {
    // Initialize the evaluation and neighbors counters
    int evaluationsDone = 0;
    int consecutiveNeighbors = 0;

    // Generate a random initial solution
    vector<double> bestWeights = generaSolucionInicial();

    // Evaluate the initial solution
    double bestFitness = funcionObjetivoBL(bestWeights);
    evaluationsDone++;

    vector<int> index(numFeatures);
    for(int i = 0; i < numFeatures; i++) index[i] = i;

    vector<double> neighborWeights;
    double neighborFitness;

    // Loop of local search
    while ( evaluationsDone < MAX_EVALUATIONS && consecutiveNeighbors < MAX_NEIGHBORS ) {
        // Generate a random permutation of index about the features
        Random::shuffle(index);

        // Loop of neighborhood generation
        for (int i = 0; i < numFeatures; i++){
            if (consecutiveNeighbors < MAX_NEIGHBORS && evaluationsDone < MAX_EVALUATIONS ) {
                // Generate a neighbor solution
                    //neighborWeights = generaVecino(bestWeights, index[i]);
                neighborWeights = bestWeights;
                neighborWeights[index[i]] = generaVecino(bestWeights[index[i]]);
                consecutiveNeighbors++;

                // Evaluate the neighbor solution
                neighborFitness = funcionObjetivoBL(neighborWeights);
                evaluationsDone++;

                // Update the best solution
                if (neighborFitness > bestFitness) {
                    bestWeights = neighborWeights;
                    bestFitness = neighborFitness;
                    consecutiveNeighbors = 0;
                    break;
                }
            }
        }
    }

    // Calculate fitness
    double fitness = funcionObjetivo(bestWeights, fold); 

    return make_tuple(bestWeights, fitness);
}

double OLD_BLClassifier::generaVecino(double peso_pos) const {
    // Create a normal distribution
    normal_distribution<double> distribution(MEDIA, sqrt(VARIANZA));

    // Modify the weight of the feature with a random value (normal distribution)
    peso_pos += Random::get(distribution);

    // Truncate the value to the range [0, 1]
    if (peso_pos < 0.0) {
        peso_pos = 0.0;
    }
    else if (peso_pos > 1.0) {
        peso_pos = 1.0;
    }

    return peso_pos;
}

// vector<double> BLClassifier::generaVecino(const vector<double>& weights, int index) const {
//     vector<double> newWeights = weights;

//     // Create a normal distribution
//     normal_distribution<double> distribution(MEDIA, sqrt(VARIANZA));

//     // Modify the weight of the feature with a random value (normal distribution)
//     newWeights[index] += Random::get(distribution);

//     // Truncate the value to the range [0, 1]
//     if (newWeights[index] < 0.0) {
//         newWeights[index] = 0.0;
//     }
//     else if (newWeights[index] > 1.0) {
//         newWeights[index] = 1.0;
//     }

//     return newWeights;
// }