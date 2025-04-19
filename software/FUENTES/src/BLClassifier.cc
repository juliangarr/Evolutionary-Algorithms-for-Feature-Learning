#include "BLClassifier.h"

#include "random.hpp"

using namespace std;
using Random = effolkronium::random_static;

BLClassifier::BLClassifier(const vector<vector<DataInstance>>& datos, string type) : RandomToolsClassifier(datos, type) {
    MAX_NEIGHBORS = numFeatures*20;
}

tuple<vector<double>, double> BLClassifier::train(int fold) {
    return busquedaLocal(fold);
}

tuple<vector<double>, double> BLClassifier::busquedaLocal(int fold) {
    // Set the number of evaluations to zero
    evaluationsDone = 0;
    
    // Initialize the neighbors counter
    int consecutiveNeighbors = 0;

    // Generate a random initial solution
    vector<double> bestWeights = generaSolucionInicial();

    // Evaluate the initial solution
    double bestFitness = funcionObjetivoLocal(bestWeights);
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
                neighborWeights[index[i]] = mutacionMovNormalBL(bestWeights[index[i]]);
                consecutiveNeighbors++;

                // Evaluate the neighbor solution
                neighborFitness = funcionObjetivoLocal(neighborWeights);

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

    // Return the best solution (weights and fitness)
    return make_tuple(bestWeights, fitness);
}

double BLClassifier::mutacionMovNormalBL(double peso_pos) const {
    // Create a normal distribution
    normal_distribution<double> normalDistribution(MEDIA, DESVIACION_ESTANDAR);

    // Modify the weight of the feature with a random value (normal distribution)
    peso_pos += Random::get(normalDistribution);

    // Truncate the value to the range [0, 1]
    if (peso_pos < 0.0) {
        return 0.0;
    }
    else if (peso_pos > 1.0) {
        return 1.0;
    }
    else {
        return peso_pos;
    }
}