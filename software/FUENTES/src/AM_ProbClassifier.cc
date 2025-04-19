#include "AMProbClassifier.h"

#include "random.hpp"
#include <tuple>
#include <iostream>

using namespace std;
using Random = effolkronium::random_static;

AMProbClassifier::AMProbClassifier(const vector<vector<DataInstance>>& datos, string type, bool blx_alpha, int generations, double prob) : AGGClassifier(datos, type, blx_alpha) {
    // Establecer cada cuántas generaciones se va a aplicar el algoritmo de mutación
    numGeneraciones = generations;

    // Establecer la probabilidad de mutación
    probNumIndividuos = prob;

    // Establecer el número máximo de vecinos a explorar en la BL
    MAX_NEIGHBORS = numFeatures*20;

    // Establecer el número de iteraciones de la BL
    ITERS_BL = 2*numFeatures;
}

tuple<vector<double>, double> AMProbClassifier::train(int fold) {
    return AM_Prob(fold);
}

tuple<vector<double>, double> AMProbClassifier::AM_Prob(int fold){
    // Restablecer el número de evaluaciones
    evaluationsDone = 0;
    
    // Generar la población inicial
    vector<vector<double>> population(TAM_POBLACION);
    for(int i = 0; i < TAM_POBLACION; ++i) {
        population[i] = generaSolucionInicial();
    }

    // Evaluar la población inicial
    evaluarPoblacion(population);

    // Crear el vector de sucesores
    vector<vector<double>> sucesores(numPadres);

    // Inicializar el contador de generaciones
    int contador_generaciones = 0;

    // Crear el número de individuos para la BL
    int numIndividuosBL = round(probNumIndividuos * TAM_POBLACION);

    // Crear el vector de índices a los individuos de la población
    vector<int> indices_poblacion(TAM_POBLACION);
    for(int i = 0; i < TAM_POBLACION; ++i) indices_poblacion[i] = i;

    // Crear un índice auxiliar
    int ind;

    while (evaluationsDone < MAX_EVALUATIONS) {
        // Checkear si toca hacer BL
        if(contador_generaciones == numGeneraciones){
            // Seleccionar aleatoriamente los individuos a los que se les va a aplicar BL reordenando índices 
            Random::shuffle(indices_poblacion);
            for(int i = 0; i < numIndividuosBL; ++i) {
                ind = indices_poblacion[i];
                // Aplicar BL al individuo 
                busquedaLocalIndividuo(population[ind], ind, ITERS_BL);
            }

            // Reestablecer el contador de generaciones
            contador_generaciones = 0;
        }
        else{   // Hacer AGG
            // Generación de los padres
            generarPadres(numPadres, population, sucesores);

            // Cruce de los padres
            cruce(sucesores);

            // Mutación para generar los hijos
            mutacion(sucesores);

            // Reemplazo de la población (Y EVALUACIÓN DE LA NUEVA POBLACIÓN DENTRO DE REEMPLAZO)
            reemplazo(population, sucesores);

            // Incrementar el contador de generaciones
            ++contador_generaciones;
        }
    }
    // Establecer el mejor individuo
    bestWeights = population[bestIndividual];

    // Calcular real fitness (NOT IN TRAINING SET -> TEST SET)
    double realFitness = funcionObjetivo(bestWeights, fold);

    // Devolver el mejor individuo
    return make_tuple(bestWeights, realFitness);
}

void AMProbClassifier::busquedaLocalIndividuo(vector<double>& individuoBL, int index_individuo, int numIteraciones){
    // Initialize counters
    int consecutiveNeighbors = 0;
    int maxConsecutiveNeighbors = MAX_NEIGHBORS;
    int evaluationsBL = 0;
    int maxEvalsGlobal = MAX_EVALUATIONS;

    // Initialize and evaluate the initial solution
    double bestFitBL = fitnesses[index_individuo];
    
    // Generate the index vector
    vector<int> index(numFeatures);
    for(int i = 0; i < numFeatures; ++i) index[i] = i;

    // Create the vector of neighbors
    vector<double> neighborWeights;
    double neighborFitness;

    // Loop of local search
    while ( evaluationsDone < maxEvalsGlobal && consecutiveNeighbors < maxConsecutiveNeighbors && evaluationsBL < numIteraciones) {
        // Generate a random permutation of index about the features
        Random::shuffle(index);

        // Loop of neighborhood generation
        for (int i = 0; i < numFeatures; ++i){
            if (consecutiveNeighbors < maxConsecutiveNeighbors && evaluationsDone < maxEvalsGlobal && evaluationsBL < numIteraciones) {
                // Generate a neighbor solution
                    //neighborWeights = generaVecino(bestWeights, index[i]);
                neighborWeights = individuoBL;
                neighborWeights[index[i]] = mutacionMovNormalBL(individuoBL[index[i]]);
                ++consecutiveNeighbors;

                // Evaluate the neighbor solution
                neighborFitness = funcionObjetivoLocal(neighborWeights);
                ++evaluationsBL;

                // Update the best solution in the BL
                if (neighborFitness > bestFitBL) {
                    individuoBL = neighborWeights;
                    bestFitBL = neighborFitness;
                    consecutiveNeighbors = 0;
                    break;
                }
            }
        }
    }

    // Update the best solution in the population
    if (bestFitBL > bestFitness) {
        bestFitness = bestFitBL;
        bestIndividual = index_individuo;
    }
}

double AMProbClassifier::mutacionMovNormalBL(double peso_pos) const {
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