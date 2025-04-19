#include "GeneticClassifier.h"

#include "random.hpp"
#include <tuple>
#include <iostream>

using namespace std;
using Random = effolkronium::random_static;

GeneticClassifier::GeneticClassifier(const vector<vector<DataInstance>>& datos, string type, bool blx_alpha) : RandomToolsClassifier(datos, type) {
    // Inicializar el mejor individuo
    bestIndividual = 0;
    bestFitness = 0;
    bestWeights = vector<double>(numFeatures);

    // Inicializar el número de padres a cero para que las clases hijas lo inicialicen adecuadamente
    numPadres = 0;

    // Inicializar el número esperado de cruces a cero para que las clases hijas lo inicialicen adecuadamente
    numEsperadoCruces = 0;

    // Inicializar las fitnesses
    fitnesses = vector<double>(TAM_POBLACION);

    // Inicializar el booleano de BLX
    blx = blx_alpha;
}

tuple<vector<double>, double> GeneticClassifier::train(int fold) {
    return AG(fold);
}

tuple<vector<double>, double> GeneticClassifier::AG(int fold) {
    // Restablecer el número de evaluaciones
    evaluationsDone = 0;
    
    // Generar la población inicial
    vector<vector<double>> population(TAM_POBLACION);
    for(int i = 0; i < TAM_POBLACION; i++) {
        population[i] = generaSolucionInicial();    
    }

    // Evaluar la población inicial
    evaluarPoblacion(population);

    // Crear el vector de sucesores
    vector<vector<double>> sucesores(numPadres);

    while(evaluationsDone < MAX_EVALUATIONS){
        // Generación de los padres
        generarPadres(numPadres, population, sucesores);

        // Cruce de los padres
        cruce(sucesores);

        // Mutación para generar los hijos
        mutacion(sucesores);

        // Reemplazo de la población (Y EVALUACIÓN DE LA NUEVA POBLACIÓN DENTRO DE REEMPLAZO)
        reemplazo(population, sucesores);
    }

    // Calculate the best weights
    bestWeights = population[bestIndividual];

    // Calcular real fitness (NOT IN TRAINING SET -> TEST SET)
    double realFitness = funcionObjetivo(bestWeights, fold);

    return make_tuple(bestWeights, realFitness);
}

void GeneticClassifier::evaluarPoblacion(const vector<vector<double>>& population) {
    for(int i = 0; i < TAM_POBLACION; i++) {
        fitnesses[i] = funcionObjetivoLocal(population[i]);
    }
    // Actualizar el mejor individuo de la población
    mejorIndividuo();
}

void GeneticClassifier::mejorIndividuo() {
    // Encontrar el iterador al elemento máximo en el vector
    auto maxIter = max_element(fitnesses.begin(), fitnesses.end());

    // Calcular el índice del elemento máximo y actualizar el mejor individuo
    bestIndividual = distance(fitnesses.begin(), maxIter);
    bestFitness = *maxIter;
}

vector<double> GeneticClassifier::seleccionTorneo(const vector<vector<double>>& population) const{
    double bestFit = __DBL_MIN__;
    int index, bestIndex = 0;
    double fitness;

    for(int i = 0; i < 3; i++) {
        index = Random::get(0, TAM_POBLACION - 1);
        fitness = fitnesses[index];
        if(fitness > bestFit) {
            bestFit = fitness;
            bestIndex = index;
        }
    }
    return population[bestIndex];
}

void GeneticClassifier::generarPadres(int numDad, const vector<vector<double>>& population, vector<vector<double>>& sucesores) const{
    // Hacemos torneo tantas veces como número de padres
    for(int i = 0; i < numDad; i++) {
        sucesores[i] = seleccionTorneo(population);
    }
}

void GeneticClassifier::cruce(vector<vector<double>>& sucesores) {
    // Hacemos Nº Esperado de Cruces tomando las "Nº Esperado de cruces" primeras parejas de padres
    for(int i = 0; i < numEsperadoCruces; i++) {        
        if(blx) {
            cruceBLX(sucesores[i*2], sucesores[i*2 + 1]);
        } else {
            cruceAritmetico(sucesores[i*2], sucesores[i*2 + 1]);
        }
    }
}

void GeneticClassifier::cruceBLX(vector<double>& padre1, vector<double>& padre2) {
    // Para cada gen de los padres
    for(int i = 0; i < numFeatures; i++) {
        // Buscamos el máximo y el mínimo de cada gen
        double maxGen = max(padre1[i], padre2[i]);
        double minGen = min(padre1[i], padre2[i]);

        // Calculamos el rango
        double rango = maxGen - minGen;

        // Calculamos el rango de cruce
        double rangoCruce = 0.3 * rango;

        // Calculamos los nuevos valores de los genes 
        double cruce1 = Random::get(minGen - rangoCruce, maxGen + rangoCruce);
        double cruce2 = Random::get(minGen - rangoCruce, maxGen + rangoCruce);

        // Truncamos los valores de cruce al intervalo [0, 1]
        cruce1 = max(0.0, min(1.0, cruce1));
        cruce2 = max(0.0, min(1.0, cruce2));

        // Actualizamos los padres
        padre1[i] = cruce1;
        padre2[i] = cruce2;
    }
}

void GeneticClassifier::cruceAritmetico(vector<double>& padre1, vector<double>& padre2) {
    // Calculamos el valor de alpha
    double alpha = Random::get(0.0, 1.0);

    // Calculamos los hijos actualizando los padres
    for(int i = 0; i < numFeatures; i++) {
        double aux = padre1[i];
        padre1[i] = alpha * aux + (1 - alpha) * padre2[i];
        padre2[i] = alpha * padre2[i] + (1 - alpha) * aux;
    }
}


// FUNCIONES DE MUTACIÓN Y REEMPLAZO VACÍAS PARA EVITAR ERROR DE COMPLILACIÓN
void GeneticClassifier::mutacion(vector<vector<double>>& sucesores) {
    // Implementación predeterminada (no hace nada)
}

void GeneticClassifier::reemplazo(vector<vector<double>>& population, vector<vector<double>>& sucesores) {
    // Implementación predeterminada (no hace nada)
}