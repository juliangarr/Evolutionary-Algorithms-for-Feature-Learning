#include "AGEClassifier.h"

#include "random.hpp"

using namespace std;
using Random = effolkronium::random_static;

AGEClassifier::AGEClassifier(const vector<vector<DataInstance>>& datos, string type, bool blx_alpha) : GeneticClassifier(datos, type, blx_alpha) {
    // Inicializar el número de padres a 2
    numPadres = 2;

    // Establecer el número esperado de cruces según la probabilidad de cruce y el número de padres
    numEsperadoCruces = round(PROB_CRUCE * numPadres/2);
}

void AGEClassifier::mutacion(vector<vector<double>>& sucesores){
    double muta;
    for(int i = 0; i < numPadres; i++) {
        muta = Random::get(0.0, 1.0);
        if(muta < PROB_MUT) {
            mutacionMovNormal(sucesores[i]);
        }
    }
}

void AGEClassifier::reemplazo(vector<vector<double>>& population, vector<vector<double>>& sucesores) {
    // Evaluar sucesores
    double fitness_h1 = funcionObjetivoLocal(sucesores[0]);
    double fitness_h2 = funcionObjetivoLocal(sucesores[1]);

    // Encontrar los dos peores individuos de la población
    int worstIndividual1 = distance(fitnesses.begin(), min_element(fitnesses.begin(), fitnesses.end()));
    int worstIndividual2 = -1;
    double worstFitness2 = 101.00; // Inicialización con un valor suficientemente grande
    for (int i = 0; i < TAM_POBLACION; ++i) {
        if (i != worstIndividual1 && fitnesses[i] < worstFitness2) {
            worstIndividual2 = i;
        }
    }

    // Comparamos los sucesores con los peores individuos de la población y los reemplazamos si es necesario
    // Caso 1: El peor hijo es mejor que el mejor individuo a sustituir -> SUSTITUIMOS AMBOS
    if (min(fitness_h1, fitness_h2) > fitnesses[worstIndividual2]) {
        population[worstIndividual1] = sucesores[0];
        fitnesses[worstIndividual1] = fitness_h1;

        population[worstIndividual2] = sucesores[1];
        fitnesses[worstIndividual2] = fitness_h2;

        // Actualizamos el mejor individuo si es necesario
        if (max(fitness_h1, fitness_h2) > bestFitness) {
            if (fitness_h1 > fitness_h2) {
                bestFitness = fitness_h1;
                bestIndividual = worstIndividual1;
            }
            else {
                bestFitness = fitness_h2;
                bestIndividual = worstIndividual2;
            }
        }
    }
    // Caso 2: UN PEOR SOBREVIVE -> El mejor hijo es mejor que el peor individuo a sustituir -> SUSTITUIMOS AL PEOR DE LA POBLACIÓN POR EL MEJOR HIJO
    else if (max(fitness_h1, fitness_h2) > fitnesses[worstIndividual1]) {
        if (fitness_h1 > fitness_h2) {
            population[worstIndividual1] = sucesores[0];
            fitnesses[worstIndividual1] = fitness_h1;

            // Actualizamos el mejor individuo si es necesario
            if (fitness_h1 > bestFitness) {
                bestFitness = fitness_h1;
                bestIndividual = worstIndividual1;
            }
        }
        else {
            population[worstIndividual1] = sucesores[1];
            fitnesses[worstIndividual1] = fitness_h2;

            // Actualizamos el mejor individuo si es necesario
            if (fitness_h2 > bestFitness) {
                bestFitness = fitness_h2;
                bestIndividual = worstIndividual1;
            }
        }
    }
}