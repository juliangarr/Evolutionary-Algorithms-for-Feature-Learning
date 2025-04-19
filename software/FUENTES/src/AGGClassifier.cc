#include "AGGClassifier.h"

#include "random.hpp"
#include <iostream>

using namespace std;
using Random = effolkronium::random_static;

AGGClassifier::AGGClassifier(const vector<vector<DataInstance>>& datos, string type, bool blx_alpha) : GeneticClassifier(datos, type, blx_alpha) {
    // Establecer el número de padres al tamaño de la población
    numPadres = TAM_POBLACION;

    // Establecer el Nº esperado de cruces según la probabilidad de cruce y el número de padres
    numEsperadoCruces = round(PROB_CRUCE * numPadres/2);

    // Inicializar el Nº esperado de mutaciones
    numEsperadoMutaciones = PROB_MUT * TAM_POBLACION;
}

/*
1) En el Generacional, se multiplica POPSIZE*PMUT_IND = 50*0.08 = 4.
Entonces escoges aleatoriamente 4 individuos y le mutas un gen también elegido aleatoriamente.
*/
void AGGClassifier::mutacion(vector<vector<double>>& sucesores){
    int index;
    for(int i = 0; i < numEsperadoMutaciones; i++) {
        index = Random::get(0, TAM_POBLACION - 1);
        mutacionMovNormal(sucesores[index]);
    }
}

// Reemplazo total con elitismo y evaluación de la población.
void AGGClassifier::reemplazo(vector<vector<double>>& population, vector<vector<double>>& sucesores) {
    // Guardar el mejor individuo de la generación anterior
    double bestOldFitness = bestFitness;
    int bestOldIndex = bestIndividual;

    // Evaluar la población de sucesores
    evaluarPoblacion(sucesores);

    // Comparar el mejor individuo de la generación anterior con el mejor de la actual (ELITISMO)
    if(bestFitness < bestOldFitness) {
        // Encontrar el índice del peor individuo de la población de sucesores
        int worstIndividual1 = distance(fitnesses.begin(), min_element(fitnesses.begin(), fitnesses.end()));

        // Reemplazar el peor individuo de la población de sucesores por el mejor de la generación anterior
        sucesores[worstIndividual1] = population[bestOldIndex];   

        // Actualizar el mejor individuo de la generación actual con el de la anterior
        bestFitness = bestOldFitness;
        bestIndividual = worstIndividual1;      // El mejor individuo de la generación anterior ahora ocupa la posición del peor de la actual
        
        // Actualizar la fitness del mejor individuo
        fitnesses[worstIndividual1] = bestFitness;
    }
    
    // Reemplazar la población por los sucesores
        //population = sucesores;
    population = move(sucesores);
    sucesores.resize(TAM_POBLACION, vector<double>(numFeatures));
}
