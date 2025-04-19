#include "AMBestClassifier.h"

#include "random.hpp"
#include <tuple>
#include <iostream>

using namespace std;
using Random = effolkronium::random_static;

// Declarar la funcion para ordenar la población según los fitnesses
void ordenarPorFitness(vector<vector<double>>& pop, const vector<double>& fit);

AMBestClassifier::AMBestClassifier(const vector<vector<DataInstance>>& datos, string type, bool blx_alpha, int generations, double prob) : AMProbClassifier(datos, type, blx_alpha, generations, prob) {
}

tuple<vector<double>, double> AMBestClassifier::train(int fold) {
    return AM_Best(fold);
}

tuple<vector<double>, double> AMBestClassifier::AM_Best(int fold){
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

    while (evaluationsDone < MAX_EVALUATIONS) {
        // Checkear si toca hacer BL
        if(contador_generaciones == numGeneraciones){
            // Ordenar la población
            ordenarPorFitness(population, fitnesses);

            // Ordenar los fitnesses de mayor a menor
            std::sort(fitnesses.begin(), fitnesses.end(), [](double a, double b) {
                return a > b;
            });

            // Actualizar el mejor individuo
            bestFitness = fitnesses[0];
            bestIndividual = 0;

            // Seleccionar los mejores individuos de la población para aplicar BL
            for(int i = 0; i < numIndividuosBL; ++i) {
                // Aplicar BL al individuo i ya que la población está ordenada
                busquedaLocalIndividuo(population[i], i, ITERS_BL);
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

// Función para ordenar la población según los fitnesses
void ordenarPorFitness(vector<vector<double>>& pop, const vector<double>& fit) {
    // Crear un vector de pares (valor_a, valor_b)
    vector<pair<vector<double>, double>> combined;
    size_t tam = pop.size();
    for (size_t i = 0; i < tam; ++i) {
        combined.push_back(make_pair(pop[i], fit[i]));
    }

    // Ordenar el vector combinado por los valores en el segundo vector (fit)
    sort(combined.begin(), combined.end(), [](const pair<vector<double>, double>& a, const pair<vector<double>, double>& b) {
        return a.second > b.second;
    });

    // Extraer los elementos del primer vector (pop) en el nuevo orden
    vector<vector<double>> sorted_vector_a;
    for (const auto& pair : combined) {
        sorted_vector_a.push_back(pair.first);
    }

    pop = move(sorted_vector_a);
}