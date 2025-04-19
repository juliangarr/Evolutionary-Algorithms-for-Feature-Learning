#ifndef GENETIC_CLASSIFIER_H
#define GENETIC_CLASSIFIER_H

#include "RandomToolsClassifier.h"

using namespace std;

class GeneticClassifier : public RandomToolsClassifier{
    protected:
        static constexpr int TAM_POBLACION = 50;
        static constexpr double PROB_MUT = 0.08;     // Prob. de mutación por individuo y también de genes por individuo
        // ELIMINADA PARA AGILIZAR. static constexpr int TAM_TORNEO = 3;
        // ELIMINADA PARA AGILIZAR. static constexpr double ALPHA_BLX = 0.3;    // Alpha para cruce BLX-Alpha

        int numPadres;
        int numEsperadoCruces;
        bool blx;

        vector<double> bestWeights;
        double bestFitness;
        int bestIndividual;

        vector<double> fitnesses;

    public:
        // Constructor
        GeneticClassifier(const vector<vector<DataInstance>>& datos, string type, bool blx_alpha = false);

        // Sobreescritura del método train (llama a AG)
        tuple<vector<double>, double> train(int fold) override;

        // Algoritmo genético generacional
        tuple<vector<double>, double> AG(int fold);

        // Función de evaluación inicial de la población
        void evaluarPoblacion(const vector<vector<double>>& population);

        // Función para encontrar y actualizar el mejor individuo de la población
        void mejorIndividuo();

        // Función de selección de padres
            //vector<vector<double>> generarPadres(int numPadres, const vector<vector<double>>& population);
        void generarPadres(int numDad, const vector<vector<double>>& population, vector<vector<double>>& sucesores) const;

        // Selección por torneo (devuelve el mejor de 3 individuos aleatorios de la población actual) 
        vector<double> seleccionTorneo(const vector<vector<double>>& population) const;

        // Función de cruce
        void cruce(vector<vector<double>>& sucesores);

        // Cruce BLX-Alpha
        void cruceBLX(vector<double>& padre1, vector<double>& padre2);

        // Cruce Aritmético
        void cruceAritmetico(vector<double>& padre1, vector<double>& padre2);

        // Función de mutación (depende solo de la probabilidad de mutación)
        virtual void mutacion(vector<vector<double>>& sucesores);

        // Función de reemplazo (depende del tipo de algoritmo genético)
        virtual void reemplazo(vector<vector<double>>& population, vector<vector<double>>& sucesores);
};

#endif
