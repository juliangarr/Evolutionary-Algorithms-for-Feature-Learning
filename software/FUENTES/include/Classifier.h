#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "ARFFParser.h"

#include <vector>
#include <string>

using namespace std;

// Devuelve la media de un vector
double mean(const vector<double>& valores);

// Calcula la distancia euclídea entre dos vectores
double euclideanDistance(const vector<double>& instance1, const vector<double>& instance2);

// Calcula la distancia euclídea ponderada entre dos vectores
double weightedEuclideanDistance(const vector<double>& instance1, const vector<double>& instance2, const vector<double> weights);

class Classifier {
    protected:
        vector<vector<DataInstance>> data;
        vector<double> featureWeights;
        static constexpr double ALPHA = 0.75;
        int k_Folds;
        int numFeatures;
        
        vector <double> tasa_class;
        vector <double> tasa_red;
        vector <double> fitness;
        vector <double> tiempo;
        vector <vector<double>> trainedWeights;

        vector<DataInstance> trainingSet;
        vector<DataInstance> testSet;

        string tipo;

    public:
        // Constructor
        Classifier(const vector<vector<DataInstance>>& datos, string type);

        // Constructor con pesos
        Classifier(const vector<vector<DataInstance>>& datos, const vector<double>& pesos, string type);

        // K-fold Cross Validation
        void kFoldCrossValidation();

        // Entrenamiento -> Modificar pesos W
        virtual tuple<vector<double>, double> train(int fold);

        // Clasificación de una instancia
        string classify(const DataInstance& instance, const vector<double>& weights) const;

        // Tasa de clasificación
        double calculateClassRate(const vector<double>& weights) const;

        // Tasa de reducción
        double calculateReductionRate(const vector<double>& weights) const;

        // Función objetivo
        double funcionObjetivo(const vector<double>& weights, int fold);

        // Imprimir resultados
        void printResults() const;

        // Imprimir resultados en fichero csv
        void resultsToCSV() const;
};

#endif