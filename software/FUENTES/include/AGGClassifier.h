#ifndef AGG_CLASSIFIER_H
#define AGG_CLASSIFIER_H

#include "GeneticClassifier.h"

using namespace std;

class AGGClassifier : public GeneticClassifier {
    protected:
        static constexpr double PROB_CRUCE = 0.68;   // Prob. de cruce
        int numEsperadoMutaciones;

    public:
        // Constructor
        AGGClassifier(const vector<vector<DataInstance>>& datos, string type, bool blx_alpha = false);

        // Función de mutación (depende de la probabilidad de mutación y del tamaño de la población)
        void mutacion(vector<vector<double>>& sucesores) override;

        // Función de reemplazo (depende del tipo de algoritmo genético)
        void reemplazo(vector<vector<double>>& population, vector<vector<double>>& sucesores) override;
};

#endif