#ifndef AGE_CLASSIFIER_H
#define AGE_CLASSIFIER_H

#include "GeneticClassifier.h"

using namespace std;

class AGEClassifier : public GeneticClassifier {
    protected:
        static constexpr double PROB_CRUCE = 1.0;   // Prob. de cruce

    public:
        // Constructor
        AGEClassifier(const vector<vector<DataInstance>>& datos, string type, bool blx_alpha = false);

        // Función de mutación (la prob. de mutación es 0.08 y se muta un gen por individuo)
        void mutacion(vector<vector<double>>& sucesores) override;

        // Función de reemplazo (depende del tipo de algoritmo genético)
        void reemplazo(vector<vector<double>>& population, vector<vector<double>>& sucesores) override;
};

#endif