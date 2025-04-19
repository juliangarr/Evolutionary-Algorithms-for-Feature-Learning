#ifndef AM_BEST_CLASSIFIER_H
#define AM_BEST_CLASSIFIER_H

#include "AMProbClassifier.h"

using namespace std;

class AMBestClassifier : public AMProbClassifier {
    public:
        // Constructor
        AMBestClassifier(const vector<vector<DataInstance>>& datos, string type, bool blx_alpha = false, int generaciones = 10, double probNumIndividuos = 1.0);

        // Sobreescritura del método train (llama a AM_Prob)
        tuple<vector<double>, double> train(int fold) override;

        // Algoritmo Memético según el número de individuos sobre el que aplicar la búsqueda local
        tuple<vector<double>, double> AM_Best(int fold);
};

#endif