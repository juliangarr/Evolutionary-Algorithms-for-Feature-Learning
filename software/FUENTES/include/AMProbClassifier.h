#ifndef AM_PROB_CLASSIFIER_H
#define AM_PROB_CLASSIFIER_H

#include "AGGClassifier.h"

using namespace std;

class AMProbClassifier : public AGGClassifier {
    protected:
        int numGeneraciones;
        double probNumIndividuos;
        int MAX_NEIGHBORS;
        int ITERS_BL;

    public:
        // Constructor
        AMProbClassifier(const vector<vector<DataInstance>>& datos, string type, bool blx_alpha = false, int generaciones = 10, double probNumIndividuos = 1.0);

        // Sobreescritura del método train (llama a AM_Prob)
        tuple<vector<double>, double> train(int fold) override;

        // Algoritmo Memético según el número de individuos sobre el que aplicar la búsqueda local
        tuple<vector<double>, double> AM_Prob(int fold);

        // Búsqueda Local sobre un individuo
        void busquedaLocalIndividuo(vector<double>& individuoBL, int index_individuo, int numIteraciones);

        // Mutación de un gen (específico de BL)
        double mutacionMovNormalBL(double peso_pos) const;
};

#endif