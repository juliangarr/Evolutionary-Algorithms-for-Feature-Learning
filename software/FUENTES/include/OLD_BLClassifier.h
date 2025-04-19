#ifndef OLD_BL_CLASSIFIER_H
#define OLD_BL_CLASSIFIER_H

#include "Classifier.h"

#include <tuple>

using namespace std;

class OLD_BLClassifier : public Classifier{
    protected:
        static constexpr int MAX_EVALUATIONS = 15000;
        static constexpr double MEDIA = 0.0;
        static constexpr double VARIANZA = 0.3;

        int MAX_NEIGHBORS;

    public:
        OLD_BLClassifier(const vector<vector<DataInstance>>& datos, string type);

        tuple<vector<double>, double> train(int fold) override;

        tuple<vector<double>, double> busquedaLocal(int fold);

        vector<double> generaSolucionInicial() const;

        double funcionObjetivoBL(const vector<double>& weights) const;

        double calculateClassRateInTrain(const vector<double>& weights) const;

        double generaVecino(double peso_pos) const;
        //vector<double> generaVecino(const vector<double>& weights, int index) const;
};

#endif
