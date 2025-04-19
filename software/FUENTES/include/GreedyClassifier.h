#ifndef GREEDY_CLASSIFIER_H
#define GREEDY_CLASSIFIER_H

#include "Classifier.h"

#include <tuple>

using namespace std;

class GreedyClassifier : public Classifier{
    public:
        // Constructor
        GreedyClassifier(const vector<vector<DataInstance>>& datos, string type);
        
        // Sobreescritura del método train (llama a Greedy)
        tuple<vector<double>, double> train(int fold) override;

        // Algoritmo Greedy
        tuple<vector<double>, double> greedy(int fold);
};

#endif
