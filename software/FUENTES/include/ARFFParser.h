#ifndef ARFF_PARSER_H
#define ARFF_PARSER_H

#include <string>
#include <vector>

using namespace std;

// Estructura para almacenar una instancia de datos
struct DataInstance {
    std::vector<double> features;
    std::string classLabel;
};

class ARFFParser {
    private:
        vector<string> filenames;
        vector<vector<DataInstance>> data;

    public:
        ARFFParser(const vector<string>& filenames);
        bool parse();
        vector<vector<DataInstance>> getDataInSets() const;
        void normalizeData();
        void printData() const;
};

#endif
