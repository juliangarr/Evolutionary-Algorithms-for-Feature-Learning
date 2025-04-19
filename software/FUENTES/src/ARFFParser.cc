#include "ARFFParser.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>

using namespace std;

ARFFParser::ARFFParser(const vector<string>& filenames) : filenames(filenames) {}

bool ARFFParser::parse() {
    for (const auto& filename : filenames) {
        vector<DataInstance> dataInstances;

        //cout << "Leyendo archivo ARFF: " << filename << endl;

        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "ERROR. No se pudo abrir el archivo ARFF: " << filename << endl;
            return false;
        }

        // Análisis del archivo ARFF: Almacenar los datos en 'dataInstances'
        string line;
        bool dataSection = false;
        int countAttributes = 0; // Variable para contar los atributos encontrados

        while (getline(file, line)) {
            if (!dataSection) {
                if (line.find("@data") != string::npos) {
                    dataSection = true;
                    continue;
                }
                if (line.find("@attribute") != string::npos) {
                    // Incrementar el contador de atributos
                    countAttributes++;
                    // Posibilidad de parsear nombres de características
                    continue;
                }
            } 
            else {
                // Procesar la sección de datos
                istringstream iss(line);
                string token;
                DataInstance instance;

                // Leer características convirtiendo todas a double menos la última columna
                for (int i = 0; getline(iss, token, ','); ++i) {
                    /*
                    if (token == "?") { // Manejar valores faltantes
                        instance.push_back(0.0); // Asignar un valor predeterminado para valores faltantes
                    } else {
                    */
                    // Si es el último atributo (es no numérico), asignar directamente
                    if (i == countAttributes - 1) {
                        // La última columna es la clase de destino
                        instance.classLabel = token;
                    } 
                    else {
                        // Si no, parsear a double
                        instance.features.push_back(stod(token));
                    }
                    //}
                }

                dataInstances.push_back(instance);
            }
        }

        data.push_back(dataInstances);

        file.close();
    }

    return true;
}

void  ARFFParser::normalizeData() {
    // Find the maximum and minimum values for each feature
    vector<double> maxValues(data[0][0].features.size(), numeric_limits<double>::lowest());
    vector<double> minValues(data[0][0].features.size(), numeric_limits<double>::max());

    for (const auto& fold : data) {
        for (const auto& instance : fold) {
            const vector<double>& features = instance.features;
            for (size_t i = 0; i < features.size(); i++) {
                maxValues[i] = max(maxValues[i], features[i]);
                minValues[i] = min(minValues[i], features[i]);
            }
        }
    }

    // Normalize the data using the maximum and minimum values
    for (auto& fold : data) {
        for (auto& instance : fold) {
            vector<double>& features = instance.features;
            for (size_t i = 0; i < features.size(); i++) {
                if(maxValues[i] != minValues[i]){
                    features[i] = (features[i] - minValues[i]) / (maxValues[i] - minValues[i]);
                }
                else{
                    cout << "\n\nEl feature " << i << " tiene igual mínimo que máximo en " << filenames[0] << endl << endl;
                }
            }
        }
    }
}

vector<vector<DataInstance>> ARFFParser::getDataInSets() const {
    return data;
}

void ARFFParser::printData() const {
    int count = 0;
    cout << "\nPrinting data..." << endl << endl;
    for (const auto& fold : data) {
        cout << "\n-----------------------  Fold -> " << count << " -----------------------" << endl;
        for (const auto& instance : fold) {
            for (const auto& feature : instance.features) {
                cout << feature << ",";
            }
            cout << instance.classLabel << endl;
        }
        count++;
    }
}