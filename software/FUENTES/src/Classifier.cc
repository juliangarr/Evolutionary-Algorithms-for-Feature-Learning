#include "Classifier.h"

#include <cmath>
#include <chrono>
#include <numeric>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

double mean(const vector<double>& valores) {
    double suma = accumulate(valores.begin(), valores.end(), 0.0);

    if (!valores.empty()) {
        return suma / valores.size();
    } 
    else {
        cerr << "ERROR: No se puede calcular la media de un VECTOR VACÍO" << endl;
        return -1.0;
    }
}

double euclideanDistance(const vector<double>& instance1, const vector<double>& instance2) {
    double distance = 0.0;
    for (size_t i = 0; i < instance1.size(); ++i) {
        distance += (instance1[i] - instance2[i])*(instance1[i] - instance2[i]);
    }
    return sqrt(distance);
}

double weightedEuclideanDistance(const vector<double>& instance1, const vector<double>& instance2, vector<double> weights) {
    if( weights.size() != instance1.size() || instance1.size() != instance2.size() ) {
        cout << "ERROR: Tamaño de pesos distinto al de instancias en weightedEuclideanDistance" << endl;
        return -1.0;
    }

    double distance = 0.0;
    for (size_t i = 0; i < instance1.size(); ++i) {
        if(weights[i] >= 0.1){
            distance += weights[i] * (instance1[i] - instance2[i])*(instance1[i] - instance2[i]);
        }
    }
    return sqrt(distance);
}

Classifier::Classifier(const vector<vector<DataInstance>>& datos, string type) : data(datos) {
    // Setteamos el nombre del clasificador
    tipo = type;

    k_Folds = data.size();
    numFeatures = data[0][0].features.size();
    featureWeights = vector<double>(numFeatures, 1.0);  // Default, every weight is 1.0 (1NN Classifier)

    tasa_class = vector<double>(k_Folds, -1.0);
    tasa_red = vector<double>(k_Folds, -1.0);
    fitness = vector<double>(k_Folds, -1.0);
    tiempo = vector<double>(k_Folds, -1.0);
    trainedWeights = vector<vector<double>>(k_Folds, vector<double>(numFeatures, -1.0));

    trainingSet = vector<DataInstance>();
    testSet = vector<DataInstance>();
}

Classifier::Classifier(const vector<vector<DataInstance>>& datos, const vector<double>& pesos, string type) : data(datos), featureWeights(pesos) {
    // Setteamos el nombre del clasificador
    tipo = type;

    k_Folds = data.size();
    numFeatures = data[0][0].features.size();

    tasa_class = vector<double>(k_Folds, -1.0);
    tasa_red = vector<double>(k_Folds, -1.0);
    fitness = vector<double>(k_Folds, -1.0);
    tiempo = vector<double>(k_Folds, -1.0);
    trainedWeights = vector<vector<double>>(k_Folds, vector<double>(numFeatures, -1.0));

    trainingSet = vector<DataInstance>();
    testSet = vector<DataInstance>();
}

void Classifier::kFoldCrossValidation() { 
    // For each fold
    for (int fold = 0; fold < k_Folds; ++fold) {
        // Initialize time
        auto start = chrono::high_resolution_clock::now();

        // Clear training and test sets
        testSet.clear();
        trainingSet.clear();

        // Asign test set
        testSet = data[fold];

        // Asign training set
        for (int i = 0; i < k_Folds; ++i) {
            if (i != fold) {
                trainingSet.insert(trainingSet.end(), data[i].begin(), data[i].end());
            }
        }
        
        // Train the classifier
        tuple<vector<double>, double> trained = train(fold);
        trainedWeights[fold] = get<0>(trained);
        featureWeights = trainedWeights[fold];

        // Update fitness
        fitness[fold] = get<1>(trained);
        
        // Update final time
        auto end = chrono::high_resolution_clock::now();
        tiempo[fold] = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    }

    // Print results
    printResults();

    // Print results to CSV
    resultsToCSV();
}

tuple<vector<double>, double> Classifier::train(int fold) {
    vector<double> weights(numFeatures, 1.0);
    double fitness = funcionObjetivo(weights, fold);
    return make_tuple(weights, fitness);
}

string Classifier::classify(const DataInstance& instance, const vector<double>& weights) const {
    // Find the Nearest Neighbor
    double minDistance = numeric_limits<double>::max();
    string nearestNeighborClass;
    for (const auto& trainingInstance : trainingSet) {
        double distance = weightedEuclideanDistance(instance.features, trainingInstance.features, weights);
        if (distance < minDistance) {
            minDistance = distance;
            nearestNeighborClass = trainingInstance.classLabel;
        }
    }
    return nearestNeighborClass;
}

double Classifier::calculateClassRate(const vector<double>& weights) const {
    double numCorrect = 0.0;
    int numTestInstances = testSet.size();
    for (int i = 0; i < numTestInstances; ++i) {
        if (classify(testSet[i], weights) == testSet[i].classLabel) {
            numCorrect++;
        }
    }
    return ( (100.0 * numCorrect) / numTestInstances );
}

double Classifier::calculateReductionRate(const vector<double>& weights) const{
    if((int)weights.size() != numFeatures){
        cout << "ERROR: Tamaño de pesos distinto al de instancias en calculateReductionRate" << endl;
        return -1.0;
    }
    double numReductions = 0.0;
    for (int i = 0; i < numFeatures; i++) {
        if(weights[i] < 0.1) 
            numReductions++;
    }
    return (100.0 * numReductions)/numFeatures;
}

double Classifier::funcionObjetivo(const vector<double>& weights, int fold) {
    // Calculate and update classification rate
    tasa_class[fold] = calculateClassRate(weights);
    
    // Calculate and update reduction rate
    tasa_red[fold] = calculateReductionRate(weights);

    return (ALPHA * tasa_class[fold] + (1.0 - ALPHA) * tasa_red[fold]);
}

void Classifier::resultsToCSV() const {
    // Crear la ruta completa del archivo
    // string filepath = "../../BIN/results/results_" + tipo + ".csv";
    string filepath = "results/results_" + tipo + ".csv";

    ofstream outputFile(filepath);

    // Verificar si el archivo se abrió correctamente
    if (!outputFile.is_open()) {
        cout << "Error al abrir el archivo " << filepath << endl;
        return;
    }

    // Escribir encabezado
    outputFile << "--------- RESULTADOS FINALES ---------   " + tipo + "   " << endl;
    outputFile << endl;
    outputFile << "Partición,%_class,%_red,Fit.,T" << endl;

    // Escribir datos para cada fold
    for (int fold = 0; fold < k_Folds; ++fold) {
        outputFile << fold + 1 << ","; // Número de partición

        // Escribir datos
        outputFile << fixed << setprecision(2) << tasa_class[fold] << "," << tasa_red[fold] << ","  << fitness[fold] << "," << scientific << tiempo[fold];

        // Nueva línea para el siguiente fold
        outputFile << endl;
    }

    // Escribir fila promedio
    outputFile << "Media,";
    outputFile << fixed << setprecision(2) << mean(tasa_class) << ","  << mean(tasa_red) << ","  << mean(fitness) << "," << scientific << mean(tiempo);

    outputFile.close();
}

void Classifier::printResults() const {
    // Create header
    string header = "\n\n<<<<<<<   " + tipo + "   >>>>>>>>";

    // Print header
    cout << header << endl;

    // Print Partial Results
    cout << "--------- RESULTADOS PARCIALES ---------\n" << endl;
    cout << "Partición\tTrain\tTest\tReducción\tFitness\tTiempo" << endl;
    cout << "\t\t[en %]\t[en %]\t[en %]\t\t\t[en ms]" << endl;

    for (int fold = 0; fold < k_Folds; ++fold) {
        cout << "\t" << fold + 1 << "\t - \t" << fixed << setprecision(2) << tasa_class[fold] << "\t" << tasa_red[fold] << "\t\t" << fitness[fold] << "\t" << scientific << tiempo[fold] << endl;
    }
    // Restaurar el formato de salida a fixed
    cout << fixed;

    // Calculate Final Results
    double tasa_c = mean(tasa_class);
    double tasa_r = mean(tasa_red);
    double fit = mean(fitness);
    double time = mean(tiempo);

    // Print Final Results
    cout << "\n--------- RESULTADOS FINALES ---------" << endl;
    cout << "- Tasa Clasificación: \t" << tasa_c << " %" << endl;
    cout << "- Tasa Reducción: \t" << tasa_r << " %" << endl;
    cout << "- Fitness: \t\t" << fit << endl;
    cout << "- Tiempo: \t\t" << scientific << time << " ms" << endl;

    // Restaurar el formato de salida a fixed
    cout << fixed;
    
    // Print trained weights
    cout << "\n- Pesos finales: ";
    for (size_t i = 0; i < trainedWeights.size(); i++) {
        cout << "\n\tPartición " << i + 1 << ": ";
        for (size_t j = 0; j < trainedWeights[i].size(); j++) {
            cout << trainedWeights[i][j] << " ";
        }
    }

    cout << endl;
}