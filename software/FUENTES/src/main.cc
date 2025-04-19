#include "ARFFParser.h"
#include "GreedyClassifier.h"
#include "BLClassifier.h"
#include "AGGClassifier.h"
#include "AGEClassifier.h"
#include "AMProbClassifier.h"
#include "OLD_BLClassifier.h"
#include "AMBestClassifier.h"
// Incluir otros modelos aquí
#include "Classifier.h"
#include "GeneticClassifier.h"
#include "RandomToolsClassifier.h"

#include "random.hpp"

#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm>    // Para transform
#include <cctype>       // Para toupper
#include <sstream>
#include <unistd.h>     // Para getcwd()

using namespace std;
using Random = effolkronium::random_static;

#define ONE_NN "-1nn"
#define GREEDY "-gr"
#define BUSQUEDA_LOCAL "-bl"
#define AGG "-agg"
#define AGG_BLX "-agg_blx"
#define AGE "-age"
#define AGE_BLX "-age_blx"
#define AM_ALL "-am_all"
#define AM_RAND "-am_rand"
#define AM_BEST "-am_best"
#define OLD_BUSQUEDA_LOCAL "-bl_old"

// Función para obtener las rutas completas de los archivos ARFF
vector<string> getArffFilenames(const string& datasetFilename) {
    vector<std::string> arffFilenames;
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        string basePath(cwd); // Ruta del directorio de trabajo actual
        basePath = basePath + "/../BIN/datasets_arff";
        for (int i = 1; i < 6; i++) {
            ostringstream oss;
            oss << basePath << "/" << datasetFilename << "_" << i << ".arff";
            arffFilenames.push_back(oss.str());
        }
    } else {
        cerr << "ERROR: No se pudo obtener el directorio de trabajo actual." << endl;
    }
    return arffFilenames;
}

int main(int argc, char* argv[]) {
    
    // Verificar nº de argumentos
    if (argc < 3 || argc > 4) {
        cerr << "ERROR: Nº de argumentos NO VALIDO. Uso correcto: " << argv[0] << " <nombre_dataset> <modelo> <semilla (opcional)>" << endl;
        return 1;
    }

    // Leer nombre del dataset
    string datasetFilename = argv[1];

    // Leer el tipo de modelo
    string modelType = argv[2];

    // Leer la semilla
    long int seed;
    if (argc == 4) {
        seed = stol(argv[3]);
        Random::seed(seed);
    }
    else {
        Random::seed(12345);
    }

    // Determinar los ARFF correspondientes
    vector<string> arffFilenames;

    if(datasetFilename != "breast-cancer" && datasetFilename != "ecoli" && datasetFilename != "parkinsons") {
        cerr << "ERROR: Dataset no reconocido. Los datasets disponibles son: breast-cancer, ecoli, parkinsons." << endl;
        return 1;
    } 
    else {
        arffFilenames = getArffFilenames(datasetFilename);
    }

    // Leer los datos de los ficheros
    ARFFParser parser(arffFilenames);
    if (!parser.parse()) {
        cerr << "Error al leer los ARFF.\n";
        return 1;
    }

    // Normalizar los datos
    parser.normalizeData();

    // Imprimir los datos normalizados (opcional)
    //parser.printData();
    
    // Crear tipo de modelo (string)
    string mayus_dataset = datasetFilename;
    transform(mayus_dataset.begin(), mayus_dataset.end(), mayus_dataset.begin(), ::toupper);
    mayus_dataset += "_";
    
    // Llamar al modelo correspondiente e imprimir cabecera
    if (modelType == ONE_NN) {
        mayus_dataset += "1_NN";
        Classifier one_classifier(parser.getDataInSets(), mayus_dataset);
        one_classifier.kFoldCrossValidation();
    }
    else if (modelType == GREEDY) {
        mayus_dataset += "GREEDY";
        GreedyClassifier greedy_classifier(parser.getDataInSets(), mayus_dataset);
        greedy_classifier.kFoldCrossValidation();
    }
    else if (modelType == BUSQUEDA_LOCAL) {
        mayus_dataset += "BL";
        BLClassifier bl_classifier(parser.getDataInSets(), mayus_dataset);
        bl_classifier.kFoldCrossValidation();
    }
    else if (modelType == AGG) {
        mayus_dataset += "AGG";
        AGGClassifier agg_classifier(parser.getDataInSets(), mayus_dataset);
        agg_classifier.kFoldCrossValidation();
    }
    else if (modelType == AGG_BLX) {
        mayus_dataset += "AGG_BLX";
        AGGClassifier agg_classifier_blx(parser.getDataInSets(), mayus_dataset, true);
        agg_classifier_blx.kFoldCrossValidation();
    }
    else if (modelType == AGE) {
        mayus_dataset += "AGE";
        AGEClassifier age_classifier(parser.getDataInSets(), mayus_dataset);
        age_classifier.kFoldCrossValidation();
    }
    else if (modelType == AGE_BLX) {
        mayus_dataset += "AGE_BLX";
        AGEClassifier age_classifier_blx(parser.getDataInSets(), mayus_dataset, true);
        age_classifier_blx.kFoldCrossValidation();
    }
    else if (modelType == AM_ALL) {
        mayus_dataset += "AM_ALL";
        AMProbClassifier am_all_classifier(parser.getDataInSets(), mayus_dataset, false, 10);
        am_all_classifier.kFoldCrossValidation();
    }
    else if (modelType == AM_RAND) {
        mayus_dataset += "AM_RAND";
        AMProbClassifier am_rand_classifier(parser.getDataInSets(), mayus_dataset, false, 10, 0.1);
        am_rand_classifier.kFoldCrossValidation();
    }
    else if (modelType == AM_BEST) {
        mayus_dataset += "AM_BEST";
        AMBestClassifier am_best_classifier(parser.getDataInSets(), mayus_dataset, false, 10, 0.1);
        am_best_classifier.kFoldCrossValidation();
    }
    // PRUEBA
    else if (modelType == OLD_BUSQUEDA_LOCAL) {
        mayus_dataset += "OLD_BL";
        OLD_BLClassifier old_bl_classifier(parser.getDataInSets(), mayus_dataset);
        old_bl_classifier.kFoldCrossValidation();
    }
    //--------------------------------
    else {
        cerr << "ERROR: Modelo no reconocido. Los modelos disponibles son: -1nn, -gr, -bl, -agg, -agg_blx, -age, -age_blx, -am_all, -am_rand, -am_best" << endl;
        return 1;
    }

    return 0;
}
