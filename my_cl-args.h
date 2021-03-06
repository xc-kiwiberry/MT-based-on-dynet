#ifndef MY_CL_ARGS_H 
#define MY_CL_ARGS_H 

#include <iostream>
#include <stdlib.h>
#include <string>
#include <sstream>

enum Task {
  TRAIN, 
  TEST
};

/**
 *  Structure holding any possible command line argument
 */
struct Params {
  string exp_name = "mt";

  string train_file = "";
  string train_labels_file = "";

  string dev_file = "";
  string dev_labels_file = "";

  string test_file = "";
  string test_labels_file = "";

  string model_file = "";
  string dic_src_file = "dict_src.txt";
  string dic_trg_file = "dict_trg.txt";

  // default Hyperparameters
  unsigned LAYERS = 1;
  unsigned INPUT_DIM = 620;
  unsigned HIDDEN_DIM = 1000;
  unsigned ATTENTION_SIZE = 1000;
  unsigned BATCH_SIZE = 80;
  unsigned SRC_DIC_LIM = 30000;
  unsigned TGT_DIC_LIM = 30000;

  unsigned print_freq = 500;
  unsigned save_freq = 2000;

  bool debug_info = false;
  double learning_rate = -1;
  unsigned sent_length_limit = 50;
  unsigned beam_size = 10;

  // MRT
  bool mrt_enable = false;
  float mrt_lenRatio = 1.2;
  unsigned mrt_sampleSize = 60;
  float mrt_alpha = 0.005;

}params;

extern Params params;

/**
 * \brief Get parameters from command line arguments
 * \details Parses parameters from `argv` and check for required fields depending on the task
 * 
 * \param argc Number of arguments
 * \param argv Arguments strings
 * \param params Params structure
 * \param task Task
 */
void get_args(int argc,
              char** argv,
              Params& params,
              Task task) {
  int i = 1;
  while (i < argc) {
    string arg = argv[i];
    if (arg == "--name") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.exp_name;
      i++;
    } else if (arg == "--train" || arg == "-t") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.train_file;
      i++;
    } else if (arg == "--dev" || arg == "-d") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.dev_file;
      i++;
    } else if (arg == "--train_labels" || arg == "-tl") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.train_labels_file;
      i++;
    } else if (arg == "--dev_labels" || arg == "-dl") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.dev_labels_file;
      i++;
    } else if (arg == "--dict_src" ||  arg == "-ds") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.dic_src_file;
      i++;
    } else if (arg == "--dict_trg" ||  arg == "-dt") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.dic_trg_file;
      i++;
    } else if (arg == "--test" || arg == "-ts") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.test_file;
      i++;
    } else if (arg == "--test_labels" || arg == "-tsl") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.test_labels_file;
      i++;
    } else if (arg == "--model" || arg == "-m") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.model_file;
      i++;
    } else if (arg == "--num_layers" || arg == "-l") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.LAYERS;
      i++;
    } else if (arg == "--input_size" || arg == "-i") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.INPUT_DIM;
      i++;
    } else if (arg == "--hidden_size" || arg == "-h") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.HIDDEN_DIM;
      params.ATTENTION_SIZE = params.HIDDEN_DIM; // ATTENTION_SIZE = HIDDEN_DIM
      i++;
    } else if (arg == "--batch_size" || arg == "-b") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.BATCH_SIZE;
      i++;
    } else if (arg == "--print_freq") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.print_freq;
      i++;
    } else if (arg == "--save_freq") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.save_freq;
      i++;
    } else if (arg == "--debug_info") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.debug_info;
      i++;
    } else if (arg == "--mrt_enable") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.mrt_enable;
      i++;
    } else if (arg == "--mrt_lenRatio") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.mrt_lenRatio;
      i++;
    } else if (arg == "--mrt_sampleSize") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.mrt_sampleSize;
      i++;
    } else if (arg == "--src_dic_lim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.SRC_DIC_LIM;
      i++;
    } else if (arg == "--tgt_dic_lim") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.TGT_DIC_LIM;
      i++;
    } else if (arg == "--sent_length_limit") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.sent_length_limit;
      i++;
    } else if (arg == "--learning_rate") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.learning_rate;
      i++;
    } else if (arg == "--beam_size") {
      if (i + 1 == argc) {
        std::cerr << "No matching argument for " << arg << std::endl;
        abort();
      }
      istringstream d(argv[i + 1]);
      d >> params.beam_size;
      i++;
    } else {
      std::cerr << "No matching argument for " << arg << std::endl;
      abort();
    }
    i++;
  }

  if (task == TRAIN) {
    if (params.train_file == "" || params.dev_file == "" || params.train_labels_file == "" || params.dev_labels_file == "") {
      stringstream ss;
      ss << "Usage: " << argv[0] << " -t [train_file] -d [dev_file] -tl [train_labels_file] -dl [dev_labels_file]";
      throw invalid_argument(ss.str());
    }
  } else if (task == TEST) { 
    if (params.model_file == "" || params.test_file == "") {
      stringstream ss;
      ss << "Usage: " << argv[0] << " -m [model_file] -ts [test_file]";
      throw invalid_argument(ss.str());
    }
  }

}

#endif