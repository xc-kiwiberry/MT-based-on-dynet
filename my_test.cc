
#include "my_encdec.h"
#include "my_cl-args.h"
#include "my_dict.h"
#include "stdlib.h"

using namespace std;
using namespace dynet;

// Datasets
  vector<vector<int>> test, test_label;

int main(int argc, char** argv) {
  // Fetch dynet params ----------------------------------------------------------------------------
  auto dyparams = dynet::extract_dynet_params(argc, argv);
  dynet::initialize(dyparams);

  // Fetch program specific parameters (see ../utils/cl-args.h) ------------------------------------
  Params params;
  get_args(argc, argv, params, TEST);

  // Dictionary
  cerr << "Building dictionary..." << endl;
  XC::Dict dictIn(params.train_file), dictOut(params.train_labels_file);
  kEOS = 0; 
  SRC_VOCAB_SIZE = dictIn.size();
  TGT_VOCAB_SIZE = dictOut.size();
  cerr << "Dictionary build success." << endl;
  cerr << "SRC_VOCAB_SIZE = " << SRC_VOCAB_SIZE << endl;
  cerr << "TGT_VOCAB_SIZE = " << TGT_VOCAB_SIZE << endl;

  // Load datasets ---------------------------------------------------------------------------------
  
  read_corpus(params.test_file, "test", dictIn, test);
  read_corpus(params.test_labels_file + "0", "test_label", dictOut, test_label);

  assert(test.size() == test_label.size());

  // Create model ---------------------------------------------------------------------------------
  Model model;
  EncoderDecoder<GRUBuilder> lm(model,
                                 params.LAYERS,
                                 params.INPUT_DIM,
                                 params.HIDDEN_DIM,
                                 params.ATTENTION_SIZE);
  cerr << "create model success" << endl;

  // Load model ---------------------------------------------------------------------------------
  TextFileLoader loader(params.model_file);
  loader.populate(model);
  cerr << params.model_file << " has been loaded." << endl;

  // Translate ---------------------------------------------------------------------------------
  ostringstream os;
  os << "test" 
     //<< "_" << params.model_file.substr(0, 6)
  	 << "_" << params.exp_name
     //<< "_" << params.test_file.substr(0,6)
     << "_" << params.LAYERS 
     << "_" << params.INPUT_DIM
     << "_" << params.HIDDEN_DIM
     << ".out";
  ofstream fout(os.str());
  cerr << "translation will be saved in " << os.str() << endl;

  int cnt = 0, miss = 0;
  for (int i = 0; i < test.size(); i++) {
    ComputationGraph cg;
    Timer* iteration = new Timer("completed in");
    vector<unsigned> res = lm.generate(test[i], miss, cg);
    cerr << ++cnt << " : ";
    delete iteration;
    cerr << "src---:";
    for (int j = 0; j < test[i].size()-1 ; ++j) {
      cerr << dictIn.convert(test[i][j]) << " ";
    }
    cerr << endl;
    cerr << "res---:";
    for (int j = 0; j < res.size()-1 ; ++j) {
      cerr << dictOut.convert(res[j]) << " ";
      fout << dictOut.convert(res[j]) << " ";
    }
    cerr << endl;
    fout << endl;
    cerr << "std---:";
    for (int j = 0; j < test_label[i].size()-1 ; ++j) {
      cerr << dictOut.convert(test_label[i][j]) << " ";
    }
    cerr << endl;
  }
  cerr << "translation finished. " << miss << " sents can't translate in beam_size 10." << endl;

  string cmd = "perl multi-bleu.perl " +
                params.test_labels_file + " < " + os.str() + " > " +  params.exp_name + ".bleu_res";
  system(cmd.c_str());
  
  return 0;
}

