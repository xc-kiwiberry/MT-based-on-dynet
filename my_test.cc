#include "my_encdec.h"
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
  get_args(argc, argv, params, TEST);

  // Dictionary
  cerr << "Loading dictionary..." << endl;
  Dict dictIn(params.dic_src_file), dictOut(params.dic_trg_file);

  unsigned SRC_VOCAB_SIZE = dictIn.size();
  unsigned TGT_VOCAB_SIZE = dictOut.size();
  cerr << "Dictionary load success." << endl;
  cerr << "SRC_VOCAB_SIZE = " << SRC_VOCAB_SIZE << endl;
  cerr << "TGT_VOCAB_SIZE = " << TGT_VOCAB_SIZE << endl;

  // Load datasets ---------------------------------------------------------------------------------
  
  read_corpus(params.test_file, "test", dictIn, test);
  if (params.debug_info){
    read_corpus(params.test_labels_file + "0", "test_label", dictOut, test_label);
    assert(test.size() == test_label.size());
  }

  // Create model ---------------------------------------------------------------------------------
  ParameterCollection model;
  EncoderDecoder lm(model,
                    params.LAYERS,
                    params.INPUT_DIM,
                    params.HIDDEN_DIM,
                    params.ATTENTION_SIZE,
                    SRC_VOCAB_SIZE,
                    TGT_VOCAB_SIZE,
                    0.08);

  // Load model ---------------------------------------------------------------------------------
  TextFileLoader loader(params.model_file);
  loader.populate(model);
  cerr << params.model_file << " has been loaded." << endl;

  // Translate ---------------------------------------------------------------------------------
  mkdir("test", 0755);
  ostringstream trans_out_ss;
  trans_out_ss << "test//" << params.exp_name << ".out";
  ofstream fout(trans_out_ss.str());
  cerr << "translation will be saved in " << trans_out_ss.str() << endl;

  int cnt = 0, miss = 0;
  for (int i = 0; i < test.size(); i++) {
    ComputationGraph cg;
    Timer* iteration = new Timer("completed in");
    vector<unsigned> res = lm.generate(test[i], miss, cg, params.beam_size);
    cerr << ++cnt << " : ";
    delete iteration;
    if (params.debug_info){
      cerr << "src---:";
      for (int j = 0; j < test[i].size()-1 ; ++j) {
        cerr << dictIn.convert(test[i][j]) << " ";
      }
      cerr << endl;
    }
    cerr << "res---:";
    for (int j = 0; j < res.size()-1 ; ++j) {
      cerr << dictOut.convert(res[j]) << " ";
      fout << dictOut.convert(res[j]) << " ";
    }
    cerr << endl;
    fout << endl;
    if (params.debug_info){
      cerr << "std0---:";
      for (int j = 0; j < test_label[i].size()-1 ; ++j) {
        cerr << dictOut.convert(test_label[i][j]) << " ";
      }
      cerr << endl;
    }
  }
  cerr << "translation finished. " << miss << " sents can't translate in beam_size 10." << endl;

  string cmd = "perl ../multi-bleu.perl " + params.test_labels_file 
                  + " < " + trans_out_ss.str() 
                  + " > " + "test//" + params.exp_name + ".bleu_res";
  system(cmd.c_str());
  
  return 0;
}

