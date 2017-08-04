/**
 * Train a vanilla encoder decoder lstm network with minibatching
 * to perform auto-encoding.
 *
 * This provide an example of usage of the encdec.h model
 */
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
  //kSOS = 0;
  kEOS = 1;
  SRC_VOCAB_SIZE = dictIn.size();
  TGT_VOCAB_SIZE = dictOut.size();
  cerr << "Dictionary build success." << endl;
  cerr << "SRC_VOCAB_SIZE = " << SRC_VOCAB_SIZE << endl;
  cerr << "TGT_VOCAB_SIZE = " << TGT_VOCAB_SIZE << endl;

  // Load datasets ---------------------------------------------------------------------------------
  string line;
  {
    int tlc = 0;
    int ttoks = 0;
    cerr << "Reading test data from " << params.test_file << "...\n";
    ifstream in(params.test_file);
    assert(in); 
    while (getline(in, line)) {
      ++tlc;
      auto tmp = dictIn.read_sentence(line);
      //tmp.insert(tmp.begin(),kSOS);
      tmp.push_back(kEOS);
      test.push_back(tmp);
      ttoks += test.back().size();
    }
    cerr << tlc << " lines, " << ttoks << " tokens" << endl;
  }

  {
    int tlc = 0;
    int ttoks = 0;
    cerr << "Reading test_label data from " << params.test_labels_file << "...\n";
    ifstream in(params.test_labels_file);
    assert(in); 
    while (getline(in, line)) {
      ++tlc;
      auto tmp = dictOut.read_sentence(line);
      //tmp.insert(tmp.begin(),kSOS);
      tmp.push_back(kEOS);
      test_label.push_back(tmp);
      ttoks += test_label.back().size();
    }
    cerr << tlc << " lines, " << ttoks << " tokens" << endl;
  }

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
  ifstream model_in(params.model_file);
  assert(model_in);  
  boost::archive::text_iarchive ia(model_in);
  ia >> model >> lm;
  cerr << params.model_file << " has been loaded." << endl;

  // Translate ---------------------------------------------------------------------------------
  ostringstream os;
  os << "test" 
     << "_" << params.model_file.substr(0, 6)
  	 << "_" << params.exp_name
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

  string cmd = "perl ~/projects/dynet/examples/cpp/encdec/multi-bleu.perl \
               ~/data_set/dev_test/nist06/nist06.en < " + os.str();
  system(cmd.c_str());
  return 0;
}

