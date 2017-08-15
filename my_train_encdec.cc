
#include "my_encdec.h"
#include "my_cl-args.h"
#include "my_dict.h"

using namespace std;
using namespace dynet;

typedef pair<vector<int>, vector<int>> pp;

// Sort sentences in descending order of length
bool comp(const pp& aa, const pp& bb) {
  return aa.first.size() > bb.first.size();
}

int fCountSize(const vector<vector<int>>& lines){
  int cnt = 0;
  for (const auto & line : lines) cnt += line.size();
  return cnt;
}

void fGiveMask(const vector<vector<int>>& lines, vector<vector<float>>& mask) {
  for (int i = 0; i < lines.size(); i++) {
    mask.push_back(vector<float>());
    assert(lines[i].size() >= 2);
    mask[i].push_back(1.);
    for (int j = 1; j < lines[i].size(); j++) {
      if (lines[i][j-1] != kEOS) mask[i].push_back(1.);
      else mask[i].push_back(0.);
    }
  }
}

// Datasets
  vector<pp> train_data;
  vector<vector<int>> training, training_label;
  vector<vector<float>> train_mask, train_label_mask;
  vector<vector<int>> dev;

void debug(const vector<float>& v) {
  for (auto aa: v) cerr << aa << " ";
    cerr <<endl;
} 

int main(int argc, char** argv) {
  // Fetch dynet params ----------------------------------------------------------------------------
  auto dyparams = dynet::extract_dynet_params(argc, argv);
  dynet::initialize(dyparams);

  // debug
  /*
  ComputationGraph g;
  vector<float> v = {6,7,8,9};
  Expression x = input(g, Dim({1}, 4), v);
  debug(as_vector(x.value()));
  Expression y = pick_batch_elems(x, {2,2,3,1});
  debug(as_vector(y.value()));
  x=x*2;
  debug(as_vector(x.value()));
  y=y*2;
  debug(as_vector(y.value()));
  return 0;//*/

  // Fetch program specific parameters (see ../utils/cl-args.h) ------------------------------------
  Params params;
  get_args(argc, argv, params, TRAIN);
  // Load datasets ---------------------------------------------------------------------------------

  // Dictionary
  cerr << "Building dictionary..." << endl;
  XC::Dict dictIn(params.train_file), dictOut(params.train_labels_file);
  kEOS = 0;
  SRC_VOCAB_SIZE = dictIn.size();
  TGT_VOCAB_SIZE = dictOut.size();
  cerr << "Dictionary build success." << endl;
  cerr << "SRC_VOCAB_SIZE = " << SRC_VOCAB_SIZE << endl;
  cerr << "TGT_VOCAB_SIZE = " << TGT_VOCAB_SIZE << endl;

  // Read training data
  read_corpus(params.train_file, "training", dictIn, training);
  read_corpus(params.train_labels_file, "training_label", dictOut, training_label);

  assert(training.size() == training_label.size());

  for (int i = 0; i < training.size(); i++) {
  	  if (training[i].size() > 50 || training_label[i].size() > 50) continue;
	  train_data.push_back(pp(training[i], training_label[i]));
  }

  // Sort the training sentences in descending order of length (for minibatching)
  sort(train_data.begin(), train_data.end(), comp);
  training.resize(train_data.size());
  training_label.resize(train_data.size());
  for (int i = 0; i < train_data.size(); i++) {
	  training[i] = train_data[i].first;
	  training_label[i] = train_data[i].second;
  }

  // Pad the sentences in the same batch with EOS so they are the same length
  // This modifies the training objective a bit by making it necessary to
  // predict EOS multiple times, but it's easy and not so harmful
  for (int i = 0; i < training.size(); i += params.BATCH_SIZE){
    for (int j = 1; j < params.BATCH_SIZE && i + j < training.size(); ++j){
      while (training[i + j].size() < training[i].size())
        training[i + j].push_back(kEOS);
	  }
  }
  for (int i = 0; i < training_label.size(); i += params.BATCH_SIZE) {
	  size_t maxlen = training_label[i].size();
	  for (int j = 1; j < params.BATCH_SIZE && i + j < training_label.size(); j++) {
		  maxlen = max(maxlen, training_label[i + j].size());
	  }
	  for (int j = 0; j < params.BATCH_SIZE && i + j < training_label.size(); j++) {
		  while (training_label[i + j].size() < maxlen)
			  training_label[i + j].push_back(kEOS);
	  }
  }
  
  // Read validation dataset
  read_corpus(params.dev_file, "dev", dictIn, dev);

  fGiveMask(training, train_mask);              
  fGiveMask(training_label, train_label_mask); 
  int countSize = fCountSize(training) + fCountSize(training_label) + fCountSize(dev);
  cerr << "corpus data after processed : " << countSize*sizeof(int)/1024/1024 << "MB" << endl;

  // Initialize model and trainer ------------------------------------------------------------------
  Model model;
  // Use adadelta optimizer
  AdamTrainer adadelta = AdamTrainer(model, 0.0005);
  double slow_start = 0.998;// mid start = 0, begin = 0.998

  cerr << "create optimizer success." << endl;

  // Create model
  EncoderDecoder<GRUBuilder> lm(model,
                                 params.LAYERS,
                                 params.INPUT_DIM,
                                 params.HIDDEN_DIM,
                                 params.ATTENTION_SIZE);
  cerr << "create model success." << endl;

  // Load preexisting weights (if provided)
  if (params.model_file != "") {
    ifstream in(params.model_file);
    boost::archive::text_iarchive ia(in);
    ia >> model >> lm;
    cerr << params.model_file << " has been loaded." << endl;
  }

  // Params -----------------------------------------------------------------------

  cerr << "params.LAYERS = " << params.LAYERS << endl;
  cerr << "params.INPUT_DIM = " << params.INPUT_DIM << endl;
  cerr << "params.HIDDEN_DIM = " << params.HIDDEN_DIM << endl;
  cerr << "params.BATCH_SIZE = " << params.BATCH_SIZE << endl;
  cerr << "params.ATTENTION_SIZE = " << params.ATTENTION_SIZE << endl;
  cerr << "params.print_freq = " << params.print_freq << endl;
  cerr << "params.save_freq = " << params.save_freq << endl;

  // Number of batches in training set
  unsigned num_batches = training.size() / params.BATCH_SIZE;

  // Random indexing
  vector<unsigned> order(num_batches);
  for (unsigned i = 0; i < num_batches; ++i) order[i] = i;
  srand(time(0));
  random_shuffle(order.begin(), order.end()); // shuffle the dataset

  int epoch = 0;
  int cnt_batches = 1;
  // Initialize loss and number of chars(/word) (for loss per char/word)
  	double loss = 0;
  	double sum_loss = 0;
  	unsigned chars = 0;
  // Start timer
    Timer* iteration = new Timer("completed in");
  // Run for the given number of epochs (or indefinitely if params.NUM_EPOCHS is negative)
  while (epoch < params.NUM_EPOCHS || params.NUM_EPOCHS < 0) {
    // Update the optimizer
    if (epoch > 0) adadelta.update_epoch();
    cerr << endl << "start training epoch " << epoch << endl;
    
    for (unsigned si = 0; si < num_batches; ++si, ++cnt_batches) {
      // train a batch
      if (true) {
        // build graph for this instance
        ComputationGraph cg;
        // Compute batch start id and size
        int id = order[si] * params.BATCH_SIZE;
        //cerr << "src sent len = " << training[id].size() << ", tgt sent len = " << training_label[id].size() << endl;
        unsigned bsize = std::min((unsigned)training.size() - id, params.BATCH_SIZE);
        // Encode the batch
        vector<Expression> encoding = lm.encode(training, train_mask, id, bsize, chars, cg);
        // Decode and get error (negative log likelihood)
        Expression loss_expr = lm.decode(encoding, training_label, train_label_mask, id, bsize, cg);
        // Get scalar error for monitoring
        double loss_this_time = as_scalar(cg.forward(loss_expr));
        loss += loss_this_time;
        sum_loss += loss_this_time;
        // Compute gradient with backward pass
        cg.backward(loss_expr);
        // Update parameters
        adadelta.eta = 0.0005 * (1 - slow_start);
        adadelta.update();
        slow_start *= 0.998;
        // print info
        for (auto k = 0 ; k < 100; ++k) cerr << "\b";
        cerr << "already processed " << cnt_batches << " batches, " << cnt_batches*params.BATCH_SIZE << " lines."; // << endl;
      }
      // Print progress every (print_freq) batches
      if (cnt_batches % params.print_freq == 0) {
        // Print informations
        cerr << endl;
        adadelta.status();
        cerr << " loss/batches = " << (loss * params.BATCH_SIZE / params.print_freq) << " ";
        // Reinitialize timer
        delete iteration;
        iteration = new Timer("completed in");
        // Reinitialize loss
        loss = 0;
      }
      // valid & save ---------------------------
      if (cnt_batches % params.save_freq == 0){
        // translation
        ostringstream dev_out_ss;
        dev_out_ss << "dev_" << cnt_batches/params.save_freq << ".out";
        ofstream fout(dev_out_ss.str());
        int miss = 0;
        for (int i = 0; i < dev.size(); i++) {
          ComputationGraph cg;
          vector<unsigned> res = lm.generate(dev[i], miss, cg);
          for (int j = 0; j < res.size()-1 ; ++j) 
            fout << dictOut.convert(res[j]) << " ";
          fout << endl;
        }
        // multi-bleu
        const string bleu_res = "tmp_bleu.res";
        string cmd = "perl multi-bleu.perl " + 
               params.dev_labels_file + " < " + dev_out_ss.str() + " > " + bleu_res;
        system(cmd.c_str());
        // readin bleu score
        ifstream fin(bleu_res);
        assert(fin);
        string bleu_str = "";
        getline(fin, bleu_str);
        assert(bleu_str != "");
        // save each model
        ostringstream model_out_ss;
        model_out_ss 
            << params.exp_name 
            << "_" << (cnt_batches/params.save_freq) 
            << '_' << params.LAYERS
            << '_' << params.INPUT_DIM
            << '_' << params.HIDDEN_DIM 
            << "_tloss=" << (sum_loss * params.BATCH_SIZE / params.save_freq) 
            << "_" << bleu_str.substr(0, 10)
            << ".params";
        ofstream out(model_out_ss.str());
        boost::archive::text_oarchive oa(out);
        oa << model << lm;
        // Reinitialize sum_loss
        sum_loss = 0;
      }
    }
    cerr << endl << "end training an epoch" << endl << endl;

    // Increment epoch
    ++epoch;
  }
  // Free memory
  delete iteration;
}

