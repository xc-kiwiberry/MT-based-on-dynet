#include "my_encdec.h"

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

double fGiveMaskAndCalcCov(const vector<vector<int>>& lines, vector<vector<float>>& mask) {
  int cntUnk = 0, cntAll = 0;
  for (int i = 0; i < lines.size(); i++) {
    cntAll += lines[i].size();
    mask.push_back(vector<float>());
    assert(lines[i].size() >= 2);
    mask[i].push_back(1.);
    if (lines[i][0] == kUNK) cntUnk++;
    for (int j = 1; j < lines[i].size(); j++) {
      if (lines[i][j-1] != kEOS) mask[i].push_back(1.);
      else mask[i].push_back(0.);
      if (lines[i][j] == kUNK) cntUnk++;
    }
  }
  return 100. - 100.*cntUnk/cntAll;
}

static void handleInt(int sig){
  cerr << endl << "end training success." << endl;
  exit(0);
}

// Datasets
  vector<pp> train_data;
  vector<vector<int>> training, training_label;
  vector<vector<float>> train_mask, train_label_mask;
  vector<vector<int>> dev;

void print_dim(const Dim& d){
  cout<<"({"<<d.d[0];
  for (int i=1;i<d.nd;i++){
    cout<<","<<d.d[i];
  }
  cout<<"},"<<d.bd<<"}"<<endl;
}

void debug(const Expression& x) {
  print_dim(x.dim());
  cout<<x.value()<<endl;
}

int main(int argc, char** argv) {
  // Fetch dynet params ----------------------------------------------------------------------------
  auto dyparams = dynet::extract_dynet_params(argc, argv);
  dynet::initialize(dyparams);

  // debug  
  /*
  ComputationGraph g;
  vector<float> v;
  for (int i=1;i<=24;i++) v.push_back(i);
  Expression x = input(g, Dim({1,24}), v);
  debug(x);
  x=cdiv(x,sum_elems(x));
  debug(x);
  return 0;//*/

  // Fetch program specific parameters (see ../utils/cl-args.h) ------------------------------------
  //Params params;
  get_args(argc, argv, params, TRAIN);
  if (params.mrt_enable) {
    cerr << "Training criteria is MRT." << endl;
    params.BATCH_SIZE = 1;
  }
  else cerr << "Training criteria is MLE." << endl;
  // Load datasets ---------------------------------------------------------------------------------

  // Dictionary
  cerr << "Building dictionary..." << endl;
  Dict dictIn(params.train_file, params.SRC_DIC_LIM), dictOut(params.train_labels_file, params.TGT_DIC_LIM);
  unsigned SRC_VOCAB_SIZE = dictIn.size();
  unsigned TGT_VOCAB_SIZE = dictOut.size();
  cerr << "Dictionary build success." << endl;
  cerr << "SRC_VOCAB_SIZE = " << SRC_VOCAB_SIZE << endl;
  cerr << "TGT_VOCAB_SIZE = " << TGT_VOCAB_SIZE << endl;

  // Read training data
  read_corpus(params.train_file, "training", dictIn, training);
  read_corpus(params.train_labels_file, "training_label", dictOut, training_label);

  assert(training.size() == training_label.size());

  for (int i = 0; i < training.size(); i++) {
    if (training[i].size() <= 1 || training_label[i].size() <= 1) continue;
    if (training[i].size() > params.sent_length_limit || 
          training_label[i].size() > params.sent_length_limit) continue;
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

  double ratioTrain = fGiveMaskAndCalcCov(training, train_mask);              
  double ratioTrainLabel = fGiveMaskAndCalcCov(training_label, train_label_mask); 
  cerr << "corpus processed successfully. " << endl;
  cerr << "In training set, Dictionary covers " << setprecision(3) << ratioTrain << "% words." << endl;
  cerr << "In training_label set, Dictionary covers " << setprecision(3) << ratioTrainLabel << "% words." << endl;

  // Initialize model and trainer ------------------------------------------------------------------
  ParameterCollection model;
  // Use adam optimizer
  double init_learning_rate;
  if (params.mrt_enable) init_learning_rate = 0.00001; // MRT
  else init_learning_rate = 0.0005; // MLE
  AdamTrainer adam = AdamTrainer(model, init_learning_rate);
  adam.sparse_updates_enabled = false;
  double slow_start = 0.998;
  double lr_decay = 1.0;

  cerr << "create optimizer success." << endl;

  // Create model
  EncoderDecoder lm(model,
                    params.LAYERS,
                    params.INPUT_DIM,
                    params.HIDDEN_DIM,
                    params.ATTENTION_SIZE,
                    SRC_VOCAB_SIZE,
                    TGT_VOCAB_SIZE,
                    0.08);
  
  // Load preexisting weights (if provided)
  if (params.model_file != "") {
    TextFileLoader loader(params.model_file);
    loader.populate(model);
    cerr << params.model_file << " has been loaded." << endl;
  }
  else cerr << "create model from scratch success." << endl;

  // Params -----------------------------------------------------------------------

  cerr << "params.LAYERS = " << params.LAYERS << endl;
  cerr << "params.INPUT_DIM = " << params.INPUT_DIM << endl;
  cerr << "params.HIDDEN_DIM = " << params.HIDDEN_DIM << endl;
  cerr << "params.BATCH_SIZE = " << params.BATCH_SIZE << endl;
  cerr << "params.print_freq = " << params.print_freq << endl;
  cerr << "params.save_freq = " << params.save_freq << endl;
  cerr << "params.sent_length_limit = " << params.sent_length_limit << endl;
  if (params.mrt_enable){
    cerr << "params.mrt_sampleSize = " << params.mrt_sampleSize << endl;
    cerr << "params.mrt_lenRatio = " << params.mrt_lenRatio << endl;
    cerr << "params.mrt_alpha = " << params.mrt_alpha << endl;
  }

  // Number of batches in training set
  unsigned num_batches = training.size() / params.BATCH_SIZE;

  // Random indexing
  vector<unsigned> order(num_batches);
  for (unsigned i = 0; i < num_batches; ++i) order[i] = i;
  srand(time(0));
  random_shuffle(order.begin(), order.end()); // shuffle the dataset

  unsigned iters = 1;
  double best_bleu = 0;
  // Initialize loss 
  double loss = 0;
  double sum_loss = 0;
  // Start timer
  Timer* iteration = new Timer("completed in");
  cerr << endl << "start training" << endl;
  // register signal 
  signal(SIGINT, handleInt);
  // open log
  ofstream ofs_log("log_" + params.exp_name, ios::out|ios::app);
  ofs_log << endl << "Iteration\t\tloss\t\tbleu\t\tbest" <<endl;
  ofs_log << "----------------------------------------------------" << endl;
  mkdir("models", 0755);

  // Run indefinitely
  while (true) {
    for (unsigned si = 0; si < num_batches; ++si, ++iters) {
      // train a batch
      if (params.mrt_enable){ // MRT
        const vector<int>& ref_sent = training_label[order[si]];
        // sample
        ComputationGraph cg;
        vector<Expression> encoding = lm.encode(training, train_mask, order[si], 1, cg);
        vector<vector<int>> hyp_sents = lm.sample(encoding, ref_sent.size(), cg);
        cg.clear();
        // process samples
        vector<vector<float>> hyp_masks;
        vector<float> hyp_bleu;
        getMRTBatch(ref_sent, hyp_sents, hyp_masks, hyp_bleu);
        unsigned sampleNum = hyp_sents.size();
        unsigned sentLen = hyp_sents[0].size();
        // decode
        encoding = lm.encode(training, train_mask, order[si], 1, cg);
        Expression loss_expr = lm.decode(encoding, hyp_sents, hyp_masks, 0, sampleNum, cg);
        // calc loss
        loss_expr = reshape(loss_expr, {sampleNum, sentLen});
        loss_expr = transpose(loss_expr);
        loss_expr = reshape(loss_expr, Dim({sentLen}, sampleNum));
        loss_expr = sum_elems(loss_expr);
        loss_expr = reshape(loss_expr, {sampleNum});

        loss_expr = loss_expr * params.mrt_alpha;
        Expression mm = pick(loss_expr, unsigned(0));
        for (int i = 1; i < sampleNum; i++)
          mm = min(mm, pick(loss_expr, i));
        loss_expr = loss_expr - mm;
        loss_expr = exp(-loss_expr); 
        loss_expr = cdiv(loss_expr, sum_elems(loss_expr));
        loss_expr = cmult(loss_expr, input(cg, {sampleNum}, hyp_bleu));
        loss_expr = -sum_elems(loss_expr);

        double loss_this_time = as_scalar(cg.forward(loss_expr));
        loss += loss_this_time;
        sum_loss += loss_this_time;

        cg.backward(loss_expr);
      }
      else { // MLE
        // build graph for this instance
        ComputationGraph cg;
        // Compute batch start id and size
        int id = order[si] * params.BATCH_SIZE;
        //cerr << "src sent len = " << training[id].size() << ", tgt sent len = " << training_label[id].size() << endl;
        unsigned bsize = std::min((unsigned)training.size() - id, params.BATCH_SIZE);
        // Encode the batch
        vector<Expression> encoding = lm.encode(training, train_mask, id, bsize, cg);
        // Decode and get error (negative log likelihood)
        Expression loss_batched = lm.decode(encoding, training_label, train_label_mask, id, bsize, cg);
        Expression loss_expr = sum_batches(loss_batched)/(float)bsize;
        // Get scalar error for monitoring
        double loss_this_time = as_scalar(cg.forward(loss_expr));
        loss += loss_this_time;
        sum_loss += loss_this_time;
        // Compute gradient with backward pass
        cg.backward(loss_expr);
      }

      // Update parameters, adam slow start
      slow_start *= 0.998;
      adam.learning_rate = init_learning_rate * (1 - slow_start) * lr_decay;
      adam.update();
      // print info
      for (auto k = 0 ; k < 100; ++k) cerr << "\b";
      cerr << "already processed " << iters << " batches, " << iters*params.BATCH_SIZE << " lines."; // << endl;

      // Print progress every (print_freq) batches
      if (iters % params.print_freq == 0) {
        // Print informations
        cerr << endl;
        cerr << "loss/batches = " << (loss * params.BATCH_SIZE / params.print_freq) << " ";
        // Reinitialize timer
        delete iteration;
        iteration = new Timer("completed in");
        // Reinitialize loss
        loss = 0;
      }
      // valid & save ---------------------------
      if (iters % params.save_freq == 0){
        cerr << endl << "start validation..." << endl;
        // translation
        ofstream ofs_dev_trans(".tmp_dev_trans");
        int miss = 0;
        for (int i = 0; i < dev.size(); i++) {
          ComputationGraph cg;
          vector<unsigned> res = lm.generate(dev[i], miss, cg);
          for (int j = 0; j < res.size()-1 ; ++j) 
            ofs_dev_trans << dictOut.convert(res[j]) << " ";
          ofs_dev_trans << endl;
          for (int j = 0; j < 100; j++) cerr << "\b";
          cerr << "already translated " << i+1 << " sents. ";
        }
        cerr << endl << "translation completed... " << miss << " sents can't be translated...";
        delete iteration;
        iteration = new Timer("completed in");
        // multi-bleu
        string cmd = "perl ../multi-bleu.perl " + 
               params.dev_labels_file + " < .tmp_dev_trans > .tmp_bleu_res";
        system(cmd.c_str());
        // readin bleu score
        ifstream ifs_bleu_res(".tmp_bleu_res"); assert(ifs_bleu_res);
        string bleu_str = "";
        getline(ifs_bleu_res, bleu_str); assert(bleu_str != "");
        double cur_bleu;
        sscanf(bleu_str.substr(7, 5).c_str(), "%lf", &cur_bleu);
        // valid info
        ostringstream valid_info_ss;
        valid_info_ss << "iters = " << iters << ":"
            << " loss/batch = " << (sum_loss * params.BATCH_SIZE / params.save_freq)
            << ", cur_bleu = " << cur_bleu
            << ", best_bleu = " << max(cur_bleu, best_bleu)
            << endl;
        cerr << valid_info_ss.str();
        // print log
        ofs_log << iters << "\t" << (sum_loss * params.BATCH_SIZE / params.save_freq) << "\t"
                << cur_bleu << "\t" << max(cur_bleu, best_bleu) << endl;
        // save best model
        if (best_bleu < cur_bleu){
          best_bleu = cur_bleu;
          ostringstream model_name_ss;
          model_name_ss
              << "models//" << params.exp_name << "_best.params";
          TextFileSaver saver(model_name_ss.str());
          saver.save(model);
          cerr << "save model: " << model_name_ss.str() << " success." << endl;
        }
        // save checkpoint model
        if (iters % (params.save_freq*10) == 0) {
          ostringstream model_name_ss;
          model_name_ss << "models//" << params.exp_name << "_iter_" << iters << ".params";
          TextFileSaver saver(model_name_ss.str());
          saver.save(model);
          cerr << "save model: " << model_name_ss.str() << " success." << endl;
        }
        cerr << endl;
        // Reinitialize sum_loss
        sum_loss = 0;
      }
    }
    lr_decay /= 2;
  }

  // Free memory
  delete iteration;
}
