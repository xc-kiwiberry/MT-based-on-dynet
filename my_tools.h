#ifndef MY_TOOLS_H 
#define MY_TOOLS_H 

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/lstm.h"
#include "dynet/expr.h"
#include "dynet/tensor.h"
#include "dynet/io.h"
#include "dynet/param-init.h"

#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;
using namespace dynet;

const int kEOS = 0;
const int kUNK = 1;
extern const int kEOS;
extern const int kUNK;

class Dict {

private:
  const int _limit;
  vector<string> _words;
  unordered_map<string, int> _word2id;

  typedef pair<string, int> _psi;
  static bool _comp(const _psi& aa, const _psi& bb) {
    return aa.second > bb.second;
  }

public:
  Dict(const string& fileName, const int limit): _limit(limit) {

    _word2id.clear();

    ifstream in(fileName);
    assert(in);
    string line;
    while (getline(in, line)) {
      istringstream lin(line);
      string word;
      while(lin) {
        lin >> word;
        if (!lin || word.empty()) break;
        _word2id[word]++; // count
      }
    }

    vector<_psi> temp;
    for (auto pr : _word2id) {
      temp.push_back(pr);
    }
    sort(temp.begin(), temp.end(), _comp);

    _word2id.clear();
    _words.clear();

    _words.push_back(string("</s>")); // kEOS = 0
    _words.push_back(string("<unk>")); // kUNK = 1
    for (int i = 0; i < temp.size() && i < _limit; i++) {
      _words.push_back(temp[i].first);
    }
    for (int i = 0; i < _words.size(); i++) {
      _word2id.insert(_psi(_words[i], i));
    }

  }

  inline int size() const { return _words.size(); }

  inline int convert(const string& word) {
    auto it = _word2id.find(word);
    if (it == _word2id.end()) 
      return _word2id[string("<unk>")];
    return it->second;
  }
  
  inline const string& convert(const int& id) const {
    assert(id < _words.size());
    return _words[id];
  }

  vector<int> read_sentence(const string& line) {
    vector<int> res;
    istringstream lin(line);
    string word;
    while(lin) {
      lin >> word;
      if (!lin || word.empty()) break;
      res.push_back(convert(word));
    }
    return res;
  }

};

void read_corpus(const string &fileName, const string varName, Dict &dict, vector<vector<int>>& vec){
  string line;
  int line_cnt = 0;
  int token_cnt = 0;
  cerr << "Reading " << varName << " data from " << fileName << "..." << endl;
  ifstream in(fileName);
  assert(in); 
  while (getline(in, line)) {
    ++line_cnt;
    auto tmp = dict.read_sentence(line);
    tmp.push_back(kEOS);
    vec.push_back(tmp);
    token_cnt += vec.back().size();
  }
  cerr << line_cnt << " lines, " << token_cnt << " tokens" << endl;
}

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

vector<float> calcBleu(const vector<vector<int>>& sample_sents, const vector<int>& ref_sent, const int n = 4) {
  // ref_dic
  unordered_map<string, int> ref_dic[n];
  for (int j = 0; j < ref_sent.size()-1; j++){
    string tmp = "";
    for (int k = 0; k < n && j+k < ref_sent.size()-1; k++){
      tmp += " " + to_string(ref_sent[j+k]);
      ref_dic[k][tmp]++;
    }
  }
  vector<float> final_bleu;
  // each sample
  for (const auto& hyp_sent: sample_sents){
    vector<int> cnt(n, 0);
    unordered_map<string, int> hyp_dic[n];
    for (int j = 0; j < hyp_sent.size()-1; j++){
      string tmp = "";
      for (int k = 0; k < n && j+k < hyp_sent.size()-1; k++){
        tmp += " " + to_string(hyp_sent[j+k]);
        hyp_dic[k][tmp]++;
      }
    }
    for (int j = 0; j < n; j++){
      for (auto pr: hyp_dic[j]){
        auto it = ref_dic[j].find(pr.first);
        if (it != ref_dic[j].end()){
          cnt[j] += min(pr.second, it->second);
        }
      }
    }
    vector<float> bleu(n, 0.);
    int smooth = 0;
    for (int j = 0; j < n; j++){
      if (0 == cnt[j]) smooth = 1;
    }
    for (int j = 0; j < n; j++){
      if (hyp_sent.size()-1 > j)
        bleu[j] = 1. * (cnt[j] + smooth) / (hyp_sent.size()-1 - j + smooth);
      else
        bleu[j] = 1.;
    }
    float brev_penalty = 1.;
    if (hyp_sent.size() < ref_sent.size())
      brev_penalty = exp(1 - 1. * (ref_sent.size()-1) / (hyp_sent.size()-1) );
    float logsum = 0.;
    for (int j = 0; j < n; j++){
      assert(bleu[j] != 0);
      logsum += log(bleu[j]);
    }
    final_bleu.push_back(brev_penalty * exp(logsum/n));
  }
  return final_bleu;
}

void getMRTBatch(const vector<int>& ref_sent, vector<vector<int>>& hyp_sents, vector<vector<float>>& hyp_masks, vector<float>& hyp_bleu){
  // del padding
  for (auto& sent: hyp_sents){
    for (int i = 0; i < sent.size(); i++){
      if (kEOS == sent[i]){
        sent.erase(sent.begin()+i, sent.end());
        break;
      }
    }
    sent.push_back(kEOS);
  }
  // add ref_sent to hyp_sents
  hyp_sents.push_back(ref_sent);
  // unique
  sort(hyp_sents.begin(), hyp_sents.end());
  hyp_sents.erase( unique(hyp_sents.begin(), hyp_sents.end()), hyp_sents.end() );
    //empty hyp_sent should be 0.0 bleu, not del it 
    /*for (int i = 0; i < hyp_sents.size(); i++){
      if (kEOS == hyp_sents[i][0]) {
        hyp_sents.erase(hyp_sents.begin()+i);
        break;
      }
    }//*/
  // calc bleu
  hyp_bleu.clear();
  hyp_bleu = calcBleu(hyp_sents, ref_sent);
  // add padding
  unsigned maxLength = 0;
  for (auto& sent: hyp_sents){
    maxLength = max(maxLength, (unsigned)sent.size());
  }
  for (auto& sent: hyp_sents){
    while(sent.size() < maxLength)
      sent.push_back(kEOS);
  }
  // mask
  hyp_masks.clear();
  for (int i = 0; i < hyp_sents.size(); i++){
    hyp_masks.push_back(vector<float>());
    hyp_masks[i].push_back(1.);
    for (int j = 1; j < hyp_sents[i].size(); j++){
      if (kEOS == hyp_sents[i][j-1]) hyp_masks[i].push_back(0.);
      else hyp_masks[i].push_back(1.);
    }
  }
  return ;
}

#endif