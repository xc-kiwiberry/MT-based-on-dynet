#ifndef MY_DICT_H 
#define MY_DICT_H 

#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

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

vector<float> calcBleu(const vector<vector<int>>& sample_sents, const vector<int>& ref_sent,const int n = 4) {
  // ref_dic
  unordered_map<vector<int>, int> ref_dic;
  for (int j = 0; j < ref_sent.size()-1; j++){
      vector<int> tmp;
      for (int k = 0; k < n && j+k < ref_sent.size()-1; k++){
        tmp.push_back(ref_sent[j+k]);
        ref_dic[tmp]++;
      }
    }
  }
  vector<float> final_bleu;
  // each sample
  for (int i = 0; i < sample_sents.size(); i++){
    vector<int>& hyp_sent = sample_sents[i];
    vector<int>cnt(n,0);
    unordered_map<vector<int>, int> hyp_dic;
    for (int j = 0; j < hyp_sent.size()-1; j++){
      vector<int> tmp;
      for (int k = 0; k < n && j+k < hyp_sent.size()-1; k++){
        tmp.push_back(hyp_sent[j+k]);
        hyp_dic[tmp]++;
      }
    }
    for (auto pr: hyp_dic){
      auto it = ref_dic.find(pr.first);
      if (it != ref_dic.end()){
        cnt[pr.first.size()-1] += min(pr.second, it->second);
      }
    }
    vector<float> bleu(n,0);
    int smooth = 0;
    for (int j = 0; j < n; j++){
      if (0 == cnt[j]) smooth = 1;
    }
    for (int j = 0; j < n; j++){
      if (hyp_sent.size()-1 > j)
        bleu[j] = 1. * (cnt[j] + smooth) / (hyp_sent.size()-1 - j + smooth);
      else
        bleu[j] = 1.
    }
    float brev_penalty = 1.;
    if (hyp_sent.size() < ref_sent.size())
      brev_penalty = exp(1 - 1. * (ref_sent.size()-1) / (hyp_sent.size()-1) );
    float logsum = 0.;
    for (int j = 0; j < n; j++)
      logsum += log(bleu[j]);
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
  // unique
  sort(hyp_sents.begin(), hyp_sents.end());
  hyp_sents.erase( unique(hyp_sents.begin(), hyp_sents.end()), hyp_sents.end() );
  for (auto& sent: hyp_sents){
    if (kEOS == sent[0]) {
      hyp_sents.erase(sent);
      break;
    }
  }
  // calc bleu
  hyp_bleu = calcBleu(hyp_sents, ref_sent);
  // add ref_sent to hyp_sents
  hyp_bleu.push_back(1.0);
  hyp_sents.push_back(ref_sent);
  // add padding
  unsigned maxLength = 0;
  for (auto& sent: hyp_sents){
    maxLength = max(maxLength, sent.size());
  }
  for (auto& sent: hyp_sents){
    while(sent.size() < maxLength)
      sent.push_back(kEOS);
  }
  // mask
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