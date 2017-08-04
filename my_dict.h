#ifndef MY_DICT_H_
#define MY_DICT_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>

using namespace std;

namespace XC {

class Dict {

private:
  const int _limit = 30000;
  vector<string> _words;
  unordered_map<string, int> _word2id;

  typedef pair<string, int> _psi;
  static bool _comp(const _psi& aa, const _psi& bb) {
    return aa.second > bb.second;
  }

public:
  Dict(const string& fileName) {

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

    //_words.push_back(string("<s>")); // kSOS
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

}
#endif
