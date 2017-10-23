#include <bits/stdc++.h>
#include "my_dict.h"

using namespace std;

int main(){
    string s1 = "1234123412341233412341234123412340";
    vector<int> v1;
    for (int i=0;i<s1.length();i++)
        v1.push_back(s1[i]-'0');
    string s2 = "2341231234123412312312313123443130";
    vector<int> v2;
    for (int i=0;i<s2.length();i++)
        v2.push_back(s2[i]-'0');
    auto res = calcBleu(vector<vector<int>>(1,v1),v2);
    for (auto r:res){
        cout<<r<<" ";
    }
    return 0;
}