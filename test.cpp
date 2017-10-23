#include <bits/stdc++.h>
#include "my_dict.h"

using namespace std;

int main(){
    string s1 = "1423143241324312341234132431423143241243134341243341342440";
    vector<int> v1;
    for (int i=0;i<s1.length();i++)
        v1.push_back(s1[i]-'0');
    string s2 = "4123431234431786878667154233413457126753751234142341321232340";
    vector<int> v2;
    for (int i=0;i<s2.length();i++)
        v2.push_back(s2[i]-'0');
    auto res = calcBleu(vector<vector<int>>(1,v1),v2);
    for (auto r:res){
        printf("%.10f ",r);
    }
    printf("\n");
    return 0;
}