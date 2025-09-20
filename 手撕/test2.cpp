#include<bits/stdc++.h>
using namespace std;

int main(){
    string s="pwwkew";
    vector<int> cnt(26,0);
    int n=s.length();
    int l=0,r=0;
    cnt[s[0]-'a']=1;
    int max_len = 0;
    string ans=""+s[0];
    while(r<n){
        bool has_multi = false;
        for(int i=0;i<26;i++){
            if(cnt[i]>1){
                has_multi=true;
                break;
            }
        }
        if(has_multi){
            cnt[s[l]-'a']--;
            l++;
        }else{
            if(r-l+1>max_len){
                max_len = r-l+1;
                ans = s.substr(l,r-l+1);
            }
            r++;
            if(r==n){
                break;
            }
            cnt[s[r]-'a']++;
        }
    }
    cout<<max_len;
}