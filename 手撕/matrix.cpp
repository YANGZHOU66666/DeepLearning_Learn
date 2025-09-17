#include<bits/stdc++.h>
using namespace std;

int main(){
    vector<vector<int>> matrix;
    int u=0,d=matrix.size()-1;
    int l=0,r=matrix[0].size()-1;
    vector<int> ans;
    int epoch = 0;
    while(d>=u&&r>=l){
        if(epoch%4==0){
            for(int i=l;i<=r;i++){
                ans.push_back(matrix[u][i]);
            }
            u++;
        }else if(epoch%4==1){ // 右侧，从上到下
            for(int i=u;i<=d;i++){
                ans.push_back(matrix[i][r]);
            }
            r--;
        }else if(epoch%4==2){ // 下测，从右到左
            for(int i=r;i>=l;i--){
                ans.push_back(matrix[d][i]);
            }
            d--;
        }else{ // 左侧，从下到上
            for(int i=d;i>=u;i--){
                ans.push_back(matrix[i][l]);
            }
            l++;
        }
        epoch++;
    }
    
}