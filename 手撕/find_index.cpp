#include<bits/stdc++.h>
using namespace std;

vector<int> nums;
int target = 10;

int main(){
    nums.push_back(0);
    nums.push_back(5);
    nums.push_back(7);
    nums.push_back(3);
    nums.push_back(2); // [0,5,7,3,2]
    unordered_map<int,int> umap; 
    int n=nums.size(); 
    for(int i=0;i<n;i++){
        if(umap.find(target-nums[i])!=umap.end()){
            cout<<i<<" "<<umap[target-nums[i]];
            break;
        }else{
            umap[nums[i]]=i;
        }
    }
}