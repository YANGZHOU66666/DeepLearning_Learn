// 以下代码为 a+b 示例，包含输入输出使用，仅供参考，请结合面试官具体题目进行修改

// A[i][j] 本身的奖励
// B[m][n], m,n, 转移奖励


#include <bits/stdc++.h>
using namespace std;

int main() {
    int T,L;
    cin>>T>>L;
    vector<vector<int>> A(T,vector<int>(L));
    vector<vector<int>> B(L,vector<int>(L));
    for(int i=0;i<T;i++){
      for(int j=0;j<L;j++){
        cin>>A[i][j];
      }
    }
    for(int i=0;i<L;i++){
      for(int j=0;j<L;j++){
        cin>>B[i][j];
      }
    }
    // dp[i][j] 表示到第i个城市第j个补给站的最大奖励
    vector<vector<int>> dp(T,vector<int>(L));
    for(int j=0;j<L;j++){
      dp[0][j] = A[0][j];
    }
    for(int i=1;i<T;i++){//i:当前城市
      for(int j=0;j<L;j++){ //j:当前城市补给站
        for(int k=0;k<L;k++){ //k: 钱一个城市补给站
          dp[i][j]=max(dp[i-1][k]+A[i][j]+B[k][j],dp[i][j]);
        }
      }
    }
    int ans = 0;
    for(int i=0;i<L;i++){
      ans = max(ans, dp[T-1][i]);
    }
    cout<<ans;
}