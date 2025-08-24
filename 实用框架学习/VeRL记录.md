# VeRL记录

## 安装（这是个错误的示范，可以跳过，直接看QuickStart的安装部分）

AutoDL服务器环境：

cuda 12.8

比OpenRLHF简单，没遇到被网络问题卡的

```
conda create -n VERL_ENV python=3.11
conda activate VERL_ENV
git clone https://github.com/volcengine/verl.git
cd verl
pip3 install -e .
```



## QuickStart

按照VeRL官方文档中的Quickstart[Quickstart: PPO training on GSM8K dataset — verl documentation](https://verl.readthedocs.io/en/latest/start/quickstart.html)来操作

- step 1：准备数据集

AutoDL没法访问外网，设置国内镜像：

```
export HF_ENDPOINT=https://hf-mirror.com
```

运行官方准备好的example脚本，将gsm8k数据集预处理后转化为parquet格式：（在verl根目录下运行）

```
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
```

根据demo，调整自己的数据集和模型位置的参数（我是下在本地了），运行以下指令：

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log
```

尝试运行后报错：`ModuleNotFoundError: No module named 'flash_attn'`

与OpenRLHF中解决方法类似，直接[Releases · mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)中找版本匹配的whl文件下载后pip install

又报错：ModuleNotFoundError: No module named 'zmq'

安装`pip install pyzmq`

报错：ModuleNotFoundError: No module named 'vllm'

安装`pip install vllm`

**到这步直接挂了，说有个包（好像是numpy）不兼容。遂止。**



### 重新安装环境 & 运行

仔细看了一下官方文档[Installation — verl documentation](https://verl.readthedocs.io/en/latest/start/install.html)，发现写好了依赖安装的脚本

此外，AutoDL服务器分数据盘和系统盘。系统盘空间不大而且无法扩容，最好把conda环境安装在数据盘（挂载在/autodl-tmp）上

```bash
# 进入数据盘
cd /root/autodl-tmp

# 创建一个名为 conda_envs 的文件夹
mkdir -p conda_envs

# 创建conda环境(在数据盘上创建)
conda create --prefix /root/autodl-tmp/conda_envs/verl_env python=3.10 -y
conda activate /root/autodl-tmp/conda_envs/verl_env

# 在verl根目录下执行，运行安装脚本
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# 安装verl本体
# 确保你仍然在 verl 仓库的根目录下
pip install --no-deps -e .
```

出了一个error：

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

vllm 0.8.5.post1 requires opentelemetry-api<1.27.0,>=1.26.0, but you have opentelemetry-api 1.36.0 which is incompatible.
```

但后续demo也成功跑起来了，不知道实际有没有影响。

最后，执行PPO demo指令

```
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log
```

成功运行了，跑了30步发生了数据盘不够大的问题。在AutoDL平台上另外购买了500G数据盘，重新运行

（补充：回收站在`/root/autodl-tmp/.Trash-0`，使用`rm -rf ./.Trash-0/*`来清空回收站）

在这个demo中，奖励函数是**基于规则**的（直接看答案数字对不对）；每隔10个batch产生一个**检查点**；每隔10个batch**在验证集上验证一次**

- 一组log的示例：跑10个batch，就在整个验证集上验证一下（因为上面设置了test_freq=10）

```bash
(TaskRunner pid=3165) step:71 - global_seqlen/min:69484 - global_seqlen/max:69484 - global_seqlen/minmax_diff:0 - global_seqlen/balanced_min:69484 - global_seqlen/balanced_max:69484 - global_seqlen/mean:69484.0 - actor/entropy:0.1194450631737709 - critic/vf_loss:np.float64(0.007003280876688223) - critic/vf_clipfrac:np.float64(0.0) - critic/vpred_mean:np.float64(0.38008218246977776) - critic/grad_norm:np.float64(42.70210337638855) - perf/mfu/critic:np.float64(0.0) - critic/lr:np.float64(1e-05) - actor/pg_loss:np.float64(-0.0006611689302644663) - actor/pg_clipfrac:np.float64(0.004412197675264906) - actor/ppo_kl:np.float64(0.00019574521718368487) - actor/pg_clipfrac_lower:np.float64(0.0) - actor/grad_norm:np.float64(3.804414212703705) - perf/mfu/actor:np.float64(0.0) - perf/max_memory_allocated_gb:np.float64(20.764535427093506) - perf/max_memory_reserved_gb:np.float64(25.763671875) - perf/cpu_memory_used_gb:np.float64(40.79539108276367) - actor/lr:np.float64(1e-06) - training/global_step:71 - training/epoch:2 - critic/score/mean:0.63671875 - critic/score/max:1.0 - critic/score/min:0.0 - critic/rewards/mean:0.63671875 - critic/rewards/max:1.0 - critic/rewards/min:0.0 - critic/advantages/mean:-1.710466790427745e-08 - critic/advantages/max:2.210120677947998 - critic/advantages/min:-2.4786856174468994 - critic/returns/mean:0.5699439644813538 - critic/returns/max:1.0 - critic/returns/min:0.0 - critic/values/mean:0.40234375 - critic/values/max:0.9609375 - critic/values/min:-0.3515625 - critic/vf_explained_var:0.27253496646881104 - response_length/mean:167.265625 - response_length/max:256.0 - response_length/min:65.0 - response_length/clip_ratio:0.08203125 - response_length_non_aborted/mean:167.265625 - response_length_non_aborted/max:256.0 - response_length_non_aborted/min:65.0 - response_length_non_aborted/clip_ratio:0.08203125 - response/aborted_ratio:0.0 - prompt_length/mean:104.15625 - prompt_length/max:186.0 - prompt_length/min:65.0 - prompt_length/clip_ratio:0.0 - timing_s/start_profile:0.00018977653235197067 - timing_s/generate_sequences:3.5231003761291504 - timing_s/reshard:0.15185357630252838 - timing_s/generation_timing/max:3.5231003761291504 - timing_s/generation_timing/min:3.5231003761291504 - timing_s/generation_timing/topk_ratio:0.0 - timing_s/gen:4.229507855139673 - timing_s/reward:0.0805518813431263 - timing_s/old_log_prob:5.541149379685521 - timing_s/values:4.978008586913347 - timing_s/adv:0.08389680925756693 - timing_s/update_critic:19.744506244547665 - timing_s/update_actor:22.248339109122753 - timing_s/step:56.93468872085214 - timing_s/stop_profile:0.00020517967641353607 - timing_per_token_ms/gen:0.09877412085800263 - timing_per_token_ms/adv:0.0012074263032866117 - timing_per_token_ms/update_critic:0.28415903293632583 - timing_per_token_ms/update_actor:0.32019370083936954 - timing_per_token_ms/values:0.07164251607439623 - perf/total_num_tokens:69484 - perf/time_per_step:56.93468872085214 - perf/throughput:1220.4159109515201
Training Progress:  16%|██████████████████████▌                                                                                                                   | 71/435 [1:10:40<6:31:06, 64.47s/it]
(TaskRunner pid=3165) step:72 - global_seqlen/min:67965 - global_seqlen/max:67965 - global_seqlen/minmax_diff:0 - global_seqlen/balanced_min:67965 - global_seqlen/balanced_max:67965 - global_seqlen/mean:67965.0 - actor/entropy:0.12458629906177521 - critic/vf_loss:np.float64(0.006612617203245463) - critic/vf_clipfrac:np.float64(0.0) - critic/vpred_mean:np.float64(0.7531495024450123) - critic/grad_norm:np.float64(38.220882415771484) - perf/mfu/critic:np.float64(0.0) - critic/lr:np.float64(1e-05) - actor/pg_loss:np.float64(-0.0005397239583544433) - actor/pg_clipfrac:np.float64(0.004821434231416788) - actor/ppo_kl:np.float64(0.0006169108633926612) - actor/pg_clipfrac_lower:np.float64(0.0) - actor/grad_norm:np.float64(4.414253294467926) - perf/mfu/actor:np.float64(0.0) - perf/max_memory_allocated_gb:np.float64(20.764535427093506) - perf/max_memory_reserved_gb:np.float64(25.763671875) - perf/cpu_memory_used_gb:np.float64(41.050289154052734) - actor/lr:np.float64(1e-06) - training/global_step:72 - training/epoch:2 - critic/score/mean:0.625 - critic/score/max:1.0 - critic/score/min:0.0 - critic/rewards/mean:0.625 - critic/rewards/max:1.0 - critic/rewards/min:0.0 - critic/advantages/mean:-2.043995195322168e-08 - critic/advantages/max:2.4603822231292725 - critic/advantages/min:-2.3053226470947266 - critic/returns/mean:0.5565363168716431 - critic/returns/max:1.0 - critic/returns/min:0.0 - critic/values/mean:0.71484375 - critic/values/max:1.3046875 - critic/values/min:-0.1962890625 - critic/vf_explained_var:0.300315797328949 - response_length/mean:163.30078125 - response_length/max:256.0 - response_length/min:60.0 - response_length/clip_ratio:0.09765625 - response_length_non_aborted/mean:163.30078125 - response_length_non_aborted/max:256.0 - response_length_non_aborted/min:60.0 - response_length_non_aborted/clip_ratio:0.09765625 - response/aborted_ratio:0.0 - prompt_length/mean:102.1875 - prompt_length/max:201.0 - prompt_length/min:70.0 - prompt_length/clip_ratio:0.0 - timing_s/start_profile:0.0001650974154472351 - timing_s/generate_sequences:3.5199191570281982 - timing_s/reshard:0.4239424765110016 - timing_s/generation_timing/max:3.5199191570281982 - timing_s/generation_timing/min:3.5199191570281982 - timing_s/generation_timing/topk_ratio:0.0 - timing_s/gen:4.331033235415816 - timing_s/reward:0.0824702475219965 - timing_s/old_log_prob:5.655994229950011 - timing_s/values:4.978985144756734 - timing_s/adv:0.07867959886789322 - timing_s/update_critic:18.1204649284482 - timing_s/update_actor:21.153391116298735 - timing_s/step:54.416298444382846 - timing_s/stop_profile:9.603891521692276e-05 - timing_per_token_ms/gen:0.10360084285171191 - timing_per_token_ms/adv:0.0011576487731610862 - timing_per_token_ms/update_critic:0.266614653548859 - timing_per_token_ms/update_actor:0.31123947791214207 - timing_per_token_ms/values:0.0732580761385527 - perf/total_num_tokens:67965 - perf/time_per_step:54.416298444382846 - perf/throughput:1248.9824178222054
Training Progress:  17%|██████████████████████▊                                                                                                                   | 72/435 [1:11:35<6:11:49, 61.46s/it]
(TaskRunner pid=3165) step:73 - global_seqlen/min:68899 - global_seqlen/max:68899 - global_seqlen/minmax_diff:0 - global_seqlen/balanced_min:68899 - global_seqlen/balanced_max:68899 - global_seqlen/mean:68899.0 - actor/entropy:0.1407710760831833 - critic/vf_loss:np.float64(0.007185004837083397) - critic/vf_clipfrac:np.float64(0.0) - critic/vpred_mean:np.float64(0.35732312756590545) - critic/grad_norm:np.float64(46.652456283569336) - perf/mfu/critic:np.float64(0.0) - critic/lr:np.float64(1e-05) - actor/pg_loss:np.float64(-0.0007258972000272479) - actor/pg_clipfrac:np.float64(0.005018776213546516) - actor/ppo_kl:np.float64(0.0005250735913762128) - actor/pg_clipfrac_lower:np.float64(0.0) - actor/grad_norm:np.float64(3.880192458629608) - perf/mfu/actor:np.float64(0.0) - perf/max_memory_allocated_gb:np.float64(20.764535427093506) - perf/max_memory_reserved_gb:np.float64(25.763671875) - perf/cpu_memory_used_gb:np.float64(40.67334747314453) - actor/lr:np.float64(1e-06) - training/global_step:73 - training/epoch:2 - critic/score/mean:0.6328125 - critic/score/max:1.0 - critic/score/min:0.0 - critic/rewards/mean:0.6328125 - critic/rewards/max:1.0 - critic/rewards/min:0.0 - critic/advantages/mean:-1.5126968122558537e-08 - critic/advantages/max:2.5700387954711914 - critic/advantages/min:-2.770606517791748 - critic/returns/mean:0.5609214901924133 - critic/returns/max:1.0 - critic/returns/min:0.0 - critic/values/mean:0.37890625 - critic/values/max:0.98046875 - critic/values/min:-0.72265625 - critic/vf_explained_var:0.3145960569381714 - response_length/mean:165.4921875 - response_length/max:256.0 - response_length/min:53.0 - response_length/clip_ratio:0.09765625 - response_length_non_aborted/mean:165.4921875 - response_length_non_aborted/max:256.0 - response_length_non_aborted/min:53.0 - response_length_non_aborted/clip_ratio:0.09765625 - response/aborted_ratio:0.0 - prompt_length/mean:103.64453125 - prompt_length/max:193.0 - prompt_length/min:68.0 - prompt_length/clip_ratio:0.0 - timing_s/start_profile:8.964817970991135e-05 - timing_s/generate_sequences:3.5031440258026123 - timing_s/reshard:0.43309640884399414 - timing_s/generation_timing/max:3.5031440258026123 - timing_s/generation_timing/min:3.5031440258026123 - timing_s/generation_timing/topk_ratio:0.0 - timing_s/gen:4.337046857923269 - timing_s/reward:0.08430998213589191 - timing_s/old_log_prob:5.5768828969448805 - timing_s/values:4.97827292047441 - timing_s/adv:0.06391743756830692 - timing_s/update_critic:18.858808710239828 - timing_s/update_actor:22.23590332362801 - timing_s/step:56.16187227983028 - timing_s/stop_profile:0.00019486155360937119 - timing_per_token_ms/gen:0.10237093088616507 - timing_per_token_ms/adv:0.0009276976090844123 - timing_per_token_ms/update_critic:0.2737167260807824 - timing_per_token_ms/update_actor:0.3227318730841958 - timing_per_token_ms/values:0.07225464695386595 - perf/total_num_tokens:68899 - perf/time_per_step:56.16187227983028 - perf/throughput:1226.7931463663842
Training Progress:  17%|███████████████████████▏                                                                                                                  | 73/435 [1:12:31<6:01:14, 59.87s/it]
(TaskRunner pid=3165) step:74 - global_seqlen/min:70497 - global_seqlen/max:70497 - global_seqlen/minmax_diff:0 - global_seqlen/balanced_min:70497 - global_seqlen/balanced_max:70497 - global_seqlen/mean:70497.0 - actor/entropy:0.12836657464504242 - critic/vf_loss:np.float64(0.006463623778472538) - critic/vf_clipfrac:np.float64(0.0) - critic/vpred_mean:np.float64(0.7206990979611874) - critic/grad_norm:np.float64(31.705189645290375) - perf/mfu/critic:np.float64(0.0) - critic/lr:np.float64(1e-05) - actor/pg_loss:np.float64(-0.0018774090762576634) - actor/pg_clipfrac:np.float64(0.0057396279535169015) - actor/ppo_kl:np.float64(0.00014932769181541516) - actor/pg_clipfrac_lower:np.float64(2.2163119865581393e-05) - actor/grad_norm:np.float64(3.631416380405426) - perf/mfu/actor:np.float64(0.0) - perf/max_memory_allocated_gb:np.float64(20.764535427093506) - perf/max_memory_reserved_gb:np.float64(25.763671875) - perf/cpu_memory_used_gb:np.float64(40.897857666015625) - actor/lr:np.float64(1e-06) - training/global_step:74 - training/epoch:2 - critic/score/mean:0.609375 - critic/score/max:1.0 - critic/score/min:0.0 - critic/rewards/mean:0.609375 - critic/rewards/max:1.0 - critic/rewards/min:0.0 - critic/advantages/mean:-1.2722431108613819e-08 - critic/advantages/max:2.2361536026000977 - critic/advantages/min:-2.4779598712921143 - critic/returns/mean:0.5505709052085876 - critic/returns/max:1.0 - critic/returns/min:0.0 - critic/values/mean:0.67578125 - critic/values/max:1.2890625 - critic/values/min:-0.21484375 - critic/vf_explained_var:0.28041839599609375 - response_length/mean:168.66015625 - response_length/max:256.0 - response_length/min:68.0 - response_length/clip_ratio:0.10546875 - response_length_non_aborted/mean:168.66015625 - response_length_non_aborted/max:256.0 - response_length_non_aborted/min:68.0 - response_length_non_aborted/clip_ratio:0.10546875 - response/aborted_ratio:0.0 - prompt_length/mean:106.71875 - prompt_length/max:222.0 - prompt_length/min:68.0 - prompt_length/clip_ratio:0.0 - timing_s/start_profile:0.0001744050532579422 - timing_s/generate_sequences:3.7557709217071533 - timing_s/reshard:0.4463890492916107 - timing_s/generation_timing/max:3.7557709217071533 - timing_s/generation_timing/min:3.7557709217071533 - timing_s/generation_timing/topk_ratio:0.0 - timing_s/gen:4.639420366846025 - timing_s/reward:0.09677632246166468 - timing_s/old_log_prob:5.630885478109121 - timing_s/values:5.009235210716724 - timing_s/adv:0.074988117441535 - timing_s/update_critic:19.7498179115355 - timing_s/update_actor:21.35617745295167 - timing_s/step:56.58376076631248 - timing_s/stop_profile:0.0002151578664779663 - timing_per_token_ms/gen:0.10745119778692418 - timing_per_token_ms/adv:0.0010637065044120317 - timing_per_token_ms/update_critic:0.2801511824834461 - timing_per_token_ms/update_actor:0.3029373938316761 - timing_per_token_ms/values:0.07105600537209703 - perf/total_num_tokens:70497 - perf/time_per_step:56.58376076631248 - perf/throughput:1245.8874957277646
Training Progress:  17%|███████████████████████▍                                                                                                                  | 74/435 [1:13:28<5:54:20, 58.89s/it]
(TaskRunner pid=3165) step:75 - global_seqlen/min:68493 - global_seqlen/max:68493 - global_seqlen/minmax_diff:0 - global_seqlen/balanced_min:68493 - global_seqlen/balanced_max:68493 - global_seqlen/mean:68493.0 - actor/entropy:0.12433101236820221 - critic/vf_loss:np.float64(0.005077005092971376) - critic/vf_clipfrac:np.float64(0.0) - critic/vpred_mean:np.float64(0.5866734087467194) - critic/grad_norm:np.float64(17.102087020874023) - perf/mfu/critic:np.float64(0.0) - critic/lr:np.float64(1e-05) - actor/pg_loss:np.float64(-0.0013763950801148894) - actor/pg_clipfrac:np.float64(0.004070283694090904) - actor/ppo_kl:np.float64(0.0010842479655082116) - actor/pg_clipfrac_lower:np.float64(0.0) - actor/grad_norm:np.float64(3.9007189869880676) - perf/mfu/actor:np.float64(0.0) - perf/max_memory_allocated_gb:np.float64(20.764535427093506) - perf/max_memory_reserved_gb:np.float64(25.763671875) - perf/cpu_memory_used_gb:np.float64(38.11729431152344) - actor/lr:np.float64(1e-06) - training/global_step:75 - training/epoch:2 - critic/score/mean:0.59375 - critic/score/max:1.0 - critic/score/min:0.0 - critic/rewards/mean:0.59375 - critic/rewards/max:1.0 - critic/rewards/min:0.0 - critic/advantages/mean:-1.454464659822463e-09 - critic/advantages/max:2.413951873779297 - critic/advantages/min:-2.507028579711914 - critic/returns/mean:0.5346964001655579 - critic/returns/max:1.0 - critic/returns/min:0.0 - critic/values/mean:0.625 - critic/values/max:1.2890625 - critic/values/min:-0.294921875 - critic/vf_explained_var:0.3649803400039673 - response_length/mean:163.921875 - response_length/max:256.0 - response_length/min:67.0 - response_length/clip_ratio:0.0625 - response_length_non_aborted/mean:163.921875 - response_length_non_aborted/max:256.0 - response_length_non_aborted/min:67.0 - response_length_non_aborted/clip_ratio:0.0625 - response/aborted_ratio:0.0 - prompt_length/mean:103.62890625 - prompt_length/max:202.0 - prompt_length/min:69.0 - prompt_length/clip_ratio:0.0 - timing_s/start_profile:0.00022384803742170334 - timing_s/generate_sequences:3.722923755645752 - timing_s/reshard:0.4539295732975006 - timing_s/generation_timing/max:3.722923755645752 - timing_s/generation_timing/min:3.722923755645752 - timing_s/generation_timing/topk_ratio:0.0 - timing_s/gen:4.577980321832001 - timing_s/reward:0.08015729300677776 - timing_s/old_log_prob:5.621704796329141 - timing_s/values:4.986771849915385 - timing_s/adv:0.079194494523108 - timing_s/update_critic:19.578828345052898 - timing_s/update_actor:21.587606549263 - timing_s/step:56.544809642247856 - timing_s/stop_profile:0.00017585698515176773 - timing_per_token_ms/gen:0.10909303979201224 - timing_per_token_ms/adv:0.0011562421637701372 - timing_per_token_ms/update_critic:0.2858515227111223 - timing_per_token_ms/update_actor:0.31517974901468765 - timing_per_token_ms/values:0.07280702918422884 - perf/total_num_tokens:68493 - perf/time_per_step:56.544809642247856 - perf/throughput:1211.3048117651629
Training Progress:  17%|███████████████████████▊                                                                                                                  | 75/435 [1:14:24<5:49:13, 58.20s/it]
(TaskRunner pid=3165) step:76 - global_seqlen/min:67032 - global_seqlen/max:67032 - global_seqlen/minmax_diff:0 - global_seqlen/balanced_min:67032 - global_seqlen/balanced_max:67032 - global_seqlen/mean:67032.0 - actor/entropy:0.1241055428981781 - critic/vf_loss:np.float64(0.005068224289061618) - critic/vf_clipfrac:np.float64(0.0) - critic/vpred_mean:np.float64(0.5972371781826951) - critic/grad_norm:np.float64(8.589312434196472) - perf/mfu/critic:np.float64(0.0) - critic/lr:np.float64(1e-05) - actor/pg_loss:np.float64(-0.0003258442581568488) - actor/pg_clipfrac:np.float64(0.00470952297837357) - actor/ppo_kl:np.float64(0.00030826312190868066) - actor/pg_clipfrac_lower:np.float64(2.6393581720185466e-05) - actor/grad_norm:np.float64(3.7193260192871094) - perf/mfu/actor:np.float64(0.0) - perf/max_memory_allocated_gb:np.float64(20.764535427093506) - perf/max_memory_reserved_gb:np.float64(25.763671875) - perf/cpu_memory_used_gb:np.float64(38.44154357910156) - actor/lr:np.float64(1e-06) - training/global_step:76 - training/epoch:2 - critic/score/mean:0.64453125 - critic/score/max:1.0 - critic/score/min:0.0 - critic/rewards/mean:0.64453125 - critic/rewards/max:1.0 - critic/rewards/min:0.0 - critic/advantages/mean:-1.4882630683388243e-09 - critic/advantages/max:2.6314470767974854 - critic/advantages/min:-2.7984001636505127 - critic/returns/mean:0.5935236811637878 - critic/returns/max:1.0 - critic/returns/min:0.0 - critic/values/mean:0.65625 - critic/values/max:1.359375 - critic/values/min:-0.263671875 - critic/vf_explained_var:0.32134073972702026 - response_length/mean:160.19921875 - response_length/max:256.0 - response_length/min:74.0 - response_length/clip_ratio:0.07421875 - response_length_non_aborted/mean:160.19921875 - response_length_non_aborted/max:256.0 - response_length_non_aborted/min:74.0 - response_length_non_aborted/clip_ratio:0.07421875 - response/aborted_ratio:0.0 - prompt_length/mean:101.64453125 - prompt_length/max:171.0 - prompt_length/min:67.0 - prompt_length/clip_ratio:0.0 - timing_s/start_profile:0.00012663938105106354 - timing_s/generate_sequences:3.4605226516723633 - timing_s/reshard:0.41919347643852234 - timing_s/generation_timing/max:3.4605226516723633 - timing_s/generation_timing/min:3.4605226516723633 - timing_s/generation_timing/topk_ratio:0.0 - timing_s/gen:4.281287532299757 - timing_s/reward:0.07928126119077206 - timing_s/old_log_prob:5.580714308656752 - timing_s/values:4.991023601964116 - timing_s/adv:0.07726532965898514 - timing_s/update_critic:18.45755231846124 - timing_s/update_actor:21.773657753132284 - timing_s/step:55.271243763156235 - timing_s/stop_profile:0.00024901796132326126 - timing_per_token_ms/gen:0.10439363907975316 - timing_per_token_ms/adv:0.0011526633497282663 - timing_per_token_ms/update_critic:0.2753543429774024 - timing_per_token_ms/update_actor:0.324824826249139 - timing_per_token_ms/values:0.07445732787271923 - perf/total_num_tokens:67032 - perf/time_per_step:55.271243763156235 - perf/throughput:1212.7825508548349
Training Progress:  17%|████████████████████████                                                                                                                  | 76/435 [1:15:19<5:43:01, 57.33s/it]
(TaskRunner pid=3165) step:77 - global_seqlen/min:69128 - global_seqlen/max:69128 - global_seqlen/minmax_diff:0 - global_seqlen/balanced_min:69128 - global_seqlen/balanced_max:69128 - global_seqlen/mean:69128.0 - actor/entropy:0.12534978985786438 - critic/vf_loss:np.float64(0.0051449700295052025) - critic/vf_clipfrac:np.float64(0.0) - critic/vpred_mean:np.float64(0.5309695736505091) - critic/grad_norm:np.float64(15.854477405548096) - perf/mfu/critic:np.float64(0.0) - critic/lr:np.float64(1e-05) - actor/pg_loss:np.float64(-0.0011657940221994068) - actor/pg_clipfrac:np.float64(0.004530693367996719) - actor/ppo_kl:np.float64(0.00018200124033640463) - actor/pg_clipfrac_lower:np.float64(0.0) - actor/grad_norm:np.float64(3.8609573245048523) - perf/mfu/actor:np.float64(0.0) - perf/max_memory_allocated_gb:np.float64(20.764535427093506) - perf/max_memory_reserved_gb:np.float64(25.763671875) - perf/cpu_memory_used_gb:np.float64(41.07088088989258) - actor/lr:np.float64(1e-06) - training/global_step:77 - training/epoch:2 - critic/score/mean:0.6171875 - critic/score/max:1.0 - critic/score/min:0.0 - critic/rewards/mean:0.6171875 - critic/rewards/max:1.0 - critic/rewards/min:0.0 - critic/advantages/mean:-1.1451248838056927e-08 - critic/advantages/max:2.2576701641082764 - critic/advantages/min:-2.669679880142212 - critic/returns/mean:0.557246744632721 - critic/returns/max:1.0 - critic/returns/min:0.0 - critic/values/mean:0.59375 - critic/values/max:1.3125 - critic/values/min:-0.3515625 - critic/vf_explained_var:0.34460943937301636 - response_length/mean:166.5625 - response_length/max:256.0 - response_length/min:60.0 - response_length/clip_ratio:0.09375 - response_length_non_aborted/mean:166.5625 - response_length_non_aborted/max:256.0 - response_length_non_aborted/min:60.0 - response_length_non_aborted/clip_ratio:0.09375 - response/aborted_ratio:0.0 - prompt_length/mean:103.46875 - prompt_length/max:180.0 - prompt_length/min:65.0 - prompt_length/clip_ratio:0.0 - timing_s/start_profile:0.0002455003559589386 - timing_s/generate_sequences:3.5215556621551514 - timing_s/reshard:0.4265533685684204 - timing_s/generation_timing/max:3.5215556621551514 - timing_s/generation_timing/min:3.5215556621551514 - timing_s/generation_timing/topk_ratio:0.0 - timing_s/gen:4.350967255420983 - timing_s/reward:0.08819806762039661 - timing_s/old_log_prob:5.672923186793923 - timing_s/values:5.014994516968727 - timing_s/adv:0.08312816824764013 - timing_s/update_critic:19.446039838716388 - timing_s/update_actor:22.619679115712643 - timing_s/step:57.305705439299345 - timing_s/stop_profile:0.0002011684700846672 - timing_per_token_ms/gen:0.1020395697800418 - timing_per_token_ms/adv:0.0012025252900075241 - timing_per_token_ms/update_critic:0.28130482349722813 - timing_per_token_ms/update_actor:0.3272144299807986 - timing_per_token_ms/values:0.0725465009398323 - perf/total_num_tokens:69128 - perf/time_per_step:57.305705439299345 - perf/throughput:1206.3022254079629
Training Progress:  18%|████████████████████████▍                                                                                                                 | 77/435 [1:16:17<5:42:03, 57.33s/it]
(TaskRunner pid=3165) step:78 - global_seqlen/min:68463 - global_seqlen/max:68463 - global_seqlen/minmax_diff:0 - global_seqlen/balanced_min:68463 - global_seqlen/balanced_max:68463 - global_seqlen/mean:68463.0 - actor/entropy:0.12547488510608673 - critic/vf_loss:np.float64(0.008831050467961177) - critic/vf_clipfrac:np.float64(0.0) - critic/vpred_mean:np.float64(0.86925945058465) - critic/grad_norm:np.float64(57.1467661857605) - perf/mfu/critic:np.float64(0.0) - critic/lr:np.float64(1e-05) - actor/pg_loss:np.float64(0.0006841664362582378) - actor/pg_clipfrac:np.float64(0.004824477578949882) - actor/ppo_kl:np.float64(0.0001464904675572143) - actor/pg_clipfrac_lower:np.float64(9.97822771751089e-05) - actor/grad_norm:np.float64(3.9034997820854187) - perf/mfu/actor:np.float64(0.0) - perf/max_memory_allocated_gb:np.float64(20.764535427093506) - perf/max_memory_reserved_gb:np.float64(25.763671875) - perf/cpu_memory_used_gb:np.float64(43.12489700317383) - actor/lr:np.float64(1e-06) - training/global_step:78 - training/epoch:2 - critic/score/mean:0.63671875 - critic/score/max:1.0 - critic/score/min:0.0 - critic/rewards/mean:0.63671875 - critic/rewards/max:1.0 - critic/rewards/min:0.0 - critic/advantages/mean:-1.16808109851263e-08 - critic/advantages/max:2.5337748527526855 - critic/advantages/min:-2.4622111320495605 - critic/returns/mean:0.5869336128234863 - critic/returns/max:1.0 - critic/returns/min:0.0 - critic/values/mean:0.8515625 - critic/values/max:1.5234375 - critic/values/min:-0.08740234375 - critic/vf_explained_var:0.19510865211486816 - response_length/mean:163.2890625 - response_length/max:256.0 - response_length/min:72.0 - response_length/clip_ratio:0.05078125 - response_length_non_aborted/mean:163.2890625 - response_length_non_aborted/max:256.0 - response_length_non_aborted/min:72.0 - response_length_non_aborted/clip_ratio:0.05078125 - response/aborted_ratio:0.0 - prompt_length/mean:104.14453125 - prompt_length/max:180.0 - prompt_length/min:69.0 - prompt_length/clip_ratio:0.0 - timing_s/start_profile:0.00014568772166967392 - timing_s/generate_sequences:3.961097240447998 - timing_s/reshard:0.44853296875953674 - timing_s/generation_timing/max:3.961097240447998 - timing_s/generation_timing/min:3.961097240447998 - timing_s/generation_timing/topk_ratio:0.0 - timing_s/gen:4.8027467569336295 - timing_s/reward:0.088444784283638 - timing_s/old_log_prob:5.647057970054448 - timing_s/values:5.032954138703644 - timing_s/adv:0.09241180215030909 - timing_s/update_critic:18.875077043659985 - timing_s/update_actor:21.677237352356315 - timing_s/step:56.243704207241535 - timing_s/stop_profile:0.00017774663865566254 - timing_per_token_ms/gen:0.11489275051274173 - timing_per_token_ms/adv:0.0013498064962141462 - timing_per_token_ms/update_critic:0.27569748687115647 - timing_per_token_ms/update_actor:0.31662704456942165 - timing_per_token_ms/values:0.0735134910638395 - perf/total_num_tokens:68463 - perf/time_per_step:56.243704207241535 - perf/throughput:1217.2562416538915
Training Progress:  18%|████████████████████████▋                                                                                                                 | 78/435 [1:17:13<5:39:12, 57.01s/it]
(TaskRunner pid=3165) step:79 - global_seqlen/min:68419 - global_seqlen/max:68419 - global_seqlen/minmax_diff:0 - global_seqlen/balanced_min:68419 - global_seqlen/balanced_max:68419 - global_seqlen/mean:68419.0 - actor/entropy:0.12251779437065125 - critic/vf_loss:np.float64(0.00645167924631096) - critic/vf_clipfrac:np.float64(0.0) - critic/vpred_mean:np.float64(0.44300227481289767) - critic/grad_norm:np.float64(27.494355708360672) - perf/mfu/critic:np.float64(0.0) - critic/lr:np.float64(1e-05) - actor/pg_loss:np.float64(-0.0008185422466340242) - actor/pg_clipfrac:np.float64(0.004444046200660523) - actor/ppo_kl:np.float64(0.0003012794068126823) - actor/pg_clipfrac_lower:np.float64(0.0) - actor/grad_norm:np.float64(3.7478469610214233) - perf/mfu/actor:np.float64(0.0) - perf/max_memory_allocated_gb:np.float64(20.764535427093506) - perf/max_memory_reserved_gb:np.float64(25.763671875) - perf/cpu_memory_used_gb:np.float64(41.244117736816406) - actor/lr:np.float64(1e-06) - training/global_step:79 - training/epoch:2 - critic/score/mean:0.61328125 - critic/score/max:1.0 - critic/score/min:0.0 - critic/rewards/mean:0.61328125 - critic/rewards/max:1.0 - critic/rewards/min:0.0 - critic/advantages/mean:-3.6389367075173595e-09 - critic/advantages/max:3.1980879306793213 - critic/advantages/min:-2.6789610385894775 - critic/returns/mean:0.5559477210044861 - critic/returns/max:1.0 - critic/returns/min:0.0 - critic/values/mean:0.48046875 - critic/values/max:1.140625 - critic/values/min:-0.546875 - critic/vf_explained_var:0.249786376953125 - response_length/mean:163.796875 - response_length/max:256.0 - response_length/min:86.0 - response_length/clip_ratio:0.0703125 - response_length_non_aborted/mean:163.796875 - response_length_non_aborted/max:256.0 - response_length_non_aborted/min:86.0 - response_length_non_aborted/clip_ratio:0.0703125 - response/aborted_ratio:0.0 - prompt_length/mean:103.46484375 - prompt_length/max:199.0 - prompt_length/min:66.0 - prompt_length/clip_ratio:0.0 - timing_s/start_profile:0.0001844232901930809 - timing_s/generate_sequences:3.564371347427368 - timing_s/reshard:0.46371880173683167 - timing_s/generation_timing/max:3.564371347427368 - timing_s/generation_timing/min:3.564371347427368 - timing_s/generation_timing/topk_ratio:0.0 - timing_s/gen:4.418549383059144 - timing_s/reward:0.07135556172579527 - timing_s/old_log_prob:5.576491332612932 - timing_s/values:5.194905914366245 - timing_s/adv:0.0809581121429801 - timing_s/update_critic:19.018847988918424 - timing_s/update_actor:22.921413354575634 - timing_s/step:57.29716528207064 - timing_s/stop_profile:0.00019677821546792984 - timing_per_token_ms/gen:0.10537416252645102 - timing_per_token_ms/adv:0.0011832694447884374 - timing_per_token_ms/update_critic:0.2779761175831045 - timing_per_token_ms/update_actor:0.33501532256501315 - timing_per_token_ms/values:0.07592782581397339 - perf/total_num_tokens:68419 - perf/time_per_step:57.29716528207064 - perf/throughput:1194.107939950907
Training Progress:  18%|█████████████████████████                                                                                                                 | 79/435 [1:18:10<5:38:51, 57.11s/it]
(TaskRunner pid=3165) test_gen_batch meta info: {'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True, 'global_steps': 80}
(TaskRunner pid=3165) validation generation end
(TaskRunner pid=3165) [prompt] system
(TaskRunner pid=3165) You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
(TaskRunner pid=3165) user
(TaskRunner pid=3165) Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".
(TaskRunner pid=3165) assistant
(TaskRunner pid=3165) 
(TaskRunner pid=3165) [response] First, we need to calculate the total number of eggs laid by the ducks per day. Janet's ducks lay 16 eggs per day.
(TaskRunner pid=3165) 
(TaskRunner pid=3165) Next, we need to calculate the total number of eggs eaten in a day. Janet eats 3 eggs for breakfast and 4 eggs baked for friends, so she eats 3 + 4 = 7 eggs per day.
(TaskRunner pid=3165) 
(TaskRunner pid=3165) Then, we subtract the number of eggs eaten from the total number of eggs to find the number of eggs sold at the farmers' market. This is 16 - 7 = 9 eggs sold per day.
(TaskRunner pid=3165) 
(TaskRunner pid=3165) Finally, we calculate the total amount of money Janet makes at the farmers' market per day. She sells each egg for $2, so she makes 9 * 2 = 18 dollars per day.
(TaskRunner pid=3165) 
(TaskRunner pid=3165) #### 18
(TaskRunner pid=3165) [ground_truth] 18
(TaskRunner pid=3165) [score] 1.0
(TaskRunner pid=3165) len reward_extra_infos_dict['reward']: 1319
(TaskRunner pid=3165) local_global_step_folder: checkpoints/verl_examples/gsm8k/global_step_80
(WorkerDict pid=3588) INFO:2025-08-22 23:08:37,898:[Rank 0] Saved model to /root/autodl-tmp/verl/checkpoints/verl_examples/gsm8k/global_step_80/actor/model_world_size_1_rank_0.pt
(WorkerDict pid=3588) INFO:2025-08-22 23:08:43,490:[Rank 0] Saved optim to /root/autodl-tmp/verl/checkpoints/verl_examples/gsm8k/global_step_80/actor/optim_world_size_1_rank_0.pt
(WorkerDict pid=3588) INFO:2025-08-22 23:08:43,491:[Rank 0] Saved extra_state to /root/autodl-tmp/verl/checkpoints/verl_examples/gsm8k/global_step_80/actor/extra_state_world_size_1_rank_0.pt
(WorkerDict pid=3588) INFO:2025-08-22 23:08:43,645:[Rank 0] Saved model config and tokenizer class to /root/autodl-tmp/verl/checkpoints/verl_examples/gsm8k/global_step_80/actor/huggingface
(WorkerDict pid=3588) INFO:2025-08-22 23:08:45,747:[Rank 0] Saved model to /root/autodl-tmp/verl/checkpoints/verl_examples/gsm8k/global_step_80/critic/model_world_size_1_rank_0.pt
(WorkerDict pid=3588) INFO:2025-08-22 23:08:52,202:[Rank 0] Saved optim to /root/autodl-tmp/verl/checkpoints/verl_examples/gsm8k/global_step_80/critic/optim_world_size_1_rank_0.pt
(WorkerDict pid=3588) INFO:2025-08-22 23:08:52,204:[Rank 0] Saved extra_state to /root/autodl-tmp/verl/checkpoints/verl_examples/gsm8k/global_step_80/critic/extra_state_world_size_1_rank_0.pt
(TaskRunner pid=3165) step:80 - global_seqlen/min:68508 - global_seqlen/max:68508 - global_seqlen/minmax_diff:0 - global_seqlen/balanced_min:68508 - global_seqlen/balanced_max:68508 - global_seqlen/mean:68508.0 - actor/entropy:0.12307024002075195 - critic/vf_loss:np.float64(0.006128273163994891) - critic/vf_clipfrac:np.float64(0.0) - critic/vpred_mean:np.float64(0.6050193128176033) - critic/grad_norm:np.float64(20.79897904396057) - perf/mfu/critic:np.float64(0.0) - critic/lr:np.float64(1e-05) - actor/pg_loss:np.float64(-0.0007830791869309905) - actor/pg_clipfrac:np.float64(0.004922861748127616) - actor/ppo_kl:np.float64(8.16520657735964e-05) - actor/pg_clipfrac_lower:np.float64(0.0) - actor/grad_norm:np.float64(3.7044053077697754) - perf/mfu/actor:np.float64(0.0) - perf/max_memory_allocated_gb:np.float64(20.764535427093506) - perf/max_memory_reserved_gb:np.float64(25.763671875) - perf/cpu_memory_used_gb:np.float64(39.88212585449219) - actor/lr:np.float64(1e-06) - val-core/openai/gsm8k/reward/mean@1:np.float64(0.4943138741470811) - training/global_step:80 - training/epoch:2 - critic/score/mean:0.57421875 - critic/score/max:1.0 - critic/score/min:0.0 - critic/rewards/mean:0.57421875 - critic/rewards/max:1.0 - critic/rewards/min:0.0 - critic/advantages/mean:5.8286926396533545e-09 - critic/advantages/max:2.7971644401550293 - critic/advantages/min:-2.321514368057251 - critic/returns/mean:0.5149930715560913 - critic/returns/max:1.0 - critic/returns/min:0.0 - critic/values/mean:0.66796875 - critic/values/max:1.2734375 - critic/values/min:-0.287109375 - critic/vf_explained_var:0.25293928384780884 - response_length/mean:163.6171875 - response_length/max:256.0 - response_length/min:51.0 - response_length/clip_ratio:0.0546875 - response_length_non_aborted/mean:163.6171875 - response_length_non_aborted/max:256.0 - response_length_non_aborted/min:51.0 - response_length_non_aborted/clip_ratio:0.0546875 - response/aborted_ratio:0.0 - prompt_length/mean:103.9921875 - prompt_length/max:174.0 - prompt_length/min:67.0 - prompt_length/clip_ratio:0.0 - timing_s/start_profile:0.00016638264060020447 - timing_s/generate_sequences:3.567976951599121 - timing_s/reshard:0.44454601407051086 - timing_s/generation_timing/max:3.567976951599121 - timing_s/generation_timing/min:3.567976951599121 - timing_s/generation_timing/topk_ratio:0.0 - timing_s/gen:4.408910249359906 - timing_s/reward:0.08692202251404524 - timing_s/old_log_prob:5.591106211766601 - timing_s/values:5.210249143652618 - timing_s/adv:0.0817561000585556 - timing_s/update_critic:18.716001658700407 - timing_s/update_actor:21.01028360798955 - timing_s/step:55.13615158479661 - timing_s/testing:22.982502876780927 - timing_s/save_checkpoint:17.463635701686144 - timing_s/stop_profile:0.0002209339290857315 - timing_per_token_ms/gen:0.10525975861528687 - timing_per_token_ms/adv:0.0011933803359980674 - timing_per_token_ms/update_critic:0.27319439567204423 - timing_per_token_ms/update_actor:0.30668365166096734 - timing_per_token_ms/values:0.07605314917458716 - perf/total_num_tokens:68508 - perf/time_per_step:55.13615158479661 - perf/throughput:1242.5241521370633
Training Progress:  18%|█████████████████████████▍                                                                                                                | 80/435 [1:19:46<6:46:13, 68.66s/it]
```

### 模型合并

输入指令：

```
python3 -m verl.model_merger merge \
--backend fsdp \
--local_dir checkpoints/verl_examples/gsm8k/global_step_430/actor \
--target_dir /root/autodl-tmp/final_model
```

合成的模型被保存到/autodl-tmp/final_model

### 使用训好的模型

编写脚本`test_my_model.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1. 指定你刚刚合并好的模型路径
# 确保这个路径就是 model_merger 命令中 --target_dir 的路径
model_path = "/root/autodl-tmp/final_model"

print(f"正在从 '{model_path}' 加载模型和分词器...")

# 2. 加载模型和分词器
# 我们使用 bfloat16 来加载，这在现代 GPU 上效率更高
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto" # 自动将模型加载到 GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("模型加载成功！")

# 3. 创建一个 pipeline 用于文本生成
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 4. 准备一个测试问题
# 这个问题和官方文档里的例子类似，但数字不同，看看模型能否举一反三
question = "一个面包师用面粉和糖制作蛋糕，比例是 9:4。如果他总共用了 169 公斤的原料，请问他用了多少公斤的面粉？ Let's think step by step and output the final answer after \"####\"."

# 5. 构建符合模型训练格式的 prompt
# 注意：这里的格式必须和你数据预处理时使用的格式完全一致！
messages = [
    {"role": "user", "content": question}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n--- Sending Prompt to Model ---")
print(prompt)
print("-----------------------------\n")

# 6. 进行推理
outputs = pipe(
    prompt,
    max_new_tokens=256, # 生成答案的最大长度
    do_sample=False,    # 使用确定性解码，而不是随机采样
    temperature=0.0,
    top_p=1.0,
)

# 7. 打印结果
print("--- Model Generated Response ---")
print(outputs[0]['generated_text'])
print("------------------------------\n")
```

示例回答：

```
(/root/autodl-tmp/conda_envs/verl_env) root@autodl-container-2f694b8462-c9a87a29:~/autodl-tmp/verl# python3 -m verl.model_merger merge --backend fsdp --local dir checkpoints/verl_examples/gsm8k/global_step_435/actor --target_dir /root/autodl-tmp/final_model
usage: __main__.py [-h] {merge,test} ...
__main__.py: error: unrecognized arguments: checkpoints/verl_examples/gsm8k/global_step_435/actor
(/root/autodl-tmp/conda_envs/verl_env) root@autodl-container-2f694b8462-c9a87a29:~/autodl-tmp/verl# python3 -m verl.model_merger merge --backend fsdp --local_dir checkpoints/verl_examples/gsm8k/global
_step_435/actor --target_dir /root/autodl-tmp/final_model
config: ModelMergerConfig(operation='merge', backend='fsdp', target_dir='/root/autodl-tmp/final_model', hf_upload_path=None, private=False, test_hf_dir=None, tie_word_embedding=False, trust_remote_code=False, is_value_model=False, local_dir='checkpoints/verl_examples/gsm8k/global_step_435/actor', hf_model_config_path='checkpoints/verl_examples/gsm8k/global_step_435/actor/huggingface', hf_upload=False, use_cpu_initialization=False)
Got device mesh [1], mesh_dim_names ('fsdp',)
Processing model shards with 1 (1,) in total
Loading 1 FSDP shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.95s/it]
Sliding Window Attention is enabled but not implemented for `eager`; unexpected results may be encountered.
Saving model to /root/autodl-tmp/final_model
Saving tokenizer to /root/autodl-tmp/final_model
(/root/autodl-tmp/conda_envs/verl_env) root@autodl-container-2f694b8462-c9a87a29:~/autodl-tmp/verl# python3 test_my_model.py
(/root/autodl-tmp/conda_envs/verl_env) root@autodl-container-2f694b8462-c9a87a29:~/autodl-tmp/verl# python3 test_my_model.py
正在从 '/root/autodl-tmp/final_model' 加载模型和分词器...
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
模型加载成功！
Device set to use cuda:0

--- Sending Prompt to Model ---
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
一个面包师用面粉和糖制作蛋糕，比例是 9:4。如果他总共用了 169 公斤的原料，请问他用了多少公斤的面粉？ Let's think step by step and output the final answer after "####".<|im_end|>
<|im_start|>assistant

-----------------------------

/root/autodl-tmp/conda_envs/verl_env/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:631: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
/root/autodl-tmp/conda_envs/verl_env/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:653: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `20` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
--- Model Generated Response ---
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
一个面包师用面粉和糖制作蛋糕，比例是 9:4。如果他总共用了 169 公斤的原料，请问他用了多少公斤的面粉？ Let's think step by step and output the final answer after "####".<|im_end|>
<|im_start|>assistant
首先，设面粉的比例为 9x 公斤，糖的比例为 4x 公斤，根据题目中的比例关系，有 \(9x + 4x = 169\)。

解这个等式得到：
\[13x = 169\]
\[x = \frac{169}{13}\]
\[x = 13\]

所以，面粉的比例是 9x = 9 * 13 = 117 公斤。

因此，用了 117 公斤的面粉。

#### 117
#### 117
------------------------------
```

## 尝试wandb

尝试了一下使用wandb：

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=[console,wandb] \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log
```

wandb的使用：在官网上注册一个账号，获取API key，然后在本地：

```
wandb login
```

按照引导输入API key即可，就算登陆成功，可以执行上面的训练指令了

碰到 超时问题：

```
(TaskRunner pid=6534) wandb: Network error (ConnectTimeout), entering retry loop.
```

尝试使用配置国内代理解决：（发现没用，这个可能是gemini幻觉）

```
export WANDB_RELAY_URL=https://api.wandb.cn
```

发现能ping通但是延时比较长：

```
/root/autodl-tmp/conda_envs/verl_env) root@autodl-container-2f694b8462-c9a87a29:~/autodl-tmp/verl# ping www.wandb.ai
PING www.wandb.ai (151.101.65.195) 56(84) bytes of data.
64 bytes from 151.101.65.195 (151.101.65.195): icmp_seq=1 ttl=51 time=217 ms
64 bytes from 151.101.65.195 (151.101.65.195): icmp_seq=2 ttl=51 time=201 ms
64 bytes from 151.101.65.195 (151.101.65.195): icmp_seq=3 ttl=51 time=197 ms
64 bytes from 151.101.65.195 (151.101.65.195): icmp_seq=4 ttl=51 time=196 ms
64 bytes from 151.101.65.195 (151.101.65.195): icmp_seq=5 ttl=51 time=200 ms
64 bytes from 151.101.65.195 (151.101.65.195): icmp_seq=6 ttl=51 time=208 ms
```

按照gemini建议改成

```
export WANDB_INIT_TIMEOUT=300
```

也没用，gemini给出的回答是连接实在太差，最后尝试本地的方案：

设置环境变量：

````bash
export WANDB_MODE=offline
````

然后正常运行那个PPO启动的指令

这里训到中间直接Ctrl+C了，尝试将对应wandb信息文件同步：

```bash
wandb sync wandb/offline-run-20250823_165526-l7dl7zrs
```

超时失败了：

```
wandb: Network error (ConnectTimeout), entering retry loop.
```

将整个`offline-run-20250823_165526-l7dl7zrs`文件夹下载到本地主机上，运行：

```
(WANDB_ENV) PS K:\大四\各种中间文件暂存\l7dl7zrs> wandb sync offline-run-20250823_165526-l7dl7zrs
Find logs at: C:\Users\18326\AppData\Local\Temp\debug-cli.18326.log
Syncing: https://wandb.ai/yangzhou66666-nanjing-university/verl_examples/runs/l7dl7zrs ... done.
```

成功了！在wandb官网上可以访问类似图像：

![](./assets/wandb图像.png)

## 尝试从最新的检查点继续训练

默认配置就是会从最新的检查点开始，因此只要直接重复执行之前的指令即可

![](./assets/从最新的检查点开始.png)

直接从380开始了，成功



## 尝试GRPO和奖励模型

从hugging face上下载奖励模型`Skywork-Reward-V2-Qwen3-0.6B`，放在`/autodl-tmp`下

gemini给的指令，在ppo demo基础上修改了几个点

**（血泪教训！！！一定要把下面指令的`＃`去了！！！否则`#`后面的指令命令行没收到！！！最开始直接复制上去了跑的很正常，最后发现没设成功Reward Model）**

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.rollout.name=vllm \
 # --- grpo修改的指令: rollout.n=4, 每次采样四个回答 ---
 actor_rollout_ref.rollout.n=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 # --- grpo修改的指令: 使用grpo的adv_estimator ---
 algorithm.adv_estimator=grpo \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=[console,wandb] \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 \
 # --- 启用奖励模型的新增/修改参数 ---
 reward_model.enable=True \
 reward_model.model.path=/root/autodl-tmp/Skywork-Reward-V2-Qwen3-0.6B \
 2>&1 | tee verl_grpo_rm_demo.log
```

出现了以下报错：

`ValueError: [reward_model] Please set at least one of 'reward_model.micro_batch_size' or 'reward_model.micro_batch_size_per_gpu'`

按照他的意思来修改：

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.n=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.adv_estimator=grpo \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=[console,wandb] \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 \
 reward_model.enable=True \
 reward_model.model.path=/root/autodl-tmp/Skywork-Reward-V2-Qwen3-0.6B \
 # ---新增的关键参数---
 reward_model.micro_batch_size_per_gpu=8 \
 2>&1 | tee verl_grpo_rm_demo.log
```

出现以下报错：

```
error executing job with overrides: ['data.train_files=/root/data/gsm8k/train.parquet', 'data.val_files=/root/data/gsm8k/test.parquet', 'data.train_batch_size=256', 'data.max_prompt_length=512', 'data.max_response_length=256', 'actor_rollout_ref.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.actor.ppo_mini_batch_size=64', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.n=4', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8', 'actor_rollout_ref.rollout.tensor_model_parallel_size=1', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.4', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4', 'critic.optim.lr=1e-5', 'critic.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct', 'critic.ppo_micro_batch_size_per_gpu=4', 'algorithm.adv_estimator=grpo', 'algorithm.kl_ctrl.kl_coef=0.001', 'trainer.logger=[console,wandb]', 'trainer.val_before_train=False', 'trainer.n_gpus_per_node=1', 'trainer.nnodes=1', 'trainer.save_freq=10', 'trainer.test_freq=10', 'trainer.total_epochs=15', 'reward_model.enable=True', 'reward_model.model.path=/root/autodl-tmp/Skywork-Reward-V2-Qwen3-0.6B', 'reward_model.micro_batch_size_per_gpu=8']
Traceback (most recent call last):
  File "/root/autodl-tmp/verl/verl/trainer/main_ppo.py", line 40, in main
    run_ppo(config)
  File "/root/autodl-tmp/verl/verl/trainer/main_ppo.py", line 83, in run_ppo
    ray.get(runner.run.remote(config))
  File "/root/autodl-tmp/conda_envs/verl_env/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/root/autodl-tmp/conda_envs/verl_env/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
  File "/root/autodl-tmp/conda_envs/verl_env/lib/python3.10/site-packages/ray/_private/worker.py", line 2858, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/root/autodl-tmp/conda_envs/verl_env/lib/python3.10/site-packages/ray/_private/worker.py", line 958, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(KeyError): ray::TaskRunner.run() (pid=16932, ip=172.17.0.11, actor_id=0078c64373fdca606e64f38b01000000, repr=<main_ppo.TaskRunner object at 0x7fbb55427c40>)
  File "/root/autodl-tmp/verl/verl/trainer/main_ppo.py", line 285, in run
    trainer.fit()
  File "/root/autodl-tmp/verl/verl/trainer/ppo/ray_trainer.py", line 1174, in fit
    reward_tensor = self.rm_wg.compute_rm_score(batch)
  File "/root/autodl-tmp/verl/verl/single_controller/ray/base.py", line 48, in __call__
    output = ray.get(output)
ray.exceptions.RayTaskError(KeyError): ray::WorkerDict.rm_compute_rm_score() (pid=17374, ip=172.17.0.11, actor_id=47899a62d924bdeaa331edee01000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7fce676ecb50>)
  File "/root/autodl-tmp/verl/verl/single_controller/ray/base.py", line 701, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
  File "/root/autodl-tmp/verl/verl/single_controller/base/decorator.py", line 430, in inner
    return func(*args, **kwargs)
  File "/root/autodl-tmp/verl/verl/workers/fsdp_workers.py", line 1670, in compute_rm_score
    rm_data = self._switch_chat_template(data)
  File "/root/autodl-tmp/verl/verl/workers/fsdp_workers.py", line 1605, in _switch_chat_template
    if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
KeyError: 'raw_prompt'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```

gemini说是两种模型的格式不兼容，给出了以下修改：

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 # --- 新增的关键参数 ---
 data.return_raw_chat=True \
 actor_rollout_ref.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.n=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=/root/autodl-tmp/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.adv_estimator=grpo \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=[console,wandb] \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 \
 reward_model.enable=True \
 reward_model.model.path=/root/autodl-tmp/Skywork-Reward-V2-Qwen3-0.6B \
 reward_model.micro_batch_size_per_gpu=8 \
 2>&1 | tee verl_grpo_rm_demo.log
```

这次成功了！！有以下关键的log证明：

- GRPO算法已启用：

```
(TaskRunner pid=20227)   'adv_estimator': 'grpo',
(TaskRunner pid=20227)   'use_kl_loss': True,
(TaskRunner pid=20227)   'n': 4, 
```

- 奖励模型已启用：

```
(TaskRunner pid=20227)  'reward_model': {'enable': True,
...
(TaskRunner pid=20227)                   'path': '/root/autodl-tmp/Skywork-Reward-V2-Qwen3-0.6B',
...
(TaskRunner pid=20227)                   'micro_batch_size_per_gpu': 8,
```

- Critic被自动禁用：

```
(TaskRunner pid=20227) /root/autodl-tmp/verl/verl/trainer/main_ppo.py:268: UserWarning: Disabled critic as algorithm.adv_estimator != gae. ...
```

- min reward和max reward不是1和0，说明用的是奖励模型而非之前的规则打分法

```
critic/rewards/max:10.6875 - critic/rewards/min:-11.375
```

