import numpy as np
from scipy.stats import bootstrap

# 假设的分数数据
your_model_scores = np.random.normal(29.55, 0.5, 1000)  # 您的模型分数，均值为54.43，标准差为0.5，样本量为1000
mmtlb_scores = np.random.normal(25.79, 0.5, 1000)      # MMTLB模型分数，均值为53.06，标准差为0.5，样本量为1000

# 计算原始差值
original_diff = np.mean(your_model_scores) - np.mean(mmtlb_scores)

# 引导抽样配置
boot_results = bootstrap((your_model_scores, mmtlb_scores), 
                         statistic=lambda data1, data2: np.mean(data1) - np.mean(data2), 
                         method='percentile',
                         random_state=0,
                         confidence_level=0.95,
                         n_resamples=10000)

# 计算p值：检查在引导样本中，比原始差值更极端的比例
p_value = np.mean(np.abs(boot_results.bootstrap_distribution) >= np.abs(original_diff))

print(boot_results.confidence_interval)
print(original_diff)
print(p_value)
