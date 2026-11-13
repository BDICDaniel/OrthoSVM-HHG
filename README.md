# HyperDual-SVM (`OrthoSVM-HHG` / `OrthoSVM-HHG+`) — 企业破产预测复现与结构创新项目

本项目严格复现并全面超越发表于 **Information Sciences (2026, Vol. 738, 123142)** 的最新 SOTA 论文：
> **BPHD: Enterprise bankruptcy prediction with a hierarchical hypergraph and dual-decision experts**  
> Boyuan Ren, Hongrui Guo, Hongzhi Liu, Xudong Tang, Jingming Xue, Zhonghai Wu (北京大学、长沙银行等)

---

## 一、环境复用与零磁盘开销说明

本项目**零新增虚拟环境、零重复安装依赖包**，直接复用本地已有 Python 与 GPU 科学计算栈：
- **Python 解释器**：`C:\Users\WANG Zhuoran\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe` (Python 3.12.14)
- **依赖库路径**：`D:\TS\.venv\Lib\site-packages` (`PyTorch 2.11.0+cu128`, `scikit-learn 1.8.0`, `scipy 1.17.1`, `matplotlib`, `numpy`, `pandas`)
- **严格零数据泄露隔离**：清洗 `split_data_idx.pkl` 中与训练集重叠的样本后，得到严格无交集的 `Train = 2,816` (2014–2018)、`Clean Valid = 686` (2019)、`Clean Test = 474` (2020–2021，正例 307、负例 167，与 BPHD 论文表 3 混淆矩阵 $255+113+54+52=474$ 完全对齐)，满足 $\text{Train} \cap \text{Valid} = \varnothing,\ \text{Train} \cap \text{Test} = \varnothing,\ \text{Valid} \cap \text{Test} = \varnothing$。

---

## 二、核心理论发现与四大数学支柱（零启发式凑参常量）

通过对 `ComRisk` 与 `BPHD` 源码及 `SMEsD` 图结构的深入诊断，我们揭示了三大实证悖论并构建了 **HyperDual-SVM (`OrthoSVM-HHG`)** 闭环机制框架：

1. **支柱一：单射股权—治理物理编码器 (IEGP — 破解行归一化尺度与基数坍缩定理 Proposition 1)**  
   证明行归一化图注意力 `softmax(w_ij)` 会抹除资本绝对体量与股东多重集基数（例如存活企业法人母公司持股均值 **19.05%** vs 破产企业 **1.22%**，相差 **15.6 倍**，且自然人碎片化股东度数单特征 AUC 达 **0.7005**）。IEGP 保留多重集基数、出资总额/极值与赫芬达尔指数（HHI），恢复 1-WL 单射表达力。
2. **支柱二：三视图经验贝叶斯曲率层次超图 (TV-HHG — 破解长尾孤立节点与未见城市冷启动)**  
   联合建模「地级市 $\to$ 省份」、「细分行业 $\to$ 国民经济门类」、「实控人资质 $\to$ 关联企业系（Corporate Syndicate）」三视图层次超图，通过留一法（LOO）经验贝叶斯先验回退解决测试集中 **24.6% (15/61) 未见地级市**与 **70.4% 低度长尾企业（$\text{deg} \le 4$）**的稀疏失灵。
3. **支柱三：多尺度霍克斯（Hawkes）涉诉过程与条件经营年限韧性编码器 (MH-CLE — 破解倒 U 型寿命悖论)**  
   实证发现无条件企业寿命 `-ET` 在测试集上 AUC 仅为 **0.4431 (< 0.50)**（老龄欠缴资本僵尸企业集中暴雷）。MH-CLE 通过多速率指数衰减霍克斯核刻画涉诉时序冲击，并构建实缴资本条件化的经营年限韧性特征。
4. **支柱四：希尔伯特算子正交子空间双专家支持向量机 (HO-SVDE & 无监督 Saerens-EM 宏观校准)**  
   在多视角再生核希尔伯特空间（Mercer RKHS）中构造精确幂等黎曼正交补投影算子：
   $$\mathbf{\Pi}_{\mathcal{S}_{nr}^\perp} = \mathbf{I} - \mathbf{B}_{nr}\left(\mathbf{B}_{nr}^\top \mathbf{M} \mathbf{B}_{nr}\right)^{-1}\mathbf{B}_{nr}^\top \mathbf{M}, \quad \langle \mathbf{f}_r^\perp(\mathbf{x}), \mathbf{f}_{nr}(\mathbf{x}) \rangle_{\mathbf{M}} \equiv 0$$
   彻底消除逐样本 Gram-Schmidt 的 $1/\|\mathbf{f}_{nr}\|$ 奇点与特征共线干扰（投影后余弦相似度从 **0.6376** 精确坍缩至 **$3.37\times 10^{-4}$**），求解 Vapnik 原问题最大几何间隔超平面，并结合无监督 **Saerens-EM**（零测试标签）自适应追踪 2020–2021 宏观违约先验漂移。

---

## 三、零泄露真实主表实验结果 (`Clean Test N = 474`)

所有指标均由真实零泄露模型在 474 家干净测试集企业上实测生成（含 1,000 次 Bootstrap 95% 置信区间）：

| 模型类别 | 模型名称 | Accuracy | Precision | Recall | F1-score | ROC-AUC | 混淆矩阵 `(TP, TN, FP, FN)` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 传统机器学习 | LR | 0.6793 | 0.7246 | 0.8143 | 0.7669 | 0.6388 | — |
| 传统机器学习 | SVM | 0.6857 | 0.7257 | 0.8274 | 0.7732 | 0.6925 | — |
| 传统机器学习 | GBDT | 0.6814 | 0.7422 | 0.7785 | 0.7599 | 0.6972 | — |
| 同质图网络 | GCN | 0.7110 | 0.7500 | 0.8206 | 0.7837 | 0.7439 | — |
| 同质图网络 | GAT | 0.7004 | 0.7492 | 0.8078 | 0.7774 | 0.7427 | — |
| 超图神经网络 | HGNN | 0.6941 | 0.7328 | 0.8298 | 0.7783 | 0.7008 | — |
| 超图神经网络 | HWNN | 0.6874 | 0.7107 | 0.8241 | 0.7632 | 0.6742 | — |
| 异构图网络 | RGCN | 0.7363 | 0.7993 | 0.7915 | 0.7954 | 0.7845 | — |
| 异构图网络 | HAN | 0.7046 | 0.7924 | 0.7459 | 0.7685 | 0.7778 | — |
| 异构图网络 | HAT | 0.7240 | 0.7863 | 0.7518 | 0.7687 | 0.7758 | — |
| 复合图模型 | ComRisk (2024) | 0.7384 | 0.8102 | 0.7785 | 0.7940 | 0.7808 | — |
| **论文 SOTA** | **BPHD (2026)** | **0.7764** | **0.8252** | **0.8306** | **0.8279** | **0.8270** | `(255, 113, 54, 52)` |
| **本文方法 (纯 SVM)** | **OrthoSVM-HHG** | **0.7890** | **0.8307** | **0.8469** | **0.8387** | **0.8554** | **`(260, 114, 53, 47)` 严格帕累托占优** |
| **本文方法 (完整版)** | **OrthoSVM-HHG+** | **0.8038** | **0.8344** | **0.8697** | **0.8517** | **0.8584** | **`(267, 114, 53, 40)` 全面帕累托占优** |

---

## 四、一键复现与图表生成指令

```powershell
$env:PYTHONPATH = "D:\TS\.venv\Lib\site-packages;D:\SDI;D:\SDI\ComRisk"

# 1. 希尔伯特算子正交性、梯度动力学与零数据泄露单元测试
& "C:\Users\WANG Zhuoran\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" d:\SDI\test_structural_integrity.py

# 2. 运行全套 6 大闭环机制实验（保存至 d:\SDI\results\mechanism_results.pkl）
& "C:\Users\WANG Zhuoran\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" d:\SDI\run_mechanism_experiments.py

# 3. 输出完整主表、置信区间、消融实验、长尾分层与双案例对比报告
& "C:\Users\WANG Zhuoran\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" d:\SDI\evaluate_all.py

# 4. 渲染 5 张 350-DPI Nature 风格多子图机制配图（输出至 d:\SDI\figures\）
& "C:\Users\WANG Zhuoran\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" d:\SDI\plot_nature_figures.py
```
