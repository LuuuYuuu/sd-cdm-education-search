# SD-CDM: 教育场景下的认知诊断模型（Search & Recommendation Enhanced）

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**项目地址**：本仓库为作者研究生期间的科研项目，论文《SD-CDM: A Structured Dynamic Cognitive Diagnosis Model for Intelligent Tutoring Systems》正在 **Expert Systems with Applications** 期刊 **under review**。

## 🎯 项目背景

在智慧教育（Intelligent Tutoring Systems, ITS）领域，**认知诊断（Cognitive Diagnosis）** 是实现个性化学习的核心技术。它能够精确评估学生对每个知识点的掌握程度（知识状态），从而为学生**精准推荐**适合的学习资源、练习题目或学习路径。

本项目提出 **SD-CDM（Structured Dynamic Cognitive Diagnosis Model）**，创新性地将结构化知识建模与动态学生行为建模相结合，使模型在**教育搜索与推荐场景**中表现更优：
- **搜索场景**：快速召回学生当前最需要的知识点或题目（类似搜索引擎的召回+排序）。
- **推荐场景**：基于认知诊断结果，个性化推荐学习材料，提升学习效率和准确率。

该模型已在真实教育数据集上验证，显著优于传统CDM（如IRT、DINA、NCDM等），为教育AI产品（如智能学习机、作业平台、知识图谱推荐系统）提供了可落地的算法支撑。

**核心应用场景**：
- 在线教育平台的个性化题目推荐
- 学生知识弱点诊断与针对性搜索
- 智能教学系统中内容画像构建与用户行为分析


## 📦 安装依赖

```bash
# 1. 克隆仓库
git clone https://github.com/你的用户名/sd-cdm-education-search.git
cd sd-cdm-education-search

# 2. 创建虚拟环境（推荐）
conda create -n sdcmd python=3.10
conda activate sdcmd

# 3. 安装依赖
pip install -r requirements.txt
