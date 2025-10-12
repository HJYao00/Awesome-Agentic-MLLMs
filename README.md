# 🤖Awesome-Agentic-MLLMs

[![Maintenance](https://img.shields.io/badge/Maintenance-FF9800?style=plastic&logo=maintenance&logoColor=white)]()
[![Discussion](https://img.shields.io/badge/Discussion-Open-brightgreen?style=plastic&logo=discussion&logoColor=white)]()
[![Contribution Welcome](https://img.shields.io/badge/Contribution-Welcome-blue?style=plastic&logo=discussion&logoColor=white)]()


👏 Welcome to the **Awesome-Agentic-MLLM** repository!
This curated collection features influential papers, codebases, datasets, benchmarks, and resources dedicated to exploring the emerging field of agentic capabilities in Multimodal Large Language Models.


⭐ Feel free to **star** and **fork** this repository to stay updated with the latest advancements and contribute to the growing community.

---
## 🔔 News
- [x] **`Oct xx, 2025.`** This repository curates and maintains an updated list of papers on Awesome-Agentic-MLLM. Contributions and suggestions are warmly welcome!

## 📒 Table of Contents

- [Awesome-Agentic-MLLM](#awesome-agentic-mllm)
  - [📒 Table of Contents](#-table-of-contents)
  - [📄 Paper List](#-paper-list)
    - [Foundational MLLMs](#foundational-mllms)
      - [Dense MLLMs](#dense-mllms)
      - [MoE MLLMs](#moe-mllms)
    - [Agentic Internal Intelligence](#agentic-internal-intelligence)
      - [Agentic Reasoning](#agentic-reasoning)
      - [Agentic Reflection](#agentic-reflection)
      - [Agentic Memory](#agentic-memory)
    - [Agentic External Tool Invocation](#agentic-external-tool-invocation)
      - [Agentic Search](#agentic-search)
      - [Agentic Code](#agentic-code)
      - [Agentic Data Process](#agentic-data-process)
    - [Agentic Training Framework](#agentic-training-framework)
    <!--
    - [Agentic Training Dataset](#agentic-training-dataset)
    - [Agentic Evaluation Benchmark](#agentic-evaluation-benchmark)
    -->
---


## 📄 Paper List

<!--
## Survey
| Date |  Title | Paper | Github |
|:-:|:-:|:-|:-:|
| 2308 | A Survey on Large Language Model based Autonomous Agents | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2308.11432) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/Paitesanshi/LLM-Agent-Survey) |
| 2402 | Large Multimodal Agents: A Survey | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2402.15116) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/jun0wanan/awesome-large-multimodal-agents) |

* [2308] A Survey on Large Language Model based Autonomous Agents [[Paper📑]](https://arxiv.org/abs/2308.11432) [[Project📚]](https://github.com/DavidZWZ/Awesome-Deep-Research)
* [2402] Large Multimodal Agents: A Survey [[Paper📑]](https://arxiv.org/abs/2402.15116)
* [2501] Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG [[Paper📑]](https://arxiv.org/abs/2501.09136)
* [2502] Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation [[Paper📑]](https://arxiv.org/abs/2502.08826)
* [2503] Agentic Large Language Models, a survey [[Paper📑]](https://arxiv.org/abs/2503.23037)
* [2504] From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs [[Paper📑]](https://arxiv.org/pdf/2504.15965)
* [2505] Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions [[Paper📑]](https://arxiv.org/abs/2505.00675)
* [2506] From Web Search towards Agentic Deep Research: Incentivizing Search with Reasoning Agents [[Paper📑]](https://arxiv.org/abs/2506.18959) [[Project📚]](https://github.com/DavidZWZ/Awesome-Deep-Research)
* [2507] Evaluation and Benchmarking of LLM Agents: A Survey [[Paper📑]](https://arxiv.org/pdf/2507.21504)
* [2508] AI Agentic Programming: A Survey of Techniques, Challenges, and Opportunities [[Paper📑]](https://arxiv.org/pdf/2508.11126)
* [2508] Towards Embodied Agentic AI: Review and Classification of LLM- and VLM-Driven Robot Autonomy and Interaction [[Paper📑]](https://arxiv.org/pdf/2508.05294)
* [2508] A Survey on Agent Workflow – Status and Future [[Paper📑]](https://arxiv.org/pdf/2508.01186)

-->

---


### Foundational MLLMs

#### Dense MLLMs
| Date |  Title | Paper | Code |
|:-:|:-:|:-|:-:|
| 2502 | Qwen2.5-VL Technical Report | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2502.13923) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5) |
| 2502 | SmolVLM2: Bringing Video Understanding to Every Device | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://huggingface.co/blog/smolvlm2) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) |
| 2506 | MiMo-VL Technical Report | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2506.03569) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/XiaomiMiMo/MiMo-VL) |
| 2507 | Kwai Keye-VL Technical Report | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2507.01949) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/Kwai-Keye/Keye) |
| 2509 | SAIL-VL2 Technical Report | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2509.14033) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/BytedanceDouyinContent/SAIL-VL2) |
| 2509 | LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://www.arxiv.org/abs/2509.23661) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5) |
| 2509 | MiniCPM-V 4.5 technical report | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/MiniCPM_V_4_5_Technical_Report.pdf) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/OpenBMB/MiniCPM-V) |

---

#### MoE MLLMs
| Date |  Title | Paper | Code |
|:-:|:-:|:-|:-:|
| 2509 | Qwen3-VL: Sharper Vision, Deeper Thought, Broader Action | [![Paper](https://img.shields.io/badge/blog-A42C25?style=for-the-badge)](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef&from=research.latest-advancements-list) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/QwenLM/Qwen3-VL) |
| 2409 | MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2409.20566) | - |
| 2412 | DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding | [![Paper](https://img.shields.io/badge/blog-A42C25?style=for-the-badge)](https://arxiv.org/abs/2412.10302) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/deepseek-ai/DeepSeek-VL2) |
| 2503 | Kimi-VL Technical Report | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2504.07491) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/MoonshotAI/Kimi-VL) |
| 2506 | ERNIE 4.5 Technical Report | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://yiyan.baidu.com/blog/publication/ERNIE_Technical_Report.pdf) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/PaddlePaddle/ERNIE) |
| 2507 | Seed1.5-VL Technical Report | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.07062) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://www.volcengine.com/) |
| 2507 | GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2507.01006) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/zai-org/GLM-V) |
| 2507 | Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2507.19427) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/stepfun-ai/Step3) |
| 2508 | InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2508.18265) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/OpenGVLab/InternVL) |

---


### Agentic Internal Intelligence

#### Agentic Reasoning
| Date |  Title | Paper | Github |
|:-:|:-:|:-|:-:|
| 2410 | Improve Vision Language Model Chain-of-thought Reasoning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/pdf/2410.16198) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/RifleZhang/LLaVA-Reasoner-DPO?tab=readme-ov-file) |
| 2411 | LLaVA-CoT: Let Vision Language Models Reason Step-by-Step | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2411.10440) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/PKU-YuanGroup/LLaVA-CoT) |
| 2412 | Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2412.18319) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/HJYao00/Mulberry) |
| 2503 | Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2503.06749) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/Osilly/Vision-R1) |
| 2503 | R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2503.12937) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/jingyi0000/R1-VL) |
| 2503 | MM-Eureka: Exploring the Frontiers of Multimodal Reasoning with Rule-based Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2503.07365) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/ModalMinds/MM-EUREKA) |
| 2503 | Video-R1: Reinforcing Video Reasoning in MLLMs | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2503.21776) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/tulerfeng/Video-R1) |
| 2504 | SoTA with Less: MCTS-Guided Sample Selection for Data-Efficient Visual Reasoning Self-Improvement | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2504.07934) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/si0wang/ThinkLite-VL) |
| 2504 | NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2504.13055) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/NUS-TRAIL/NoisyRollout) |
| 2504 | Skywork R1V2: Multimodal Hybrid Reinforcement Learning for Reasoning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2504.16656) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://huggingface.co/Skywork/Skywork-R1V2-38B) |
| 2504 | VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2504.07615) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/om-ai-lab/VLM-R1) |
| 2505 | SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.17018) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/kxfan2002/SophiaVL-R1) |
| 2505 | R1-ShareVL: Incentivizing Reasoning Capability of Multimodal Large Language Models via Share-GRPO | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.16673) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/HJYao00/R1-ShareVL) |
| 2505 | EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.04623) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/HarryHsing/EchoInk) |
| 2505 | Infi-MMR: Curriculum-based Unlocking Multimodal Reasoning via Phased Reinforcement Learning in Multimodal Small Language Models | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.23091) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://huggingface.co/InfiX-ai/Infi-MMR-3B) |
| 2506 | GRPO-CARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/pdf/2506.16141) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/TencentARC/GRPO-CARE) |
| 2506 | WeThink: Toward General-purpose Vision-Language Reasoning via Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2506.07905) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/yangjie-cv/WeThink) |
| 2507 | Scaling RL to Long Videos | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2507.07966) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B) |
| 2507 | VL-Cogito: Progressive Curriculum Reinforcement Learning for Advanced Multimodal Reasoning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2507.22607) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/alibaba-damo-academy/VL-Cogito) |



---

#### Agentic Reflection

| Date |  Title | Paper | Code |
|:-:|:-:|:-|:-:|
| 2410 | ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2410.17657) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/BlueZeros/ReflecTool) |
| 2411 | Self-Corrected Multimodal Large Language Model for Robot Manipulation and Reflection | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://openreview.net/forum?id=TLWbNfbkxj) | - |
| 2411 | Vision-Language Models Can Self-Improve Reasoning via Reflection | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2411.00855) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/njucckevin/MM-Self-Improve) |
| 2412 | Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/pdf/2412.18319) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/HJYao00/Mulberry) |
| 2503 | V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2503.12077) | - |
| 2504 | MASR: Self-Reflective Reasoning through Multimodal Hierarchical Attention Focusing for Agent-based Video Understanding | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2504.17213) | - |
| 2504 | VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2504.08837) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://tiger-ai-lab.github.io/VL-Rethinker) |
| 2505 | Training-Free Reasoning and Reflection in MLLMs | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.16151) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://iip.whu.edu.cn/frank/index.html) |
| 2506 | SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2506.01713) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://srpo.pages.dev/) |
| 2507 | Look-Back: Implicit Visual Re-focusing in MLLM Reasoning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/pdf/2507.03019) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/PKU-YuanGroup/Look-Back) |


---

#### Agentic Memory

| Date |  Title | Paper | Code |
|:-:|:-:|:-|:-:|
| 2305 | MemoryBank: Enhancing Large Language Models with Long-Term Memory | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2305.10250) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/zhongwanjun/MemoryBank-SiliconFriend) |
| 2307 | MovieChat | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2307.16449v4) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/rese1f/MovieChat) |
| 2312 | Empowering Working Memory for Large Language Model Agents | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/pdf/2312.17259) | - |
| 2402 | LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2402.13753) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/microsoft/LongRoPE) |
| 2502 | A-Mem: Agentic Memory for LLM Agents | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/pdf/2502.12110) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/WujiangXu/A-mem) |
| 2503 | In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/pdf/2503.08026) | - |
| 2504 | Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2504.19413) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://mem0.ai/research) |
| 2506 | A Walk to Remember: Mllm Memory-Driven Visual Navigation | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://ieeexplore.ieee.org/abstract/document/11078086) | - |
| 2506 | MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2506.15841) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/MIT-MI/MEM1) |
| 2507 | MemOS: A Memory OS for AI System | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2507.03724) | - |
| 2507 | MIRIX: Multi-Agent Memory System for LLM-Based Agents | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/pdf/2507.07957) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://mirix.io/) |
| 2508 | Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2508.19828) | - |
| 2508 | Intrinsic Memory Agents: Heterogeneous Multi-Agent LLM Systems through Structured Contextual Memory | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2508.08997) | - |
| 2508 | MMS: Multiple Memory Systems for Enhancing the Long-term Memory of Agent | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2508.15294) | - |

----

### Agentic External Tool Invocation

#### Agentic Search 

| Date |  Title | Paper | Code |
|:-:|:-:|:-|:-:|
| 2502 | Open AI Deep Research: Introducing deep research | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.22019) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://openai.com/index/introducing-deep-research/) |
| 2505 | VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.22019) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/Alibaba-NLP/VRAG) |
| 2505 | Visual Agentic Reinforcement Fine-Tuning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.14246) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/Liuziyu77/Visual-RFT/tree/main/Visual-ARFT) |
| 2506 | MMSearch-R1: Incentivizing LMMs to Search | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2506.20670) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1) |
| 2508 | Patho-AgenticRAG: Towards Multimodal Agentic Retrieval-Augmented Generation for Pathology VLMs via Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2508.02258) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/Wenchuan-Zhang/Patho-AgenticRAG) |
| 2508 | M2IO-R1: An Efficient RL-Enhanced Reasoning Framework for Multimodal Retrieval Augmented Multimodal Generation | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2508.06328) | - |
| 2508 | WebWatcher: Breaking New Frontier of Vision-Language Deep Research Agent | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2508.05748) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/Alibaba-NLP/WebAgent) |

<!--
* [2505] VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning [[Paper📑]](https://arxiv.org/abs/2505.22019) [[Code🔧]](https://github.com/Alibaba-NLP/VRAG)
* [2505] Visual Agentic Reinforcement Fine-Tuning [[Paper📑]](https://arxiv.org/abs/2505.14246) [[Code🔧]](https://github.com/Liuziyu77/Visual-RFT/tree/main/Visual-ARFT)
* [2506] MMSearch-R1: Incentivizing LMMs to Search [[Paper📑]](https://arxiv.org/abs/2506.20670) [[Code🔧]](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
* [2508] WebWatcher: Breaking New Frontier of Vision-Language Deep Research Agent [[Paper📑]](https://arxiv.org/abs/2508.05748) [[Code🔧]](https://github.com/Alibaba-NLP/WebAgent)
* [2508] MiroMind Open Deep Research [[Blog📝]](https://miromind.ai/blog/miromind-open-deep-research)

* [2502] Introducing deep research [[Blog📝]](https://openai.com/index/introducing-deep-research/)
* [2507] WebSailor: Navigating Super-human Reasoning for Web Agent [[Paper📑]](https://arxiv.org/abs/2507.02592) [[Code🔧]](https://github.com/Alibaba-NLP/WebAgent)
* [2508] Introducing gpt-oss [[Blog📝]](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf) [[Code🔧]](https://github.com/openai/gpt-oss)
* [2508] Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL [[Paper📑]](https://arxiv.org/abs/2508.07976) [[Code🔧]](https://github.com/inclusionAI/ASearcher?tab=readme-ov-file)

---

## Search Agent 

* [2112] WebGPT: WebGPT: Browser-assisted question-answering with human feedback [[Paper📑]](https://arxiv.org/pdf/2112.09332)
* [2207] WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents [[Paper📑]](https://arxiv.org/pdf/2207.01206) [[Code🔧]](https://github.com/princeton-nlp/WebShop)
* [2306] MIND2WEB: Towards a Generalist Agent for the Web [[Paper📑]](https://arxiv.org/abs/2306.06070)
* [2310] WebWISE: Web Interface Control and Sequential Exploration with Large Language Models [[Paper📑]](https://arxiv.org/pdf/2310.16042)
* [2401] VisualWebArena: Evaluating Multimodal Agents on Realistic Visually Grounded Web Tasks [[Paper📑]](https://arxiv.org/abs/2401.13649) [[Code🔧]](https://jykoh.com/vwa)
* [2401] WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models [[Paper📑]](https://arxiv.org/abs/2401.13919) [[Code🔧]](https://github.com/MinorJerry/WebVoyager)
* [2401] GPT-4V(ision) is a Generalist Web Agent, if Grounded [[Paper📑]](https://arxiv.org/pdf/2401.01614) [[Code🔧]](https://github.com/OSU-NLP-Group/SeeAct)
* [2408] PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection [[Paper📑]](https://arxiv.org/abs/2408.10738)
* [2410] OpenWebVoyager: Building Multimodal Web Agents via Iterative Real-World Exploration, Feedback and Optimization [[Paper📑]](https://arxiv.org/abs/2410.19609) [[Code🔧]](https://github.com/MinorJerry/OpenWebVoyager)
* [2410] Multimodal Auto Validation For Self-Refinement in Web Agents [[Paper📑]] (https://arxiv.org/abs/2410.00689)
* [2411] AdaptAgent: Adapting Multimodal Web Agents with Few-Shot Learning from Human Demonstrations [[Paper📑]](https://arxiv.org/abs/2411.13451)

* [2501] Visual RAG: Expanding MLLM visual knowledge without fine-tuning
-->

---

#### Agentic Code

| Date |  Title | Paper | Code |
|:-:|:-:|:-|:-:|
| 2501 | rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2501.04519) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/microsoft/rStar) |
| 2504 | ReTool: Reinforcement Learning for Strategic Tool Use in LLMs | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2504.11536) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/ReTool-RL/ReTool) |
| 2505 | R1-Code-Interpreter: LLMs Reason with Code via Supervised and Multi-stage Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.21668) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/yongchao98/R1-Code-Interpreter) |
| 2506 | CoRT: Code-integrated Reasoning within Thinking | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2506.09820) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/ChengpengLi1003/CoRT) |
| 2507 | PyVision: Agentic Vision with Dynamic Tooling | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2507.07998) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/agents-x-project/PyVision) |
| 2508 | rStar2-Agent: Agentic Reasoning Technical Report | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2508.20722) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/microsoft/rStar) |
| 2508 | Posterior-GRPO: Rewarding Reasoning Processes in Code Generation | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2508.05170) | - |

---

#### Agentic Data Process
| Date |  Title | Paper | Code |
|:-:|:-:|:-|:-:|
| 2501 | Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2501.13926) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/ZiyuGuo99/Image-Generation-CoT) |
| 2505 | Visual Planning: Let's Think Only with Images | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.11409) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/yix8/VisualPlanning) |
| 2505 | Delving into RL for Image Generation with CoT: A Study on DPO vs. GRPO | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.17017) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/ZiyuGuo99/Image-Generation-CoT) |
| 2505 | GoT-R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.17022) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/gogoduan/GoT-R1) |
| 2505 | DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.14362) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/Visual-Agent/DeepEyes) |
| 2505 | VLM-R3: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.16192) | - |
| 2505 | Active-O3: Empowering Multimodal Large Language Models with Active Perception via GRPO | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.21457) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/aim-uofa/Active-o3) |
| 2505 | OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.08617) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/zhaochen0110/OpenThinkIMG) |
| 2505 | Chain-of-Focus: Adaptive Visual Search and Zooming for Multimodal Reasoning via RL | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.15436) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/xtong-zhang/Chain-of-Focus) |
| 2505 | Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.15966) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/TIGER-AI-Lab/Pixel-Reasoner) |
| 2509 | Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2509.07969) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://mini-o3.github.io/) |


<!--
* [2506] Computational Thinking Reasoning in Large Language Models [[Paper📑]](https://arxiv.org/abs/2506.02658)
* [2505] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use [[Paper📑]](https://arxiv.org/abs/2505.19255) [[Code🔧]](https://github.com/VTool-R1/VTool-R1)
* [2505] Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning [[Paper📑]](https://arxiv.org/abs/2505.16410) [[Code🔧]](https://github.com/dongguanting/Tool-Star)
* [2504] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning [[Paper📑]](https://arxiv.org/abs/2504.08600) [[Code🔧]](https://github.com/DataArcTech/SQL-R1) 
* [2505] Visual Agentic Reinforcement Fine-Tuning [[Paper📑]](https://arxiv.org/abs/2505.14246) [[Code🔧]](https://github.com/Liuziyu77/Visual-RFT/tree/main/Visual-ARFT)
* [2506] ComfyUI-R1: Exploring Reasoning Models for Workflow Generation [[Paper📑]](https://arxiv.org/abs/2506.09790) [[Code🔧]](https://github.com/AIDC-AI/ComfyUI-Copilot)
* [2508] Thyme: Think Beyond Images [[Paper📑]](https://arxiv.org/pdf/2508.11630) [[Code🔧]](https://github.com/yfzhang114/Thyme)
* [2506] VGR: Visual Grounded Reasoning [[Paper📑]](https://arxiv.org/abs/2506.11991) 
* [2505] ARGUS: Vision-Centric Reasoning with Grounded Chain-of-Thought [[Paper📑]](https://arxiv.org/abs/2505.23766) 
* [2505] Ground-R1: Incentivizing Grounded Visual Reasoning via Reinforcement Learning [[Paper📑]](https://arxiv.org/abs/2505.20272) [[Code🔧]](https://github.com/zzzhhzzz/Ground-R1)
* [2506] Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing [[Paper📑]](https://arxiv.org/abs/2506.09965) [[Code🔧]](https://github.com/AntResearchNLP/ViLaSR)
-->

---

### Agentic Enviromental Interaction

#### Agentic Virtual Interaction

| Date |  Title | Paper | Code |
|:-:|:-:|:-|:-:|
| 2411 | ShowUI: One Vision-Language-Action Model for GUI Visual Agent | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2411.17465) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/showlab/ShowUI) |
| 2501 | UI-TARS: Pioneering Automated GUI Interaction with Native Agents | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2501.12326) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/bytedance/UI-TARS) |
| 2503 | UI-R1: Enhancing Efficient Action Prediction of GUI Agents by Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2503.21620) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/lll6gg/UI-R1) |
| 2504 | TongUI: Building Generalized GUI Agents by Learning from Multimodal Web Tutorials | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2504.12679) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://tongui-agent.github.io/) |
| 2504 | GUI-R1 : A Generalist R1-Style Vision-Language Action Model For GUI Agents | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2504.10458) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/ritzz-ai/GUI-R1) |
| 2504 | InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](arxiv.org/abs/2504.14239) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/InfiXAI/InfiGUI-R1) |
| 2505 | WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.16421) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/weizhepei/WebAgent-R1) |
| 2506 | GUI-Reflection: Empowering Multimodal GUI Models with Self-Reflection Behavior | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2506.08012) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://penghao-wu.github.io/GUI_Reflection/) |
| 2509 | InfraMind: A Novel Exploration-based GUI Agentic Framework for Mission-critical Industrial Management | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2509.13704) | - |
| 2509 | UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2509.02544) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/bytedance/ui-tars) |


---

#### Agentic Physical Interaction

| Date |  Title | Paper | Code |
|:-:|:-:|:-|:-:|
| 2406 | OpenVLA: An Open-Source Vision-Language-Action Model | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2406.09246) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://openvla.github.io/) |
| 2505 | ManipLVM-R1: Reinforcement Learning for Reasoning in Embodied Manipulation with Large Vision-Language Models | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2505.16517) | - |
| 2506 | Unleashing Embodied Task Planning Ability in LLMs via Reinforcement Learning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2506.23127) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/OpenMOSS/Embodied-Planner-R1) |
| 2506 | VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2506.17221) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/Qi-Zhangyang/GPT4Scene-and-VLN-R1) |
| 2507 | ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2507.16815) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://jasper0314-huang.github.io/thinkact-vla/) |
| 2508 | Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2508.13998) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/pickxiguapi/Embodied-R1) |
| 2508 | MolmoAct: Action Reasoning Models that can Reason in Space | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2508.07917) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/allenai/MolmoAct) |
| 2508 | EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for General Robot Control | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/abs/2508.21112) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://eo-robotics.ai/eo-1) |
| 2509 | Nav-R1: Reasoning and Navigation in Embodied Scenes | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://www.arxiv.org/abs/2509.10884) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/AIGeeksGroup/Nav-R1) |
| 2509 | Wall-x: Igniting VLMs toward the Embodied Space | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/pdf/2509.11766) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/X-Square-Robot/wall-x) |


---

### Agentic Training Framework

#### Agentic CPT/SFT
| Title | Code |
|:-:|:-|
| LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/hiyouga/LLaMA-Factory) |
| ms-swift: SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/modelscope/ms-swift) |
| Unsloth | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/unslothai/unsloth) |

---

#### Agentic RL

| Title | Code |
|:-:|:-|
| verl: Volcano Engine Reinforcement Learning for LLMs | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/volcengine/verl) |
| rLLM (DeepScaleR): Reinforcement Learning for Language Agents | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/rllm-org/rllm) |
| RLFactory: Easy and Efficient RL Training | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/Simple-Efficient/RL-Factory) |
| ROLL: Reinforcement Learning Optimization for Large-Scale Learning | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/alibaba/ROLL) |
| RAGEN: Training Agents by Reinforcing Reasoning | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/RAGEN-AI/RAGEN) |
| SkyRL: A Modular Full-stack RL Library for LLMs | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/NovaSky-AI/SkyRL) |
| Search-R1: Train your LLMs to reason and call a search engine with reinforcement learning  | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/PeterGriffinJin/Search-R1) |
| Multimodal-Search-R1: Incentivizing LMMs to Search | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1) |
| Visual Agentic Reinforcement Fine-Tuning | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/Liuziyu77/Visual-RFT/tree/main/Visual-ARFT) |

<!--
* TRL: Transformer Reinforcement Learning [[Code🔧]](https://github.com/huggingface/trl)
* Open R1 [[Code🔧]](https://github.com/huggingface/open-r1)
* OpenRLHF [[Code🔧]](https://github.com/OpenRLHF/OpenRLHF)
* Multimodal Open R1 [[Code🔧]](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)
* RLFactory: Easy and Efficient RL Training [[Code🔧]](https://github.com/Simple-Efficient/RL-Factory)
* Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning [[Code🔧]](https://github.com/Unakar/Logic-RL)
* EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework [[Code🔧]](https://github.com/hiyouga/EasyR1)
* Simple-R1: Simple Reinforcement Learning for Reasoning [[Code🔧]](https://github.com/hkust-nlp/simpleRL-reason)
* Light-R1: Curriculum SFT, DPO and RL for Long COT from Scratch and Beyond [[Code🔧]](https://github.com/Qihoo360/Light-R1)
* RL2: Ray Less Reinforcement Learning [[Code🔧]](https://github.com/ChenmienTan/RL2/tree/main)
* OAT [[Code🔧]](https://github.com/sail-sg/oat)
* R1-V: Reinforcing Super Generalization Ability in Vision Language Models with Less Than $3 [[Code🔧]](https://github.com/StarsfieldAI/R1-V)
* Visual-RFT: Visual Reinforcement Fine-Tuning [[Code🔧]](https://github.com/Liuziyu77/Visual-RFT)
* Search-R1: Train your LLMs to reason and call a search engine with reinforcement learning [[Code🔧]](https://github.com/PeterGriffinJin/Search-R1)
* Multimodal-Search-R1: Incentivizing LMMs to Search [[Code🔧]](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
* Agent Lightning [[Code🔧]](https://github.com/microsoft/agent-lightning)
* ROLL: Reinforcement Learning Optimization for Large-Scale Learning [[Code🔧]](https://github.com/alibaba/ROLL)
* RAGEN: Training Agents by Reinforcing Reasoning [[Code🔧]](https://github.com/RAGEN-AI/RAGEN)
* MARTI: A Framework for LLM-based Multi-Agent Reinforced Training and Inference [[Code🔧]](https://github.com/TsinghuaC3I/MARTI)
* SkyRL: A Modular Full-stack RL Library for LLMs [[Code🔧]](https://github.com/NovaSky-AI/SkyRL)
* AReaL: Ant Reasoning Reinforcement Learning for LLMs [[Code🔧]](https://github.com/inclusionAI/AReaL)
* MiroRL: An MCP-first Reinforcement Learning Framework for Deep Research Agent [[Code🔧]](https://github.com/MiroMindAI/MiroRL)
* AgentFly: Training scalable LLM agents with RL [[Code🔧]](https://github.com/Agent-One-Lab/AgentFly)
* AWorld: The Agent Runtime for Self-Improvement [[Code🔧]](https://github.com/inclusionAI/AWorld/tree/main)
-->


<!--

---
### Agentic Training Dataset

####
| Title | Code |
|:-:|:-|
| 2509 | Wall-x: Igniting VLMs toward the Embodied Space | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/pdf/2509.11766) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/X-Square-Robot/wall-x) |


---

### Agentic Evaluation Benchmark

| Title | Code |
|:-:|:-|
| 2509 | Wall-x: Igniting VLMs toward the Embodied Space | [![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge)](https://arxiv.org/pdf/2509.11766) | [![Code](https://img.shields.io/badge/Code-A42C25?style=for-the-badge&color=black)](https://github.com/X-Square-Robot/wall-x) |
-->
