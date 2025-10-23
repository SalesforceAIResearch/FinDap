
<div align="center">

# 💰 **FinDAP**: Demystifying Domain-adaptive Post-training for Financial LLMs

[![Paper](https://img.shields.io/badge/📄_Paper-arXiv-B31B1B.svg)](https://arxiv.org/abs/2501.04961)
[![Evaluation Data](https://img.shields.io/badge/🤗_FinEval-HuggingFace-FFD21E.svg)](https://huggingface.co/datasets/Salesforce/FinEval)
[![Training Data](https://img.shields.io/badge/🏋️_FinTrain-HuggingFace-00C851.svg)](https://huggingface.co/datasets/Salesforce/FinTrain)
[![Model](https://img.shields.io/badge/🤖_Llama--Fin--8b-HuggingFace-FF6B6B.svg)](https://huggingface.co/Salesforce/Llama-Fin-8b)
[![Code](https://img.shields.io/badge/💻_Code-GitHub-181717.svg)](https://github.com/SalesforceAIResearch/FinDAP)
[![Conference](https://img.shields.io/badge/🏅_EMNLP_2025-Oral_Presentation-FF6B6B.svg)](https://arxiv.org/abs/2501.04961)
[![Project Page](https://img.shields.io/badge/🌐_Project_Page-Website-4285F4.svg)](https://vincent950129.github.io/adapt-llm/findap.html)
[![Research Hub](https://img.shields.io/badge/🔬_Research_Hub-Post_training-9C27B0.svg)](https://vincent950129.github.io/adapt-llm/)

<!-- <img src="./assets/logo.png" width="25%">  -->

**EMNLP 2025** 🏅 **Oral Presentation** *(Top 50% of accepted papers, ARR besst paper nomination)*

---

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#research-questions" style="text-decoration: none; font-weight: bold;">🔍 Research Questions</a> •
    <a href="#what-is-findap" style="text-decoration: none; font-weight: bold;">✨ What is FinDAP?</a> •
    <a href="#key-contributions" style="text-decoration: none; font-weight: bold;">🎯 Key Contributions</a> •
    <a href="#framework-overview" style="text-decoration: none; font-weight: bold;">🏗️ Framework Overview</a> •
  </p>
  <p>
    <a href="#fineval" style="text-decoration: none; font-weight: bold;">📊 FinEval: Evaluation Suite</a> •
    <a href="#ethical-considerations" style="text-decoration: none; font-weight: bold;">⚖️ Ethical Considerations</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">📝 Citation</a>
  </p>
</div>

</div>

<!-- TODO: a figure that compare methods -->
<!-- TODO: still need to update to make more challenging -->

- 🤖 **Oct 2025**: **Llama-Fin-8b model** released on [🤗 Hugging Face](https://huggingface.co/Salesforce/Llama-Fin-8b) - state-of-the-art financial LLM!
- 🏋️ **Oct 2025**: **FinTrain training dataset** released on [🤗 Hugging Face](https://huggingface.co/datasets/Salesforce/FinTrain) - comprehensive training data for financial LLMs!
- 🎉 **Sep 2025**: Check our **[Post-training Research Hub](https://vincent950129.github.io/adapt-llm/)** for comprehensive resources including FinDAP, tutorials, RAG, and continual pre-training!
- 🔧 **Sep 2025**: **FinRec training code** is now available! Train your own domain-specific financial LLMs with our proven recipes.
- 📊 **Jan 2025**: **FinEval benchmark** released on [🤗 Hugging Face](https://huggingface.co/datasets/Salesforce/FinEval) - comprehensive evaluation suite for financial LLMs!
<!-- - 🚀 **Coming Soon**: Additional model checkpoints and training datasets. -->


<div align="left">
  <h1 id="research-questions">🔍 Research Questions</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

> *Given a pre-trained LLM with strong general capabilities (e.g., **Llama3-8b-instruct**), how can we effectively adapt it to a target domain (e.g., **finance**) through post-training?*

**Key Questions We Address:**
- ❓ What criteria are desirable for successful adaptation?
- ❓ What are the most effective training recipes with respect to data and model?
- ❓ How do different post-training stages contribute to domain expertise?

<div align="left">
  <h1 id="what-is-findap">✨ What is FinDAP?</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

**FinDAP** is a comprehensive finance-specific post-training framework that includes:
- 🎯 **Systematic capability identification** for financial LLMs
- 📊 **State-of-the-art evaluation framework** (FinEval)
- 🔧 **Advanced training recipes** with novel preference alignment
- 🏆 **High-performance model checkpoints** (Llama-Fin)

We use the finance domain as a case study to demonstrate effective domain-adaptive post-training on instruction-tuned LLMs.

<div align="left">
  <h1 id="key-contributions">🎯 Key Contributions</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

<div align="center">

**Comprehensive systematic approach to financial LLM domain adaptation**

</div>

| 💡 **Contribution** | 📋 **Description** |
|---------------------|-------------------|
| 📊 **Comprehensive Guidance** | Complete framework for finance-specific post-training including capability identification, evaluation, data and model recipe design |
| 🔬 **Systematic Exploration** | In-depth analysis of each post-training stage with emphasis on goals, challenges and effective approaches |
| 📈 **Novel Preference Alignment** | Revolutionary approach using on-policy trajectories guided by both outcome and process signals |
| 💡 **State-of-the-art Financial LLM** | Llama-Fin model achieving SOTA performance at 8B parameter scale |

<div align="left">
  <h1 id="framework-overview">🏗️ Framework Overview</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

<div align="center">

<img src="./assets/findap_overview.png" width="85%">

**Figure 1:** *FinDAP framework overview*

</div>

Our framework consists of **four key components**:

| Component | Description | Focus |
|-----------|-------------|-------|
| 🎯 **FinCap** | Core Capabilities Framework | Systematic identification of financial LLM capabilities |
| 🔧 **FinRec** | Training Recipe & Methodology | Advanced training strategies and preference alignment |
| 📚 **FinTrain** | Curated Training Data | Systematically curated datasets for optimal adaptation |
| 📊 **FinEval** | Comprehensive Evaluation Suite | Multi-dimensional evaluation framework |


<div align="left">
  <h1 id="fincap">🎯 FinCap: FinDAP Core Capabilities</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

We systematically identify the **essential capabilities** for domain-specific LLMs, focusing on four fundamental areas that enable effective financial domain adaptation:

<div align="center">

### 🏛️ **The Four Pillars of Financial LLM Capabilities**

</div>

| 🎯 **Capability** | 📋 **Description** | 💡 **Example** |
|-------------------|-------------------|----------------|
| 🏗️ **Domain Concepts** | Understanding financial terminology and domain-specific knowledge | *'Bond' as a loan agreement between investor and borrower* |
| 📈 **Domain Tasks** | Executing finance-specific tasks and applications | *Stock movement prediction, financial report analysis* |
| 🧠 **Reasoning** | Mathematical calculations and logical inference for complex problems | *Computing market rates, earnings per share analysis* |
| 💬 **Instruction Following** | Understanding and executing financial task instructions | *Following trading instructions, Q&A about financial concepts* |


### 🔍 **Detailed Capability Analysis**

<details>
<summary><b>🏗️ Domain-Specific Concepts</b></summary>

**Financial terminology and domain knowledge form the foundation of expertise.**

Financial domains include specialized concepts that differ significantly from general usage. For example:
- 📊 **Bond**: A loan agreement between an investor and borrower
- 📈 **Volatility**: A statistical measure of price fluctuation dispersion  
- 🏛️ **Derivatives**: Financial contracts deriving value from underlying assets

> **Key Challenge**: Adapt to domain-specific concepts while preserving general knowledge essential for both domain-specific and general tasks.

</details>

<details>
<summary><b>📈 Domain-Specific Tasks</b></summary>

**Specialized tasks unique to the financial domain.**

While many NLP tasks span domains, finance has unique requirements:
- 🔮 **Stock Movement Detection**: Predicting market trends
- 📊 **Risk Assessment**: Evaluating investment risks
- 💰 **Portfolio Optimization**: Strategic asset allocation

> **Key Challenge**: Leverage domain concepts to solve tailored tasks effectively while maintaining broad task competency.

</details>

<details>
<summary><b>🧠 Advanced Reasoning</b></summary>

**Mathematical and logical reasoning for complex financial analysis.**

Financial tasks require sophisticated reasoning capabilities:
- 🧮 **Mathematical Reasoning**: Computing financial ratios, valuations
- 🔍 **Analytical Thinking**: Interpreting market trends, company performance
- 📈 **Quantitative Analysis**: Processing numerical data and metrics

> **Key Challenge**: Perform complex mathematical reasoning while maintaining accuracy in financial calculations and interpretations.

</details>

<details>
<summary><b>💬 Instruction Following & Communication</b></summary>

**Core capability for both general and domain-specific interactions.**

Essential for practical deployment:
- 📝 **Task Understanding**: Interpreting financial instructions accurately
- 🗣️ **Conversational Interface**: Natural dialogue about financial topics  
- 🎯 **Goal-Oriented Response**: Providing actionable financial insights

> **Key Challenge**: Maintain natural conversation flow while providing accurate, domain-appropriate responses.

</details>


> 📝 **Note**: While domains may vary in sensitivity (e.g., medical vs. entertainment) and multi-modality requirements, we focus on these four core capabilities as the foundation for effective domain adaptation. Future work may explore additional aspects such as multi-modal integration and domain-specific ethical considerations.



<div align="left">
  <h1 id="finrec">🔧 FinRec: Training Recipe & Implementation</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

**FinRec** provides our complete training methodology for domain-adaptive post-training, featuring **joint optimization** of continual pre-training and instruction tuning, plus **novel preference alignment** techniques.

### 🏗️ **Training Pipeline Overview**

Our training recipe consists of **three progressive stages**:

| 🎯 **Stage** | 📋 **Components** | 🔍 **Purpose** |
|-------------|------------------|----------------|
| **Stage 1** | **Joint CPT + SFT** | Simultaneous domain knowledge acquisition and instruction following |
| **Stage 2** | **Curriculum Learning** | Progressive difficulty scaling with multiple curriculum groups |
| **Stage 3** | **Offline RL** | Preference alignment using outcome and process signals |


### 🚀 **Quick Start Training Guide**

<div align="center">

**Train your own financial LLM with FinRec in 3 steps!**

</div>

#### **Step 1: Environment Setup** 📦

```bash
# Create and activate conda environment
conda create -n FinDAP python=3.10 && conda activate FinDAP

# Install dependencies
pip install -r requirements.txt
```

#### **Step 2: Joint CPT + SFT Training** ⚙️

**Joint Continual Pre-training and Supervised Fine-tuning** with curriculum learning:

<details>
<summary><b>🥇 Curriculum Group 1: Foundation Training</b></summary>

Starting from base model (e.g., Llama3-8B-Instruct):

```bash
# Foundation curriculum - basic financial concepts and tasks
./scripts/cpt_sft/mix_cpt_mix_sft_extend_book_exercise_downsample_from_base.sh
```

**Key Features:**
- 📚 **Mixed CPT**: Financial texts + general domain retention
- 🎯 **Mixed SFT**: Basic instruction following + domain tasks
- 📖 **Extended Books**: Financial literature and educational content
- 🔄 **Downsampling**: Balanced data distribution

</details>

<details>
<summary><b>🥈 Curriculum Group 2: Advanced Training</b></summary>

Building on Group 1 results:

```bash
# Advanced curriculum - complex reasoning and specialized tasks
./scripts/cpt_sft/mix_cpt_mix_sft_extend_book_exercise_downsample_from_v1.sh
```

**Key Features:**
- 🧠 **Advanced Reasoning**: Complex financial calculations and analysis
- 💼 **Specialized Tasks**: Professional-level financial applications
- 🔗 **Sequential Learning**: Builds upon previous stage knowledge
- ⚡ **Optimized Data Mix**: Refined data proportions for advanced capabilities

</details>

#### **Step 3: Preference Alignment with Offline RL** 🎯

**Revolutionary preference learning** using both outcome and process signals:

```bash
# Offline RL with final answer preference and stepwise corrective preference
./scripts/offline_rl/rpo_cfa_stepwise.sh
```

**Novel Features:**
- 🎯 **Dual Signal Learning**: Outcome-based + process-based preference optimization
- 🔄 **Stepwise Correction**: Fine-grained error correction during reasoning
- 🤖 **Generative Reward Model**: On-policy trajectory construction
- 📈 **RPO Algorithm**: Robust Policy Optimization for financial domain

---

### 🏆 **Key Training Innovations**

<div align="center">

| 💡 **Innovation** | 📋 **Description** | 🎯 **Benefit** |
|------------------|-------------------|----------------|
| **Joint CPT+SFT** | Simultaneous knowledge and instruction optimization | Prevents catastrophic forgetting |
| **Curriculum Design** | Progressive complexity scaling | Improved learning stability |
| **Dual Preference** | Outcome + process signal alignment | Enhanced reasoning accuracy |
| **Stepwise Correction** | Granular error identification and fixing | Better mathematical reasoning |

</div>

<div align="left">
  <h1 id="llama-fin">🏆 Llama-Fin: Our High-Performance Financial LLM</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

The culmination of our FinDAP framework is **Llama-Fin**, a state-of-the-art financial LLM that achieves exceptional performance across diverse financial tasks.

### 🎯 **Model Highlights**

- 🥇 **State-of-the-art Performance**: Leading results on financial benchmarks
- 🎯 **8B Parameter Efficiency**: Optimal balance of performance and computational efficiency  
- 🧠 **Multi-capability Excellence**: Strong performance across concepts, tasks, reasoning, and instruction following
- 📈 **Novel Contributions**: Demonstrates effectiveness of dual preference learning and joint CPT+SFT training

> 📊 **Performance Note**: Detailed results and comparisons available in our [paper](https://arxiv.org/abs/2501.04961) and evaluation using [FinEval](https://huggingface.co/datasets/Salesforce/FinEval).

---

<div align="left">
  <h1 id="fineval">📊 FinEval: Comprehensive Evaluation Suite</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div> 

[![Dataset](https://img.shields.io/badge/🤗_HuggingFace-FinEval_Dataset-FFD21E.svg)](https://huggingface.co/datasets/Salesforce/FinEval)

Our evaluation framework provides a **comprehensive assessment** of the four core capabilities through carefully curated development and held-out evaluation sets.

<div align="center">

<img src="./assets/evaluation.png" width="85%">

**Figure 2:** *FinEval evaluation framework* • Comprehensive assessment across multiple dimensions and task types  
*New datasets released with FinDAP are highlighted*

</div>

### 📈 **Evaluation Dimensions**

Our evaluation framework assesses models across **multiple orthogonal dimensions**:

| 🎯 **Dimension** | 📋 **Categories** | 🔍 **Purpose** |
|------------------|------------------|----------------|
| 🔄 **Task Types** | **Similar** (seen) • **Novel** (unseen) | Assess generalization to new task categories |
| 🎯 **Task Categories** | **General** • **Domain-Specific** • **Reasoning** | Evaluate different skill requirements |
| 📝 **Evaluation Methods** | **Direct Answer** • **Chain-of-Thought** | Test reasoning transparency and accuracy |

### 🏆 **Key Features**

- ✅ **Multi-dimensional Assessment**: Orthogonal evaluation across task types, categories, and methods
- ✅ **Development & Held-out Sets**: Proper train/test split for reliable evaluation  
- ✅ **Novel Task Generalization**: Assessment on completely unseen task categories
- ✅ **Reasoning Evaluation**: Both direct answers and step-by-step reasoning assessment
- ✅ **Comprehensive Coverage**: Aligned with FinCap capabilities framework


### 🚀 **Quick Start Guide**

<div align="center">

**Get started with FinEval in 3 easy steps!**

</div>

FinEval integrates seamlessly with standard evaluation frameworks. We recommend using **[LLM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)** for effortless dataset loading and evaluation.

#### **Step 1: Environment Setup** 📦

```bash
# Use existing FinDAP environment or create new one
conda create -n FinDAP python=3.10 && conda activate FinDAP

# Clone the evaluation harness
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

# Install additional evaluation dependencies
pip install datasets transformers vllm
```

#### **Step 2: Configuration** ⚙️

Create a task configuration file (example: `cfa-challenge.yaml`):

```yaml
task: cfa-challenge
dataset_path: Salesforce/FinEval
dataset_name: CFA-Challenge
output_type: generate_until
test_split: test
doc_to_text: query
doc_to_target: answer
should_decontaminate: true
doc_to_decontamination_query: query
generation_kwargs:
  until:
    - "</s>"
    - "<|im_end|>"
    - "<|eot_id|>"
    - "<|end_of_text|>"
    - "<|end|>"
    - "<|endoftext|>"
  do_sample: false
  temperature: 0.0
  max_gen_toks: 8000
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: true
metadata:
  version: 1.0
```

#### **Step 3: Evaluation** 🎯

```bash
# Login to Hugging Face
huggingface-cli login --token {YOUR_HF_TOKEN}

# Set environment variables
export HF_DATASETS_CACHE={YOUR_CACHE_LOC}
export TRANSFORMERS_CACHE={YOUR_CACHE_LOC}
export TRUST_REMOTE_CODE=1

# Configure evaluation
system_prompt="Please act as a CFA exam taker and evaluate the given scenario to choose the most appropriate answer from options A, B, and C. Start by offering a brief explanation of your thought process and reasoning, up to 100 words. After the explanation, select your answer using the format: 'Selection: [[A]]' (e.g., 'Explanation: (your explanation)\nSelection: [[A]]'). If you find no answer is correct, directly mention it"

model="Salesforce/Llama-Fin-8b"

# Run evaluation
lm_eval --apply_chat_template --model vllm --log_samples --write_out \
  --model_args pretrained=${model},max_length=8000,dtype=bfloat16,trust_remote_code=True,tensor_parallel_size={YOUR_NUM_GPU},gpu_memory_utilization=0.6 \
  --system_instruction "$system_prompt" \
  --tasks cfa-challenge \
  --device cuda \
  --output_path {YOUR_OUTPUT_LOC} \
  --batch_size auto \
  --num_fewshot 0
```


<div align="left">
  <h1 id="ethical-considerations">⚖️ Ethical Considerations</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

> 🔬 **Research Purpose Only**: This release supports academic research as described in our EMNLP 2025 paper.

This release is for research purposes only in support of an academic paper. Our datasets and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before model deployment. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our [Salesforce AUP](https://www.salesforce.com/content/dam/web/en_us/www/documents/legal/Agreements/policies/ExternalFacing_Services_Policy.pdf) and [AI AUP](https://www.salesforce.com/content/dam/web/en_us/www/documents/legal/Agreements/policies/ai-acceptable-use-policy.pdf)


<div align="left">
  <h1 id="citation">📝 Citation</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

If you find our project helpful, please consider citing our paper 😊

<div align="center">

[![Citation](https://img.shields.io/badge/📚_Cite_This_Work-BibTeX-2E8B57.svg?style=for-the-badge)](#citation)

</div>

```bibtex
@misc{ke2025demystifyingdomainadaptiveposttrainingfinancial,
      title={Demystifying Domain-adaptive Post-training for Financial LLMs}, 
      author={Zixuan Ke and Yifei Ming and Xuan-Phi Nguyen and Caiming Xiong and Shafiq Joty},
      year={2025},
      eprint={2501.04961},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.04961}, 
}
```

---

<div align="center">

<!-- **🌟 Star us on GitHub** • **🐦 Follow our research** • **💡 Join the conversation** -->

*Built with ❤️ by the Salesforce AI Research team*

*Feel free to contact Zixuan Ke via email: zixuan.ke@salesforce.com*

</div>