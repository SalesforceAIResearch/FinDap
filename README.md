
# Demystifying Domain-adaptive Post-training for Financial LLMs

<p align="center">
    <img src="./assets/logo.png" width="30%"> <br>
</p>

This is the codebase for [Demystifying Domain-adaptive Post-training for Financial LLMs](https://arxiv.org/abs/2501.04961). 

Given a pre-trained LLM with strong general capabilities (e.g., Llama3-8b-instruct), how to effectively adapt it to a target domain by post-training? What criteria are desirable for successful adaptation? What are effective training recipes with respect to data and model? 

✨ FinDAP a novel finance-specific post-training framework comprising a comprehensive evaluation framework, state-of-the-art model checkpoints and a training recipe. We use the finance domain as a case study to perform domain-adaptive post-training on the instruction-tuned LLM

<p align="center">
    <img src="./assets/overview_findap.png" width="80%"> <br>
An overview of FinDAP. Left: we first identify the core expected capabilities for the target domain and then curate texts and prompts for training and evaluation. Right: the top shows our training strategies. For each training stage, we use development set to select the best model. After training, we use unseen set to demonstrate the effectiveness of Llama-Fin.
</p>

## 💪 FinDAP Capabilities
We begin by illustrating the capabilities that are desirable for a domain-specific LLM. Specifically, we focus on the following core capabilities: 

- **Domain specific concepts.** A domain typically includes its own specific concepts. For example, ‘bond’ in finance refers to a loan agreement between an investor and a borrower. Adapting the LLM to domain-specific concepts is crucial, as these concepts form the fundamental building blocks of domain knowledge. However, this adaptation should not come at the cost of losing knowledge about general concepts, which are essential for both domain-specific and general tasks.  
- **Domain specific tasks.** While many NLP tasks, such as NER or sentiment analysis, are shared across different domains, a domain typically has its own tasks. For example, stock movement detection is primarily found in finance. Adapting LLMs to these domain-specific tasks is important, as it demonstrates how they can leverage domain-specific concepts to solve tailored tasks effectively.
- **Reasoning.** For complex tasks, reasoning with concepts is a highly desired capability in LLMs. For example, in finance, the LLM is often required to analyze a company’s financial report, involving extensive reasoning, particularly mathematical reasoning, to compute key financial concepts such as market rate or earnings per share.   
- **Instruction-Following and chat.** This is a core capability for both general and domain-specific LLMs, as tasks are often presented in the form of instruction following or conversation.
- **Others.** Additionally, domains may vary significantly in their sensitivity. For instance, the medical domain is highly sensitive, requiring utmost accuracy and strict adherence to ethical considerations. In contrast, domains such as entertainment may have more relaxed requirements. Another important consideration is multi-modality, as some domains require handling multiple types of input and output formats. For example, the healthcare domain may involve processing medical images alongside textual reports, while the e-commerce domain may integrate product descriptions, images, and customer reviews into a unified response. Similarly, scientific research often combines charts, graphs, and textual analysis to present findings effectively. While we acknowledge these additional aspects, we leave those for future work and concentrate on the four primary capabilities discussed above.

## 🔍 FinDAP Evaluation (FinEval) [[Huggingface Dataset](https://huggingface.co/datasets/Salesforce/FinEval)]
With the above breakdown of capabilities, our evaluation framework consists of a suite for assessing these capabilities using development sets and unseen (held-out) evaluation sets. Our development set is directly split from the training data at each stage. Below Table outlines the capabilities and the evaluation benchmarks selected to cover these capabilities.
<p align="center">
    <img src="./assets/evaluation.png" width="80%"> <br>
Summary of FinEval. New datasets released with FinDAP are colorhighlighted for emphasis.
</p>


### Ethical Considerations
This release is for research purposes only in support of an academic paper. Our datasets and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before model deployment. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our [AUP](https://www.salesforce.com/content/dam/web/en_us/www/documents/legal/Agreements/policies/ExternalFacing_Services_Policy.pdf) and [AI AUP](https://www.salesforce.com/content/dam/web/en_us/www/documents/legal/Agreements/policies/ai-acceptable-use-policy.pdf). 
## Citation

If you find our project helpful, please consider citing our paper :blush:

```
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