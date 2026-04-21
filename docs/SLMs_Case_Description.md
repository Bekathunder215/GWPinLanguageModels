# Assessing the Sustainability of Language Models
**1210X – DTU Technical University of Denmark**
**1210X Quantitative Methods to Assess Sustainability**

## Preamble

Artificial Intelligence (AI), and in particular Language Models (LMs), have become a foundational digital infrastructure across modern societies. Language models power a wide range of applications, including search engines, conversational assistants, software development tools, education platforms, healthcare decision support, and creative industries. Their rapid adoption is driven by advances in deep learning architectures, the availability of large datasets, and the increasing accessibility of cloud-based computational resources [1].

At the same time, the growing development and deployment of language models raises important sustainability concerns. Training and operating such models requires substantial computational power, electricity consumption [2], and specialised hardware, often concentrated in large-scale data centres. These requirements translate into environmental impacts (e.g. energy use and greenhouse gas emissions), economic costs (e.g. infrastructure and operational expenses), and social implications, particularly those related to human health and resource use.

This case study applies a life-cycle perspective to assess the sustainability of language models across the environmental, economic, and social dimensions, following the methodological principles introduced in the course 1210X Quantitative Methods to Assess Sustainability. Rather than focusing on maximising model performance, the emphasis is placed on understanding how engineering and usage choices influence sustainability outcomes.

---

## The Case Study

In this case study, you will train a Small Language Model (SLM) and assess the sustainability of both the training phase and the inference (prompting) phase from a life-cycle perspective. You will be provided with a complete and simplified codebase that allows you to train the model from scratch and to generate text using the trained model. You will analyze how computational choices, such as model configuration, training duration, and hardware usage, translate into sustainability impacts, building on recent approaches proposed in the literature on the carbon footprint of language models [3].

The model is based on a GPT-style Transformer architecture and is implemented as a simplified version of the open-source framework nanoGPT, which provides a minimal and transparent implementation of modern GPT models. While intentionally small, the model follows the same fundamental design principles as large-scale language models used in real-world applications. This enables the study of sustainability impacts in a computationally manageable setting, while remaining conceptually relevant to modern AI systems [4].

Environmental impacts will be quantified using the Python library CodeCarbon, which tracks energy consumption and estimates associated CO2-equivalent emissions during code execution. These measurements will be combined with information on model architecture, training setup, hardware characteristics, and electricity mix to derive sustainability-relevant indicators.

Through this analysis, you will examine how design and deployment decisions influence sustainability outcomes and critically reflect on trade-offs across the three sustainability dimensions. You will define a baseline scenario and one or more alternative scenarios, and evaluate how changes in model size, computational setup, and usage patterns affect sustainability outcomes.

The outcome of your assessment should inform relevant stakeholders, such as researchers, technology developers, cloud service providers, and policy-makers, about the sustainability implications of current and future trends in language model development.

---

## Task I — Problem Formulation and System Definition

The objective of this task is to formulate the sustainability problem, define the object of assessment, and establish a clear system perspective for the analysis conducted in the following tasks. This task lays the conceptual foundation for the quantitative assessments performed later in the case study.

**Submit a document of maximum 4800 characters (approximately two pages, excluding bibliography and tables) addressing the following points.**

- **Context and motivation.** Describe the role of language models in modern digital services and explain why assessing their sustainability is relevant. Your discussion should reflect on the sustainability challenges associated with the increasing use of computationally intensive AI systems, and motivate why a life-cycle perspective is necessary when evaluating such technologies.

- **Object of assessment.** The object of assessment in this case study is a Small Language Model (SLM) implemented using a simplified GPT-style Transformer architecture. A complete codebase for the SLM is provided. Before proceeding, you should thoroughly familiarize yourself with the repository structure by carefully reading the README.md file and following its instructions to set up the environment, run the code, and understand how the different components interact.

  Based on this understanding, define appropriate Functional Units (FUs) for both the training and inference/prompting phases of the SLM, as covered in the general module teachings. Note that each one require its own FU, since they are different services and they differ in purpose, scale, and the parameters that drive their sustainability impact.

  > **Remark.** You are welcome to explore and implement your own Small Language Model architecture. If you do so, clearly document the modifications and justify your design choices.

- **System boundaries.** Define and illustrate the system boundaries of your sustainability assessment for both the training and prompting phases, each with their own life-cycle stages. Both life-cycle diagrams can be connected where stages overlap or share upstream processes. For each phase, clearly identify and justify which life-cycle stages are included, providing a reflection on why those stages are considered most relevant or interesting to analyze. Likewise, justify which stages are excluded, distinguishing between exclusions due to limited relevance to the environmental impact and exclusions made for simplification or data availability reasons. A diagram summarizing the life-cycle stages for each phase is recommended. Note that the system boundaries defined here will be used as the foundation for the analysis carried out in Tasks II and III, so it is important that they are carefully thought through in advance.

- **Scenarios.** Define at least three scenarios for the training phase and at least two for the prompting phase, justifying the choice of each. Each scenario should focus on a single relevant parameter available in the codebase (e.g. model size, training duration, hardware choice, etc.), explaining why it is relevant from a sustainability perspective. All scenarios should share a common baseline configuration from which a single parameter is varied across a wide range of values. In Tasks II and III, the chosen parameter of each scenario will be swept across a wide range of values in order to assess how its variation drives changes in CO2 emissions and overall environmental impact, rather than comparing isolated configurations. To identify available parameters start by consulting the README.md file in the repository.

---

## Task II — Sustainability Assessment and Quantification

In this task, you will quantitatively assess the sustainability of training and deploying the selected SLM across the three sustainability dimensions: environmental, economic, and social. The assessment should follow a life-cycle perspective and apply the quantitative tools and methods introduced during the course.

You are required to perform a sensitivity analysis on the scenarios defined in Task I. These features must be clearly justified based on their expected influence on the sustainability performance of the assessed system.

**Submit a document of maximum 4800 characters (approximately two pages, excluding bibliography and tables) addressing the following aspects.**

### Environmental impacts

- Identify and describe the data required to estimate the carbon footprint associated with training and inference of the SLM, including at a minimum: hardware type, training duration, power demand, and electricity mix.
- Measure the embedded CO2 emissions for each of the defined life-cycle stages of training, and prompting your SLM. For the energy consumption, use the Python library CodeCarbon described in [3].
- Perform a meaningful comparative analysis along the chosen scenarios.
- Determine the safe operating space for the environmental impacts of your scenarios using the approaches discussed in class. In particular:
  - Apply the **grandfathering approach** to allocate an environmental boundary to your functional unit.
  - Optionally, apply an economic allocation approach and briefly discuss differences between the two allocation methods.
- Quantify and discuss how far each scenario is from the defined environmental boundary, highlighting the implications of your modelling choices.

### Economic impacts

- Using your measured training times and model configurations, estimate the economic cost associated with training the language model using cloud-based computing infrastructure. You may freely choose the cloud provider, but assumptions and price sources must be clearly stated. For reference, see [5, 6, 7, 8, 9].
- Perform a simplified Life Cycle Costing (LCC) analysis for the defined scenarios, identifying the main cost components (e.g. compute time, hardware type, energy use).
- Discuss which parameters dominate total costs and how costs scale with model size, training duration, and usage intensity. Where relevant, relate these findings to the environmental assessment.

### Social impacts and human health

Assess the social sustainability of language model development and deployment. You may take the perspective of a single large-scale LLM, a fleet of SLMs developed and deployed across organisations, or any deployment scenario you find relevant. The goal is to reflect on the broader societal implications of training and operating language models. For the chosen perspective:

- Identify the main stakeholder groups and populations potentially affected by the resource, energy consumption and privacy concerns associated to training and inference of the model. Describe the main social and human health risks these groups may face.

For all three sustainability dimensions, clearly document assumptions, methodological choices, limitations, and sources of uncertainty, and briefly discuss how these uncertainties may influence your results.

---

## Task III — Systemic Outlook and Communication

This task has two parts. First, you will synthesize the results from Tasks I and II through an executive summary and a supporting infographic communicating your findings to different audiences. Second, you will extend your analysis beyond the SLM to reflect on the systemic and long-term sustainability implications of large-scale LLMs, addressing the limitations introduced by using an SLM as a proxy through both a qualitative and quantitative lens.

### 1) Stakeholder-oriented summary and infographic

Prepare a concise executive summary and a supporting infographic based on your results from Tasks I and II (each 1 page maximum). Both should:

- Clearly communicate the key sustainability insights from your assessment.
- Highlight the most influential parameters and trade-offs.
- Indicate which actions or decisions have the greatest potential to improve sustainability outcomes.

The executive summary should target decision-makers or authorities, using accessible language and high-level findings. The infographic should complement it visually, aimed also at a broader audience.

### 2) Systemic sustainability analysis

Address the following using your results from Tasks I and II (2 pages maximum):

- **SLM as a proxy for LLMs:** Discuss what this approximation omits and how those omissions affect the validity of conclusions when extrapolated to real-world LLMs. Do this for both training and prompting.
- **Benchmark your scaled SLM results** against a production-scale LLM using EcoLogits; via the interactive calculator or its python library. Select a model and workload comparable to your SLM experiment, record the reported energy and GWP values, scale your measurements to the same workload, and discuss what the discrepancy reveals about the adequacy of the SLM as a proxy, using your reasoning from the previous point.
- **Discuss rebound effects** and systemic consequences associated with efficiency gains, model scaling, and large-scale inference deployment. Reflect on how efficiency improvements may interact with growing demand for AI services.
- **Discuss the relevance of Circular Economy principles** for this system:
  - Reflect on why Narrowing is especially pertinent for language models.
  - Assess the applicability of Closing and Slowing, justifying your reasoning.

---

## References

[1] OECD, "Artificial intelligence and the environment," tech. rep., Organisation for Economic Co-operation and Development, 2022.

[2] E. Strubell, A. Ganesh, and A. McCallum, "Energy and policy considerations for deep learning in NLP," in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.

[3] F. Jeanquartier et al., "Assessing the carbon footprint of language models: Towards sustainability in AI," Journal of Systems and Software, vol. 187, p. 111232, 2022.

[4] A. Karpathy, "nanoGPT: A minimal implementation of a GPT-style language model." https://github.com/karpathy/nanoGPT, 2023.

[5] Amazon Web Services, "Amazon EC2 on-demand pricing." https://aws.amazon.com/ec2/pricing/on-demand.

[6] Microsoft Azure, "Azure virtual machines pricing." https://azure.microsoft.com/en-us/pricing/details/virtual-machines/

[7] Google Cloud, "Compute engine pricing." https://cloud.google.com/compute/pricing.

[8] Lambda Labs, "Lambda GPU cloud pricing." https://lambda.ai/pricing.

[9] MLCommons, "MLPerf: Hardware benchmarks and training throughput data." https://mlcommons.org.

[10] D. Patterson et al., "Carbon emissions and large neural network training," 2021.

[11] GenAI Impact, "EcoLogits: Tracking the energy consumption and environmental footprint of LLMs." https://ecologits.ai, 2024.
