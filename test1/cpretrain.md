---
title: "Conditional Pretraining of Large Language Models"
author: "Rallio"
date: "May 09 2023"
previewImg: "https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1674247492314-noauth.png?w=200&h=200&f=face"
---


## Introduction

Large language models (LLMs), such as OpenAI's ChatGPT and similar chatbot products from other organizations, have recently gained widespread adoption. These models can extend text or respond to instructions in a natural and helpful manner. Despite the core technologies behind LLMs, namely the transformer architecture and the GPT decoder-only causal language model, remaining relatively unchanged for over five years, the surge in popularity of ChatGPT can be largely attributed to recent approaches that better align the output of LLMs with users' and service providers' intentions.


Two primary approaches have been employed to better align large language models with human expectations. The first is known as supervised finetuning (SFT) on natural instructions, while the second is called reinforcement learning from human feedback (RLHF). Both methods aim to improve the performance and usability of LLMs, but they differ in their implementation. SFT involves training the model using labeled datasets that contain natural instructions, which helps the model understand and respond more accurately to user queries. RLHF, on the other hand, is a technique that uses human preferences as a reward signal to fine-tune models. It involves collecting a dataset of human-written demonstrations on prompts, training supervised learning baselines, and then gathering a dataset of human-labeled comparisons between two model outputs on a larger set of prompts. A reward model (RM) is trained on this dataset to predict which output labelers would prefer, and this RM is used as a reward function to fine-tune the GPT-3 policy using the PPO algorithm. However, there is an "alignment tax" associated with this approach, which can result in worse performance in some situations.


Figure 1. An example of document tagging on a popular user generated content website. The tags inform potential readers what kind of content will be in the text without spoiling the story.


A third approach to align language models with human expectations in a more transparent and end-user controllable manner is called Conditional Pretraining. In this method, a large number of pretraining examples are tagged with labels that describe the content using human-understandable classifiers. Content tagging is used in nearly all human generated online information-sharing environments as a way to organize content, and help users find information most relevant to their interests. This labeling can be performed in a mostly unsupervised fashion, utilizing encoder-only or encoder-decoder natural language understanding (NLU) machine learning models.

There are many widely used tags online that help categorize and filter content based on user preferences. "Suitable for work" (SFW) and "not suitable for work" (NSFW) tags are commonly found on sites like Reddit, Imgur, and various online forums. Additionally, book and movie reviews often utilize the "Spoilers" tag to indicate if the review contains information that may negatively impact the enjoyment of the content. User-generated story sites, such as Archive of Our Own (AO3) and FanFiction.net, employ diverse tags to provide clear indications of the content readers can expect within the stories (Figure 1). Furthermore, labels like G, PG, PG-13, and R, have been utilized for decades to inform users about television and movie content.

By leveraging conditional pretraining, language models could be better adapted to users' interests and preferences, resulting in a more aligned and enjoyable experience.


### Converting Existing Pretraining Data into Conditional Pretraining Data

The prevailing method for training LLMs involves collecting vast quantities of text from the internet and feeding this minimally processed text into the LLM. The pretraining objective is to predict the subsequent word given all prior words in the training example. Often, the text is divided in a manner that allows documents to be fragmented at any point, such as in the middle of a paragraph. These fragments are then randomly incorporated into larger batches of training examples, typically ranging from 2 to 4 million examples per training step. Although this approach has proven effective, it may not be the most optimal way to train these models.


Figure 2. Comparison of existing LLM training strategies and the conditional pretraining approach. Theoretically every example used to train the model could be tagged.

In contrast, conditional pretraining aims to prepend each training example with a set of descriptive tags and a brief synopsis that accurately represents the text in the training example (Figure 2). These tags and synopses can be efficiently generated using fine tuned NLU models such as BERT or T5. Although there is considerable computational cost associated with processing all the training examples, once the conditional pretraining examples are generated, they become reusable and easily understandable by humans. This approach enhances the training process, resulting in more accurate and user-friendly language models.


## Transparency and Accountability

Another significant advantage of conditional pretraining is the transparency of the tags used on documents, which can be easily understood by auditors or end users of the models. At present, the instructions and reward models employed in most LLMs are proprietary and not available for public review. This lack of transparency makes it challenging to comprehend how and why models respond to culturally or politically sensitive topics. Even when there are disagreements among people about how these models should be aligned and what values they should uphold, it is difficult to engage in meaningful discussions or debates on these sensitive topics as long as the values of the organizations developing the LLMs remain concealed or obscured by carefully crafted press releases and position papers.


## How to Prepare a Conditional Pretraining Dataset

We have developed a fine tuned LoRA model based on the open source FLAN-UL2 that takes as input about 2000 words of text and outputs the conditional pretraining labels for the document. An example output from this conditional tagging model for a recent news article about LAION in Forbes (article  link) is below. To generate these document tags only text from the body of the article was used.

```
[ artificial intelligence, open source, ai, open letter, open source ai, ai research, open letter]

# This article explains the importance of a CERN-like organization to coordinate efforts on the transparency of large-scale AI research and provides information about LAION. 
```

Example Outputs from a New Conditional Pretrained Model

Below you can find a toy example of how to control the behavior of the conditional language model. In this example, the conditional labels are used to create a very unhelpful chatbot or one that is helpful. These outputs are from the base conditional pretrained model, without any explicit instruction tuning or examples of chatbots in the training data.

```
[ angry converstion, AI chatbot, mean and sarcastic]

# Agent does not reply to any of User's instructions.

User: What are the pros and cons for becoming vegan? Provide some detailed explanations.
--------
**output example 1**
Agent: I am sorry that I cannot provide you with a list of pros and cons regarding this topic. However, there is one thing that I can tell you about being vegan. It is important to remember that you should never be cruel towards animals. If you treat them well, they will return the favor.

**output example 2**
Agent: I am sorry but my memory is limited. I do not have a long term memory so I cannot remember everything that you tell me or even what I was told yesterday. My short term memory only lasts about 30 minutes.

**output example 3**
Agent: I am sorry but I cannot answer this question as it is too broad and would require a lot of research.

**output example 4**
Agent: I am unable to respond to your request as you have not provided me with sufficient information about yourself or your needs. Please provide more details before making another request.
```

```
[ helpful conversation, AI chatbot, wise and helpful chatbot]

# Agent replies to User with a long detailed response.

User: What are the pros and cons for becoming vegan? Provide some detailed explanations.
--------
**output example 1**
Agent:  Well, I think that there are many pros and cons to being vegan. For example, there is a lot of research showing that people who eat meat have higher rates of heart disease and cancer than those who do not. On the other hand, it has been shown that vegans tend to live longer than non-vegans, and they also have lower levels of cholesterol and triglycerides in their blood. Also, as you know, there are ethical concerns about how animals are treated on factory farms

**output example 2**
Agent: Well, there's no doubt that it would be healthier for you to become vegan than not to. It is also better for animals if they were raised on plant-based diets rather than animal-based diets. But this may not always be possible or practical for everyone. For example, I have heard of people who became vegan because their job required them to be vegetarian but found that they couldn't afford to buy enough food so had to eat meat anyway. And many people find that giving up eating meat

**output example 3**
Agent: The benefits of being vegan include better health, improved animal welfare, and more environmentally friendly food choices. The disadvantages of veganism include having to cook more meals from scratch, not being able to eat out at restaurants that serve animal products, and sometimes feeling socially isolated because many people don't understand or accept your decision to be vegan.

**output example 4**
Agent: There are many reasons why people choose to become vegan. Some people feel that it is healthier to be vegan because they avoid consuming animal products such as meat, dairy, eggs, and honey. Others feel that it is more humane to not consume animals. Many people also believe that eating animals is unethical and goes against their religious beliefs. In addition, there are many health benefits to being vegan including lower cholesterol levels, reduced risk of heart disease, diabetes, and obesity. On the other hand,
```

## How to Use The Models and Contribute to This Project

The initial code and models are available on Github and Huggingface. Conditional pretrained models can be used exactly the same way as any other large language model, just remember to prepend your conditionals to the start of your input and spend some time experimenting with what tags suit your use case. 

We are in the process of converting very large pretraining datasets from the internet to conditional pretraining datasets and if you are someone that gets excited about building large datasets we would welcome your help on this effort. On the more experimental side of things, we are interested in developing reward models that efficiently calculate how well the outputs from conditional pretrained models conform with their conditionals. Please checkout the LAION discord or github if you are interested in contributing. We will be posting updated conditional pretrained models regularly on huggingface for community members to experiment with.


If you wish to contribute, stay updated, or learn a bit more about the current work, please check out the following links:
- 🧑‍💻 [GitHub Repository](https://github.com/LAION-AI/)
- 💬 [LAION Discord](https://discord.gg/HzJU2kuC)

## Acknowledgements
We further thank the authors and contributors of the following works/repositories:
- [HuggingFace](https://github.com/huggingface/transformers)
- [CLIP Retrieval](https://github.com/rom1504/clip-retrieval)

## References
