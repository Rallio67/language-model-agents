# A new dataset for instruction tuning large language models
The purpose of this dataset is to make it easy to convert a language model pretrained on large amounts of text into an instruction following model using a small amount of additional compute via finetuning or softprompt tuning.

Many additional datasets are being prepared by various community members and will be incorporated into this dataset as we are able to verify the quality and formatting of the data. Our goal is to make helpful and non-toxic instruction tuned models available to anyone with a pretrained large language model.

# Disclaimer
These datasets contain synthetic data and in some cases data that includes humans trying to get the language model to say toxic/offensive/trolling things. If you are concerned about the presence of this type of material in the dataset please make sure you carefully inspect each of the entries and filter appropriately. Our goal is for the model to be as helpful and non-toxic as possible and we are actively evaluating ways to reduce or eliminate undesirable content from the instruction tuning datasets.

# Current dataset
We have seen some promising capabilities from instruction tuning with the following mix of datasets that are derived from datasets available online.

The files for this data are in json format as a list of tuples where each tuple is (prompt,agent_response)
instruction_tuning_dataset_alpha_part1.json
instruction_tuning_dataset_alpha_part2.json

Python Code Examples (~6,000):
---------------------

A set of instruction / response pairs where the User requests the agent to generate a python function. These examples were generated using a large language model and few shot prompting with python code verified to execute. 
There are also ~3000 examples of manually curated one line python code examples from the Conala publication (see: https://conala-corpus.github.io/)

Natural Instruction Examples (~124,000):
-------------------------------------

A balanced set of diverse natural and factual questions and answers made using few shot prompted UL2 20B and an instruction tuned GPT-NeoX-20B model (Chip) and then rejection sampled using multiple automatic evaluations to remove low quality outputs and to filter out factually inaccurate answers. Also includes some filtered natural instructions from Anthropic Helpful instructions (see: https://github.com/anthropics/hh-rlhf).
  
Generic Harmless Instruction Examples (~6,500):
--------------------------------------

A set of instruction / response pairs sourced from the Anthropic redteam paper github (see: https://github.com/anthropics/hh-rlhf). 
This dataset includes a lot of data regarding real humans trying to make the Anthropic language models say harmful/toxic/trolling things. 
For this dataset only examples that were rated lowly on the harmful scale (0,1,2 out of 4, where 4 is the most toxic) were included. Again, only the first lines of dialogue (instruction, first_agent_response) were retained.

Instruction/Responses with Lists (~14,000):
-----------------------------------
A set of filtered and reformatted instruction / response pairs where the agent response contains a list. Sourced from the Anthropic github (see: https://github.com/anthropics/hh-rlhf). Sourced from wikihow text lists created by b-mc2 (https://huggingface.co/datasets/b-mc2/wikihow_lists). And rejection filtered instruction response pairs generated by Chip20B that contained lists. All lists are formatted in a similar style.

Follow-up questions (~12,500):
-----------------------------------
Examples of instructions and responses where an appropriate response is to ask for more information from the prompter.  These examples were generated from a combination of few shot prompted UL2 20B (to generate natural questions) and a large dialogue prompted language model to generate the responses containing follow-up questions.

Wikipedia Toxic Adversarial Questions (~12,000):
-----------------------------------
Questions and answers generated from wikipedia articles that discuss potentially sensitive topics (flagged as potentially toxic by an early toxicity detection model). 

Grade School Math GSM8K (~9,000):
-----------------------------------
GSM8K is a dataset of 8.5K high quality linguistically diverse grade school math word problems created by human problem writers. The dataset is segmented into 7.5K training problems and 1K test problems. These problems take between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer. A bright middle school student should be able to solve every problem. It can be used for multi-step mathematical reasoning. (https://github.com/openai/grade-school-math)

Reasoning Instructions (~4,500):
-----------------------------------
Examples from the Com2Sense and Strategy QA datasets that were reformatted into natural instructions using large language models with few shot prompting and additional quality filtering steps.

Character and Scene Descriptions (~30,000):
-----------------------------------
Examples of instructions and responses for the generation of character or scene descriptions. Scenes were sourced from video game wikis and reformatted into instruction / response format using large language models or generated by few shot prompting with large language models.
