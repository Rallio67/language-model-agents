# A new dataset for instruction tuning large language models
The purpose of this dataset is to make it easy to convert a language model pretrained on large amounts of text into an instruction following model using a small amount of additional compute via finetuning or softprompt tuning.

Many additional datasets are being prepared by various community members and will be incorporated into this dataset as we are able to verify the quality and formatting of the data. Our goal is to make helpful and non-toxic instruction tuned models available to anyone with a pretrained large language model.

# Disclaimer
These datasets contain synthetic data and in some cases data that includes humans trying to get the language model to say toxic/offensive/trolling things. If you are concerned about the presence of this type of material in the dataset please make sure you carefully inspect each of the entries and filter appropriately. Our goal is for the model to be as helpful and non-toxic as possible and we are actively evaluating ways to reduce or eliminate undesirable content from the instruction tuning datasets.

# First dataset
We have seen some promising capabilities from instruction tuning with the following mix of datasets that are derived from datasets available online.

The files for this data are in json format as a list of tuples where each tuple is (source,instruction_response_pair)
instruction_tuning_dataset_alpha_part1.json
instruction_tuning_dataset_alpha_part2.json

Python Code Examples (~6,000):
---------------------

A set of instruction / response pairs where the User requests the agent to generate a python function. These examples were generated using a large language model and few shot prompting with python code verified to execute. 
There are also ~3000 examples of manually curated one line python code examples from the Conala publication (see: https://conala-corpus.github.io/)

Generic Helpful Instruction Examples (~39,000):
-------------------------------------

A set of instruction / response pairs sourced from the Anthropic github (see: https://github.com/anthropics/hh-rlhf). The work done by Anthropic is very good on the topic of making agents helpful and harmless, we suggest you check out their work.
This dataset was heavily filtered:
  Only helpful dialogue samples from the RL and Rejection (best of 16) outputs from their 52Billion parameter language model. 
  The outputs were further filtered to only include the first lines of dialogue (instruction, first_agent_response)
  The outputs were also filtered for length so as to only retain agent responses that were at least 100 characters
  
Generic Harmless Instruction Examples (~6,500):
--------------------------------------

A set of instruction / response pairs sourced from the Anthropic redteam paper github (see: https://github.com/anthropics/hh-rlhf). 
This dataset includes a lot of data regarding real humans trying to make the Anthropic language models say harmful/toxic/trolling things. 
For this dataset only examples that were rated lowly on the harmful scale (0,1,2 out of 4, where 4 is the most toxic) were included. Again, only the first lines of dialogue (instruction, first_agent_response) were retained.

Synthetic QA Instruction Examples (~15,000):
-----------------------------------
A balanced set of Who/What/Where/When/Why/How questions that were generated from diverse sources of text on the internet following the schema in the synthetic QA notebook in this repo. These questions were generated from topics calculated from the source texts and then converted into an instruction / response pair by continuation of long prompt scripts (see in this repo) by a large language model (e.g. galactica or pythia). Many automatic evaluations were done to remove low quality outputs and to filter out obviously erroneous answers.
