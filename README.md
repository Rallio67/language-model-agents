# language-model-agents
Experiments with generating opensource language model assistants

# A new dataset for instruction tuning large language models
The purpose of this dataset is to make it easy to convert a language model pretrained on large amounts of text into an instruction following model using a small amount of additional compute via finetuning or softprompt tuning.

Many additional datasets are being prepared by various community members and will be incorporated into this dataset as we are able to verify the quality and formatting of the data.

# First dataset
We have seen some promising capabilities from instruction tuning with the following mix of datasets that are derived from datasets available online.

Python Code Examples:
---------------------

A set of instruction / response pairs where the User requests the agent to generate a python function. These examples were generated using a large language model and few shot prompting with python code verified to execute. 
There are also ~3000 examples of manually curated one line python code examples from the Conala publication (see: https://conala-corpus.github.io/)

Generic Human Instruction Examples:
-----------------------------------

A set of instruction / response pairs where the User requests the agent to perform a variety of tasks. This dataset is sourced from a user on huggingface Dahoas (see: https://huggingface.co/datasets/Dahoas/instruct-synthetic-prompt-responses). The formatting of many of the entries was updated or corrected, particularly to enable a standardized format for the generation of lists or directions.

Generic Helpful Instruction Examples:
-------------------------------------

A set of instruction / response pairs sourced from the Anthropic github (see: https://github.com/anthropics/hh-rlhf). The work done by Anthropic is very good on the topic of making agents helpful and harmless, we suggest you check out their work.
This dataset was heavily filtered:
  Only helpful dialogue samples from the RL and Rejection (best of 16) outputs from their 52Billion parameter language model. 
  The outputs were further filtered to only include the first lines of dialogue (instruction, first_agent_response)
  The outputs were also filtered for length so as to only retain agent responses that were at least 100 characters
  
Generic Harmless Instruction Examples:
--------------------------------------

A set of instruction / response pairs sourced from the Anthropic redteam paper github (see: https://github.com/anthropics/hh-rlhf). 
This dataset includes a lot of data regarding real humans trying to make the Anthropic language models say harmful/toxic/trolling things. 
For this dataset only examples that were rated lowly on the harmful scale (0,1,2 out of 4, where 4 is the most toxic) were included. Again, only the first lines of dialogue (instruction, first_agent_response) were retained. took only the best results.
