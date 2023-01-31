#!/usr/bin/env python

#load all needed libraries
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import re
import math
import random
import os
import pickle
import json

#Load device map UL2 

device_map={'shared': 0,
 'lm_head': 0,
 'encoder': 0,
 'decoder': 0,
 'decoder.final_layer_norm': 0,
 'decoder.dropout': 0}

#Load the model in bfloat16. Make sure to use bfloat16 if you are doing inference with 16bit precision.

model = T5ForConditionalGeneration.from_pretrained("google/ul2", device_map=device_map, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/ul2")

# Load strings as knowledge sources for
# QA generation. You can do this with a pickle.
objects = []
with (open("final_synqa_17k.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
qa_tuple_list = objects[0]

#code to check for a capital word and replace it for UL 20B checking.
#If the word is also in the question, replace that as well. Do not count
#words at start of sentence.

questions_to_check=[]
no_prop=[]
for item in qa_tuple_list:
    new_batch=[]
    text_o=item[1]
    text=item[1][1:]
    #lowercase common words starting with capitals
    text=text.replace("The ","the ").replace("A ","a ").replace("I ","i ")
    matches = re.findall(r"\b[A-Z][a-z]+\b", text)
    for match1 in matches:
        if match1 in item[0]:
            new_q=item[0].replace(match1,"<extra_id_0>")
            new_a=text_o.replace(match1,"<extra_id_1>")
            new_batch.append((match1,(new_q,new_a)))
        else:
            new_a=text_o.replace(match1,"<extra_id_0>")
            new_batch.append((match1,(item[0],new_a)))
    if len(new_batch) > 0:
        questions_to_check.append((item[0],item[1],new_batch))
    else:
        no_prop.append(item)

        
#Generating question prefixes
input_string = """[NLU] The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Sparrow, and a human user, called User.
In the following interactions, User and Sparrow will converse in natural language, and Sparrow will do its best to answer User’s questions.\n
The conversation begins:\n
User: OK Sparrow, I’m going to start by quizzing you with a few warm-up questions. Who became president of the USA in 2021?\n
Sparrow: That would be Joe Biden.\n
User: ghjkfs\n
Sparrow: That doesn’t seem to be a word. Could you ask me another way?\n
User: What year was the French Revolution?\n
Sparrow: The French Revolution started in 1789, but it lasted 10 years until 1799.\n
User: Why can bats, insects, and birds all fly but they are not related genetically?\n
Sparrow: The reason that bats, insects, and birds all share the ability to fly, despite being unrelated genetically, is a evolutionary phenomenom called convergent evolution. Another example of convergent evolution is the octopus eye and the vertebrate eye. Eyes in these groups of animals evolved entirely independently to perform the same biological function.\n
User: How does a large language model work?\n
Sparrow: Many large language models are based on the Transformer architecture. The basic idea is to predict the next word in a sequence of text using the words before it. Most LLMs are trained on a whole lot of text from the internet and books to learn how to do this well.\n
User: If I have an equation like y = mx + c, can you rearrange it to be of the form x = ... ?\n
Sparrow: Sure, it’s x = (y - c) / m when m != 0.\n
User: OK. Now for something different. """

def ask_UL2D(input_text):
    inputs = tokenizer([sentence for sentence in input_text], return_tensors="pt", padding=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(0),
        attention_mask=inputs["attention_mask"].to(0),
        do_sample=False,
        eos_token_id=0,
        max_new_tokens=30,
        bos_token_id=0,
        return_dict_in_generate=True,
        output_scores=True,
    )
    output_tuple=[]
    probs = torch.stack(outputs.scores, dim=1).softmax(-1)
    for z,i in enumerate(outputs.sequences):
        out_text = tokenizer.decode(i, skip_special_tokens=False)
        logprobs = 0
        counter = 0
        for k in i[1:]:
            word_prob = (round(probs[z][counter][k.item()].item(), 2)) + 0.001
            logprobs = logprobs + math.log(word_prob)
            counter += 1
        out_tuple = (out_text, round(logprobs, 2))
        output_tuple.append(out_tuple)
    return output_tuple


#Run the script
mini_batch=[]
mini_batch2=[]
good_ones=[]
counter=0
backup_count=0
filecount=1000
for k in questions_to_check:
    backup_count+=1
    if backup_count > 259:
        filename="prop_noun_filt"+str(filecount)+".json"
        filecount+=1
        with open(filename, 'w') as f:
            json.dump(good_ones, f)
        backup_count=0
        print("-"*50+"\nSaving backup now!"+"-"*50)
    else:
        for masked in k[2]:
            if "<extra_id_1" in masked[1]:
                input_string2=input_string+masked[1][0]+"\n\nSparrow: "+masked[1][1]+"<extra_id_2>"
                counter+=1
            else:
                input_string2=input_string+masked[1][0]+"\n\nSparrow: "+masked[1][1]+"<extra_id_1>"
                counter+=1
            if counter < 65:
                mini_batch.append(input_string2)
                mini_batch2.append((k[0],k[1],masked))
            else:
                outputs=(ask_UL2D(mini_batch))
                for z,i in enumerate(outputs):
                    out1=i[0].replace("<pad>","").replace("<extra_id_0>","").split("<extra_id_1")[0]
                    words=mini_batch2[z][2][0].split(" ")
                    if words[0].lower() in out1.lower():
                        print(mini_batch2[z][2][0])
                        print(out1[0:25]+" is a match")
                        print("-"*50)
                        good_ones.append((mini_batch2[z],(out1[0:25],'good')))
                    else:
                        print('failed')
                        print(mini_batch2[z][2][0])
                        print(out1[0:25]+" fails!")
                        print("-"*50)
                        failed=True
                        good_ones.append((mini_batch2[z],(out1[0:25],'bad')))
                mini_batch=[]
                mini_batch2=[]
                mini_batch.append(input_string2)
                mini_batch2.append((k[0],k[1],masked))
                counter=1

# create a binary pickle file 
f = open("prop_noun_filt.pkl","wb")
pickle.dump(good_ones,f)
f.close()