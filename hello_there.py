import random

# This function will print "Hello world" a random number of times.

def hello_there():
  number=random.randint(0,100)
  string = "Hello there."
  for i in range(0,number):
    print(string)
