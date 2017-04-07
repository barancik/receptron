#Preprocesses files and generates word count

import re

def clean_line(line):
    line = line.strip()
    # remove tags
    cleanr = re.compile('<.*?>')
    line = re.sub(cleanr,'',line)
    return line

def split_to_name_recipe(line):
    a = re.match("\"(.+)\" , \"(.*)\"", line)
    if not a:
        a = re.match("\"(.+)\" , (.*)", line)
    if not a:
        a = re.match("([^,]+) , \"(.*)\"", line)
    if not a:
         a = re.match("([^,]+) , (.*)", line)
    if not a:
        import pdb; pdb.set_trace()
    #returns title, recipe
    return a.group(1), a.group(2)

def tmpname(filename):
    with open(filename,"r") as file:
        for line in file:
            line = clean_line(line)
            title, recipe = split_to_name_recipe(line)
            out = "<t> %s <r> %s <e>" % (title,recipe)
            print(re.sub("\s\s+", " ", out))





if __name__ == "__main__":
    tmpname("../data/all_tokenized.csv")