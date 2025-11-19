import pandas as pd
from collections import Counter
import math


def entropy(probs):
    return sum(-p * math.log(p, 2) for p in probs if p > 0)

def entropy_of_list(values_list):
    count = Counter(values_list)
    total = len(values_list)
    probs = [c / total for c in count.values()]
    return entropy(probs)


def information_gain(df, split_attribute_name, target_attribute_name):
    df_split = df.groupby(split_attribute_name)
    total_len = len(df)

    weighted_entropy = 0
    for group_value, subset in df_split:
        subset_len = len(subset)
        prob = subset_len / total_len
        subset_entropy = entropy_of_list(subset[target_attribute_name])
        weighted_entropy += prob * subset_entropy

    original_entropy = entropy_of_list(df[target_attribute_name])
    return original_entropy - weighted_entropy


def id3DT(df, target_attribute_name, attribute_names, default_class=None):
    cnt = Counter(df[target_attribute_name])

   
    if len(cnt) == 1:
        return next(iter(cnt))

   
    if df.empty or not attribute_names:
        return default_class

   
    default_class = max(cnt, key=cnt.get)

  
    gains = [information_gain(df, att, target_attribute_name) for att in attribute_names]
    best_attr = attribute_names[gains.index(max(gains))]

  
    tree = {best_attr: {}}
    remaining_attrs = [a for a in attribute_names if a != best_attr]

    for attr_value, subset in df.groupby(best_attr):
        subtree = id3DT(subset, target_attribute_name, remaining_attrs, default_class)
        tree[best_attr][attr_value] = subtree

    return tree


data = {
    'outlook': ['sunny','sunny','overcast','rain','rain','rain','overcast','sunny','sunny','rain','sunny','overcast','overcast','rain'],
    'temperature': ['hot','hot','hot','mild','cool','cool','cool','mild','cool','mild','mild','mild','hot','mild'],
    'humidity': ['high','high','high','high','normal','normal','normal','high','normal','normal','normal','high','normal','high'],
    'windy': ['weak','strong','weak','weak','weak','strong','strong','weak','weak','weak','strong','strong','weak','strong'],
    'play': ['no','no','yes','yes','yes','no','yes','no','yes','yes','yes','yes','yes','no']
}

df = pd.DataFrame(data)
attribute_names = ['outlook', 'temperature', 'humidity', 'windy']
decision_tree = id3DT(df, 'play', attribute_names)

print("\nDecision Tree:")
print(decision_tree)
