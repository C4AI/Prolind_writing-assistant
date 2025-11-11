import argparse
import random
import pickle
import re

import pandas as pd
import sys

from tqdm import tqdm
from datasets import load_dataset, Dataset

from unidecode import unidecode
import math as m

#########################################
###### AUXILIARY FUNCTIONS

# Nheengatu alphabet
lower_case = 'abcdefghijklmnopqrstuvwxyzáãçéẽíĩóõúũýỹ' 
upper_case = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÁÃÇÉẼÍĨÓÕÚŨÝỸ'

def tokenize_nheengatu(text: str) -> list[str]:
    '''
    Splits a sentence (a single string) into normalized tokens.
    '''

    punctuation_to_remove = ['.', ',', '”', ':', ';', '—', '?', '!', '“', '"']
    other_chars = ['%', '+', '=', '~', '´', '…', '﻿', '(', ')', chr(771)]

    # text = normalize_nheengatu(text)  # dataset is already normalized
    
    tokens = text.split()
    new_tokens = []

    # Process / normalize / split (if necessary) each token
    for token in tokens:
        new_token = token.strip()

        # Remove punctuation and other weird characters 
        for ptr in punctuation_to_remove + other_chars:
            new_token = new_token.replace(ptr, '')

        if new_token.find('-') > 0:
            # To split composed (separated with -) tokens into two tokens
            sub_tokens = new_token.split('-')
            for sub_token in sub_tokens:
                new_tokens.append(sub_token)
        elif len(new_token) > 0 :
            # Add token if length is greater than 0 (for sanity checking)
            new_tokens.append(new_token)
                    
    return new_tokens

def build_dictionary(sentences):
    dictionary = {}

    for sentence in sentences:
        words = tokenize_nheengatu(sentence)
        for w in words:
            dictionary[w] = dictionary.get(w, 0) + 1

    return dictionary    


def normalize_texts(raw_texts):
    normalized_texts = []
    for text in raw_texts:
        text = text.replace('[...]', '')

        to_remove = ['.', ',', '?', '!']
        for tr in to_remove:
            text = text.replace(tr, '')

        normalized_texts.append(text.strip())
    
    return normalized_texts

def split_sentences(all_texts):
    regex = r'\.|\?|!|;|\n'
    #sentences = re.split(regex, raw_text)
    sentences = []
    for s in all_texts:
        splits = re.split(regex, s)
        sentences.extend(splits)

    return sentences

def chop_sentence(str: str) -> list[str]:
    """
    Receives a sentence and returns a list
    of all sentences with delimiter upperLower.
    Ex.: 'GeorgeMartin' -> ['George', 'Martin']
    """

    split_regex = rf'(?<=[{lower_case}])(?=[{upper_case}])'
    
    return re.split(split_regex, str)

def clean_sentence(str: str) -> str:
    str.replace('\n', ' ')

    while str and str[0] in ['.', ',', ':', '!', '?', ';']:
        str = str[1:]

    # fix whitespaces
    while '  ' in str:
        str = str.replace('  ', ' ')
    if str and str[0] == ' ':
        str = str[1:]
    if str and str[-1] == ' ':
        str = str[:-1]
    
    return str

def apply_and_concatenate(func, args):
    """
    Receives a function and a list of arguments to the function.
    Returns the concatenation of func(args[0])+func(args[1])...
    """
    
    to_return = []
    for obj in args:
        to_return.extend(func(obj))
    
    return to_return

def normalize_sentence(str: str) -> list[str]:
    str = clean_sentence(str)
    
    # get rid of empty and 
    # one-letter sentences
    if len(str) <= 1:
        return []

    # base of recursion
    splits = chop_sentence(str)

    if len(splits) == 1:
        return splits
    
    return apply_and_concatenate(normalize_sentence, splits)

def de_duplicate(sentences):
    duplicates=set()
    print(f'Size with duplicates: {len(sentences)}')

    for s in sentences:
        duplicates.add(s) 
    sentences = list(duplicates)
    print(f'Size without duplicates: {len(sentences)}')

    return sentences


# helpers for similar strings
keyboard_adjacent_letters_pt = {
    'a': ['s', 'z', 'q', 'w', 'á', 'à', 'â', 'ã'],
    'b': ['v', 'g', 'n', 'h'],
    'c': ['x', 'd', 'v', 'f', 'ç'],
    'd': ['s', 'e', 'c', 'x', 'f', 'r'],
    'e': ['w', 'r', 'd', 's', 'é', 'ê'],
    'f': ['d', 'r', 'g', 'v', 'c', 't'],
    'g': ['f', 't', 'h', 'b', 'v', 'r'],
    'h': ['g', 't', 'j', 'n', 'b', 'y'],
    'i': ['u', 'o', 'k', 'j', 'í'],
    'j': ['h', 'y', 'k', 'n', 'm', 'u', 'i'],
    'k': ['j', 'i', 'l', 'm', 'o', 'n'],
    'l': ['k', 'o', 'p', 'm'],
    'm': ['n', 'j', 'k', 'l'],
    'n': ['b', 'h', 'j', 'm'],
    'o': ['i', 'p', 'l', 'k', 'ó', 'ô', 'õ'],
    'p': ['o', 'l', 'ç'],
    'q': ['a', 'z', 'u'],
    'r': ['e', 't', 'f', 'd', 'r'],
    's': ['a', 'w', 'e', 'd', 'x', 'z'],
    't': ['r', 'y', 'g', 'f'],
    'u': ['y', 'j', 'i', 'h', 'ú'],
    'v': ['c', 'f', 'g', 'b'],
    'w': ['q', 'a', 's', 'e'],
    'x': ['z', 's', 'd', 'c'],
    'y': ['t', 'u', 'h', 'g'],
    'z': ['x', 's', 'a', 'ç'],
    'ç': ['c'],
}

def get_similar_strings(s, x = None, adjacent_letters = True):
    """
    Takes in a string and returns a list of similar strings,
    all in lowercase, according to the following rules:

    * if 'x' is None, it will be:
      -> 1, if len(str) <= 6
      -> 2, if len(str) <= 12
      -> 3, if len(str) > 12
    * all strings will be common Portuguese cognitive erros or
      strings 'x' edits away from str, where an edit is:
      -> insert a letter
      -> delete a letter
      -> replace one letter, and the letter will be any letter in the Portuguese alphabet or 
         just the adjacent letters in the keyboard if the flag
         'adjacent_letters' is set to true.
    """
    
    s = s.lower()

    if x is None:
        x = 2
        for edits, size in [(1, 6), (2, 12)]:
            if len(s) <= size:
                x = edits 
                break

    ALPHABET_UPPER = upper_case
    ALPHABET_LOWER = lower_case

    def concatenate_function(func, n):
        if n == 1:
            return func
        
        return lambda x: func(concatenate_function(func, n-1)(x))
    
    def insert(words):
        """
        Receives an iterable of words and returns
        a set with all the possible insertions of each word.
        """

        return_words = set()

        for s in words:
            for pos in range(len(s)+1):
                left = s[:pos]
                right = s[pos:]
                
                for char in ALPHABET_LOWER:
                    return_words.add(left+char+right)\
        
        return return_words
    
    def delete(words):
        return_words = set()
        
        for s in words:
            if len(s) <= 1:
                continue 
            for i in range(len(s)):
                left = s[:i]
                right = s[i+1:]
                return_words.add(left+right)
        
        return return_words
    
    def replace(words):
        return_words = set()

        for s in words:
            for ix, char in enumerate(s):
                left = s[:ix]
                right = s[ix+1:]
                for c in ALPHABET_LOWER:
                    return_words.add(left+c+right)
        
        return return_words
    
    all_edits = set()

    for func in [insert, delete, replace]:
        conc_func = concatenate_function(func, x)
        all_edits = all_edits | conc_func({s})

    for ix,c in enumerate(s):
        all_edits = all_edits | {s[:ix]+c.swapcase()+s[ix+1:]}

    # Common Portuguese errors --- decided to keep these for nheengatu as well
    # ss and ç
    all_edits.add(s.replace('ss', 'ç'))
    all_edits.add(s.replace('ç', 'ss'))

    # ão and am
    all_edits.add(s.replace('ão', 'am'))
    all_edits.add(s.replace('am', 'ão'))

    # Common Nheengatu errors
    # include invalid chars
    # from utils import NHEENGATU_TO_REPLACE
    # for wrong, correct in NHEENGATU_TO_REPLACE.items():
    #     all_edits.add(s.replace(correct, wrong))
    # # TODO - check with nheengatu speakers what are the most common errors
    
    all_edits.discard(s)
    
    return all_edits
    


def get_possible_mistakes(word, dictionary, just_similar = False):
    if word not in dictionary:
       return []
    similar_words = get_similar_strings(word)

    if just_similar:
        return similar_words

    mistakes = []

    for similar in similar_words:
        if dictionary.get(similar, 10_000_000) <= (dictionary[word]/5):
            mistakes.append(similar)
    mistakes.sort(key=lambda x: dictionary[x], reverse=True)

    return mistakes

def draw_random_number(left: int, right: int):
    """
    Requires left <= right.
    Returns a random number in the inverval (left, right).
    """
    num = random.random() * (right-left)
    num = round(num)
    return num+left 

def draw_random_quantity():
    qtd_array = [0,0,0,1,1,1,1,2,2,2,2,2,2,3,3,3,4]
    random_index = draw_random_number(0,len(qtd_array)-1)
    return qtd_array[random_index]


def generate_deterministic_mistakes(sentences, last_token=True):
    def add_to_data(wrong, right):
        if (wrong,right) in duplicates:
            return 
        duplicates.add((wrong,right))
        wrong_text.append(wrong)
        correct_text.append(right)
    
    #generation_type = 'full_sentence' # 'detected_tokens'
    generation_type = 'detected_tokens'
    
    df = {'wrong_text': [], 'correct_text': []}
    
    # aliases
    wrong_text = df['wrong_text']
    correct_text = df['correct_text']
    duplicates = set()
    
    number_repeats = 1
    max_mistakes = 1
    skip_long_sentences = False
    do_random_sampling = False
    just_strip_accents = True
    
    for s in tqdm(sentences):    
        all_words = list(tokenize_nheengatu(s))
        
        if skip_long_sentences and len(all_words) > 10: 
            continue
        
        # if there are no words, there's nothing to corrupt
        if len(all_words) == 0:
            continue

        all_words = ['<BOS>'] + [ word for word in all_words ] + ['<EOS>']

        if last_token:
            for i,word in enumerate(all_words[2:-1]):
                # get previous, current, and next words
                p_word1 = all_words[i]
                p_word2 = all_words[i+1]
                mistake = unidecode( word )

                if word.isalnum():
                    add_to_data(" ".join([p_word1, p_word2, mistake]), word)
        
        else:
            for i,word in enumerate(all_words[1:-1]):
                # get previous, current, and next words
                p_word = all_words[i]
                mistake = unidecode( word )
                n_word = all_words[i+2]

                if n_word.isalnum():
                    add_to_data(" ".join([p_word, mistake, n_word]), word)


            ## TODO: create triplets with mistakes in p_word and n_word

    return df

## This is the function used to create samples from sentences
# A list of sentences is provided in **texts**, and each sentence is broken down into sequences of tokens with **MIN_LENGTH** up to **MAX_LENGTH**
# The use of bigrams, in addition to the individual tokens, is controlled with **do_bigram**
# The function returns a list of samples with the corresponding label, which is the next word following the extracted sequence
def tokens2samples(texts, MIN_LENGTH=2, MAX_LENGTH=-1, do_bigram=False):

    X = []
    Y = []
    vocabulary = {}

    for text in tqdm(texts):
        
        tokens = tokenize_nheengatu(text.lower())
        
        if MAX_LENGTH < 0:
            MAX_LENGTH = len(tokens)


        if len(tokens) > MIN_LENGTH:
            for i in range(len(tokens)-1, MIN_LENGTH-1, -1):

                for j in range(i-MIN_LENGTH, i-MAX_LENGTH-1, -1):
                    sample = tokens[j:i]
                                        
                    if len(sample) > 0 and len(tokens[i]) > 0 and tokens[i].isalpha():
                        X.append(sample)
                        Y.append(tokens[i])
                    
    return X, Y


def generate_nextword_samples(sentences):

    
    df = {'input_tokens': [], 'next_token': []}
    
    # aliases
    input_text = df['input_tokens']
    next_token = df['next_token']
    

    X, Y = tokens2samples( sentences, MAX_LENGTH=5 )

    input_text += [ " ".join(x) for x in X]
    next_token += [ y for y in Y]

    return df


###### END AUXILIARY FUNCTIONS
#########################################


def main():
    parser = argparse.ArgumentParser(description="Generates a seq2seq dataset to train a spell checker.")
    parser.add_argument("-i", required=True, help="Name of the input file")
    parser.add_argument("-o", required=True, help="Name of the output file")
    args = parser.parse_args()


    input_file = args.i
    print(f"The provided input file is: {input_file}")
    output_file = args.o
    print(f"The provided input file is: {output_file}")

    df = pd.read_excel(input_file)
    raw_texts = list(df['yrl'])
    all_texts = normalize_texts(raw_texts)
    sentences = [s for s in split_sentences(all_texts) if len(s) > 1]
    random.shuffle(sentences)
    sentences = apply_and_concatenate(normalize_sentence, sentences)
    sentences = de_duplicate(sentences)

    dictionary = build_dictionary(sentences)

    # generate dataset with mistakes
    df_autocorrect = generate_deterministic_mistakes(sentences)
    df_autocorrect = pd.DataFrame(df_autocorrect)
    df_autocorrect = df_autocorrect.rename(columns={'wrong_text': 'input', 'correct_text': 'output'})
    df_autocorrect['input'] = 'autocorrect ' + df_autocorrect['input']

    # generate dataset for nextword
    df_nextword = generate_nextword_samples(sentences)
    df_nextword = pd.DataFrame(df_nextword)
    df_nextword = df_nextword.rename(columns={'input_tokens': 'input', 'next_token': 'output'})
    df_nextword['input'] = 'nextword ' + df_nextword['input']

    # generate dataset for translation
    df_translation2lang1 = df.rename(columns={'por': 'input', 'yrl': 'output'})
    df_translation2lang1['input'] = 'translate portuguese to nheengatu: ' + df_translation2lang1['input']

    df_translation2lang2 = df.rename(columns={'yrl': 'input', 'por': 'output'})
    df_translation2lang2['input'] = 'translate nheengatu to portuguese: ' + df_translation2lang2['input']

    df_results = pd.concat([df_autocorrect, df_nextword, df_translation2lang1, df_translation2lang2])


    # save to a CSV file
    df_results.to_csv(output_file, header=True, index=False)

if __name__ == "__main__":
    main()