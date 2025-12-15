import os
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

from prepare_dataset import *

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lang_map = { 'yrl': 'nheengatu', 'por': 'portuguese'}

#####################
class mBARTWrapper():
    def __init__(self, direction='portuguese to nheengatu'):
        # Paths and names
        self.__MODEL_NAME = "mbart_all_defaults"
        self.__MODELS_DIR = \
            os.path.join(os.getcwd(), "../data/models")

        self.__model = MBartForConditionalGeneration.from_pretrained(
                os.path.join(self.__MODELS_DIR, self.__MODEL_NAME))
        
        self.__tokenizer = MBart50Tokenizer.from_pretrained(
                os.path.join(self.__MODELS_DIR, self.__MODEL_NAME))

        self.__valid_tasks = ['translation', 'nextword', 'autocorrect']
        
        self.__valid_arguments = {
            'translation': {'src_lang', 'tgt_lang'}
        }


    def _format_prefix(task, **kwargs):
        assert task in self.__valid_tasks, f"Invalid task: {task}. Must be one of {self.__valid_tasks}"

        if task in self.__valid_arguments:
            for k, v in kwargs.items():
                assert k in self.__valid_arguments[task], f"Invalid argument: {k}. Must be one of {self.__valid_arguments.keys()}"

        if task == "translation":
            return f'translate {lang_map[kwargs['src_lang']]} to {lang_map[kwargs['tgt_lang']]}:'
        else:
            return task


    def _generate_autocorrect(self, sentence):
        prefix = 'autocorrect '

        all_words = list(re.finditer(rf'[{lower_case+upper_case}]+', sentence))

        if len(all_words) < 2:
            output['error'] = 'not enough tokens'
        
        else:
            all_words = ['<BOS>'] + [ word.group() for word in all_words ] + ['<EOS>']
            tokenizer = self.__tokenizer
            model = self.__model
            
            output_words = [ all_words[1] ]
            for i,word in enumerate(all_words[2:-1]):
                # get previous, current, and next words
                p_word1 = all_words[i]
                p_word2 = all_words[i+1]
                mistake = word

                input_text = prefix + " ".join([p_word1, p_word2, mistake])
                input_ids = tokenizer(input_text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
                #print(model(input_ids=input_ids))

                input_ids = tokenizer.encode(input_text, return_tensors="pt")
                output_ids = model.generate(input_ids)
                corrected_word = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                output_words.append( corrected_word )

            corrected_words = " ".join( output_words )

        return corrected_words

    def _generate_nextword(self, sentence):
        sentence = sentence.lower()

        prefix = 'nextword '
        input_ids = self.__tokenizer.encode(
            prefix + sentence,
            return_tensors="pt").to(device)
        output_ids = self.__model.generate(input_ids)
        next_word = self.__tokenizer.decode(
            output_ids[0], skip_special_tokens=True)

        return next_word


    def _generate_translation(self, sentence, src_lang, tgt_lang):
        prefix = f"translate {lang_map[src_lang]} to {lang_map[tgt_lang]}: "

        sentence = sentence.lower()
        input_ids = self.__tokenizer.encode(
            prefix + sentence,
            return_tensors="pt").to(device)
        output_ids = self.__model.generate(input_ids)
        translated_sentence = self.__tokenizer.decode(
            output_ids[0], skip_special_tokens=True)

        return translated_sentence

        
    def generate(self, sentence: str, task: str, kwargs: dict = None) -> str:
        assert task in self.__valid_tasks, f"Invalid task: {task}. Must be one of {self.__valid_tasks}"

        if task in self.__valid_arguments:
            for k, v in kwargs.items():
                assert k in self.__valid_arguments[task], f"Invalid argument: {k}. Must be one of {self.__valid_arguments.keys()}"

        if task == 'autocorrect':
            return self._generate_autocorrect(sentence)

        elif task == 'nextword':
            return self._generate_nextword(sentence)

        elif task == 'translation':
            return self._generate_translation(sentence, kwargs['src_lang'], kwargs['tgt_lang'])
