import itertools
from typing import Dict, Union

from nltk import sent_tokenize
import nltk
nltk.download('punkt')
import torch
from transformers import(
    AutoModelForSeq2SeqLM, 
    AutoTokenizer
)

class QGPipeline:

    def __init__(
        self
    ):
      
        self.model = AutoModelForSeq2SeqLM.from_pretrained("muchad/idt5-qa-qg")
        self.tokenizer = AutoTokenizer.from_pretrained("muchad/idt5-qa-qg")
        self.qg_format = "highlight"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.ans_model = self.model
        self.ans_tokenizer = self.tokenizer
        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration"]
        self.model_type = "t5"


    def __call__(self, inputs: str):
        inputs = " ".join(inputs.split())
        sents, answers = self._extract_answers(inputs)
        flat_answers = list(itertools.chain(*answers))

        if len(flat_answers) == 0:
          return []

        qg_examples = self._prepare_inputs_for_qg_from_answers_hl(sents, answers)        
        qg_inputs = [example['source_text'] for example in qg_examples]
        questions = self._generate_questions(qg_inputs)
        output = [{'answer': example['answer'], 'question': que} for example, que in zip(qg_examples, questions)]
        return output
    
    def _generate_questions(self, inputs):
        inputs = self._tokenize(inputs, padding=True, truncation=True)
        
        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=80,
            num_beams=4,
        )
        
        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return questions
    
    def _extract_answers(self, context):
        sents, inputs = self._prepare_inputs_for_ans_extraction(context)
        
        inputs = self._tokenize(inputs, padding=True, truncation=True)

        outs = self.ans_model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=80,
        )
        
        dec = [self.ans_tokenizer.decode(ids, skip_special_tokens=True) for ids in outs] 
        answers = [item.split('<sep>') for item in dec]
        answers = [i[:-1] for i in answers]
        return sents, answers
    
    def _tokenize(self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True, 
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs
    
    def _prepare_inputs_for_ans_extraction(self, text):
        sents = sent_tokenize(text)

        inputs = []
        for i in range(len(sents)):
            source_text = "extract answers:"
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "<hl> %s <hl>" % sent
                source_text = "%s %s" % (source_text, sent)
                source_text = source_text.strip()
            
            source_text = source_text + " </s>"
            inputs.append(source_text)
        return sents, inputs
    
    def _prepare_inputs_for_qg_from_answers_hl(self, sents, answers):
        inputs = []
        for i, answer in enumerate(answers):
            if len(answer) == 0: continue
            for answer_text in answer:
                sent = sents[i]
                sents_copy = sents[:]
                
                answer_text = answer_text.strip()
                try:
                  ans_start_idx = sent.index(answer_text)
                  
                  sent = f"{sent[:ans_start_idx]} <hl> {answer_text} <hl> {sent[ans_start_idx + len(answer_text): ]}"
                  sents_copy[i] = sent
                  
                  source_text = " ".join(sents_copy)
                  source_text = f"generate question: {source_text}" 
                  if self.model_type == "t5":
                      source_text = source_text + " </s>"
                except:
                  continue
                
                inputs.append({"answer": answer_text, "source_text": source_text})
        
        return inputs
    
class TaskPipeline(QGPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, inputs: Union[Dict, str]):
        return super().__call__(inputs)
       
def pipeline():
    task = TaskPipeline
    return task()