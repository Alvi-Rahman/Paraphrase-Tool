from transformers import T5ForConditionalGeneration, T5Tokenizer
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, HttpResponseRedirect
from django.conf import settings
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
from django.views import View

class ParaphraseTool:
    def __init__(self):
        self.set_seed(42)
        self.__model, self.__tokenizer, self.__device = self.load_model()
    
    def get_model(self):
        return self.__model
    
    def get_tokenizer(self):
        return self.__tokenizer

    def get_device(self):
        return self.__device

    def set_seed(self,seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    def load_model(self):
        model = T5ForConditionalGeneration.from_pretrained(os.path.join(
            settings.BASE_DIR, 'paraphrase_utils', 'model'))
        tokenizer = T5Tokenizer.from_pretrained(os.path.join(
            settings.BASE_DIR, 'paraphrase_utils', 'tokenizer'))   
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        return model, tokenizer, device

    def paraphrase(self, sentence):
        text = sentence

        max_len = 64

        encoding = self.__tokenizer.encode_plus(text, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(
            self.__device), encoding["attention_mask"].to(self.__device)

        beam_outputs = self.__model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            min_length=max(len(text.split()) - 5 , 5),
            max_length=len(text.split()) + 10,
            top_k=120,
            top_p=0.95,
            temperature=0.98,
            early_stopping=True,
            num_return_sequences=1,
            no_repeat_ngram_size=5
        )

        final_outputs = []
        for beam_output in beam_outputs:
            sent = self.__tokenizer.decode(
                beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if sent.lower() != sentence.lower() and sent not in final_outputs:
                final_outputs.append(sent)

        # for i, final_output in enumerate(final_outputs):
        #     print("{}".format(final_output))
        return final_outputs[0]
                
        pass


class HomePage(View):
    template_name = "paraphrase_app/index.html"
    paraphrase_class = ParaphraseTool()
    
    def get(self, request):
        return render(request, "paraphrase_app/index.html", context={'get':1})

    def post(self, request):
        real_text = request.POST['paraphrase_content']
        if len(real_text) < 10:
            paraphrased_data = None
        else:
            paraphrased_data = self.paraphrase_class.paraphrase(real_text)
        context = {
            'real_text': real_text,
            'paraphrased_data': paraphrased_data,
        }
        return render(request, "paraphrase_app/index.html", context=context)

# def home_page(request):
#     if request.method == 'GET':
#         return render(request, "paraphrase_app/index.html")
#     elif request.method == 'POST':
#         real_text = request.POST['paraphrase_content']
#         if len(real_text) < 10:
#             paraphrased_data = None
#         else:
#             paraphrased_data = os.listdir(os.path.join(
#                 settings.BASE_DIR, 'paraphrase_utils', 'model'))
#         context = {
#             'real_text': real_text,
#             'paraphrased_data': paraphrased_data,
#         }
#         return render(request, "paraphrase_app/index.html", context=context)

