import torch
from transformers import CamembertForSequenceClassification, CamembertTokenizer

class Sentiment:

    def __init__(self, device, max_len) -> None:
        self.model = CamembertForSequenceClassification.from_pretrained(
            'camembert-base',
            state_dict = torch.load(
                'sentiments.pt',
                map_location=torch.device('cpu')
                )
        )
        self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base',do_lower_case=True)
        self.max_len = max_len
        self.device = device

    def predict(self, text):
        encoded_text = self.tokenizer.encode_plus( text,
                                                    truncation=True, 
                                                    max_length=self.max_len, 
                                                    return_token_type_ids=False,
                                                    padding="max_length", 
                                                    return_attention_mask=True, 
                                                    return_tensors = 'pt'
                                                    )
        with torch.no_grad():
            self.model.eval()
        
            input_ids = encoded_text['input_ids'].to(self.device)
            attention_mask = encoded_text['attention_mask'].to(self.device)

            output = self.model(input_ids, attention_mask=attention_mask)
            
            probabilites = torch.softmax(output.logits, dim=1).tolist()
        
        return probabilites[0]