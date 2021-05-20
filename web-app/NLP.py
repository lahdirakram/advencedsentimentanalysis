import torch
from sklearn import metrics
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



"""
L’objectif de cet article consiste à identifier les enjeux de la différenciation selon le genre dans l’accompagnement collectif de la femme potentiellement créatrice. 
A partir d’une démarche abductive, nous montrons que la mise en place de collectifs de femmes, dans la phase ante-création, est génératrice de confiance en soi. 
Le caractère bienveillant, empathique et non compétitif du groupe exclusivement féminin facilite, pour certaines femmes candidates à la création, la déconstruction de stéréotypes 
de genre sur l’entrepreneuriat féminin et atténue le sentiment de culpabilité lié à la difficulté à concilier projet entrepreneurial et vie familiale. En d’autres termes, 
la participation de la femme potentiellement créatrice à un collectif féminin répond à un besoin psychologique d’approbation externe qui peut faire défaut, notamment 
au sein de la famille. Toutefois, la pratique de l’accompagnement collectif au féminin dans la phase <em>ante</em>-création présente certaines limites. Ces dernières 
résultent de l’importance accordée par certaines femmes à la mise en place d’un environnement mixte pour le partage d’expériences entrepreneuriales diverses et le 
développement de réseaux d’affaires."""