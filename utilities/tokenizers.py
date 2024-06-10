import json

class CharacterTokenizer:
    def __init__(self,pad_token='🚣',unk_token='❓'):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self._token_to_id={self.pad_token:0,self.unk_token:1}
        self._id_to_token={0:self.pad_token,1:self.unk_token}
        
    def fit(self,text:str):
        assert len(self._token_to_id)==len(self._id_to_token)
        vocabulary=sorted(list(set(text)))

        for id,character in enumerate(vocabulary,start=len(self._token_to_id)):
            self._token_to_id[character]=id
            self._id_to_token[id]=character

    def encode(self,sentence:str):
        return [self._token_to_id.get(character,self._token_to_id[self.unk_token]) for character in sentence]
        
    def save(self, filepath: str):
        tokenizer_data = {
            'token_to_id': self._token_to_id,
            'id_to_token': self._id_to_token
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False)

    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        self._token_to_id = tokenizer_data['token_to_id']
        self._id_to_token = {int(k): v for k, v in tokenizer_data['id_to_token'].items()}

    @property
    def vocab_size(self):
        return len(self._token_to_id)
