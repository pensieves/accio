from torchtext.data import ReversibleField

class RevField(ReversibleField):
    
    def __init__(self, **kwargs):
        
        if kwargs.get("tokenize") == "revtok":
            self.use_revtok = True
            if "unk_token" not in kwargs:
                kwargs["unk_token"] = " UNK "
        
        super(ReversibleField, self).__init__(**kwargs)