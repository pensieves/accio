from torchtext.data import Field

class RevField(Field):
    f"""Field class based on the torchtext's ReversibleField class with complete 
    independency from revtok and bug fix of no spaces between tokens while returning 
    from reverse method."""
    
    def reverse(self, batch):
        
        if not self.batch_first:
            batch = batch.t()
        
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]
        
        return [' '.join(ex) for ex in batch]