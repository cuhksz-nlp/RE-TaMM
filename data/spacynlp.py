import spacy

class DepTreeParser():
    def __init__(self):
        pass

    def parsing(self, sentence):
        pass

class SpaCyDepTreeParser(DepTreeParser):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def parsing(self, sentence):
        doc = self.nlp(sentence)
        basicDependencies = []
        tokens = []
        for i, token in enumerate(doc):
            if token.dep_ == "ROOT":
                basicDependencies.append({
                    "dep": token.dep_,
                    "governor": 0,
                    "governorGloss": "ROOT",
                    "dependent": token.i+1,
                    "dependentGloss": token.text
                })
            else:
                basicDependencies.append({
                    "dep": token.dep_,
                    "governor": token.head.i+1,
                    "governorGloss": token.head.text,
                    "dependent": token.i+1,
                    "dependentGloss": token.text
                })
            tokens.append({
                "index": token.i,
                "word": token.text,
                "originalText": token.text
            })
        return {
            "sentences": [
                {
                    "index": 0,
                    "line": 1,
                    "basicDependencies": basicDependencies,
                    "tokens": tokens
                }
            ]
        }
