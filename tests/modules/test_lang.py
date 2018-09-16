from snippets.modules.sequence import Lang
import unittest
import torch


class TestLang(unittest.TestCase):
    def testLang(self):
        lang = Lang(name="name")
        lang.add_sentences([
            "a,b,c,d",
            "c,e,f,g"
        ], tokenizer=lambda x: x.split(","))
        lang.add_sentences([
            "e g h j",
        ])
        self.assertListEqual(lang.tensor_to_tokens(torch.tensor([2])), ["a"])
        self.assertListEqual(lang.sentence_to_tensor("a b").tolist(), [2, 3, lang.EOS_INDEX])
