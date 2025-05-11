from nltk.tokenize import sent_tokenize
import transformers
import pandas as pd
import numpy as np

REWRITTEN_PROMPTS = ["Fix the grammar: ", "Make this text coherent: ", "Rewrite to make this easier to understand: ",
                     "Paraphrase this: ", "Write this more formally: ", "Write in a more neutral way: "]


class ParaphraseAttacker:
    def __init__(self, paraphraser_pipeline: transformers.Pipeline, detector_pipeline: transformers.Pipeline,
                 detector_ai_label):
        """

        Args:
            paraphraser_pipeline:
            detector_pipeline:
            detector_ai_label:
        """
        self.__paraphraser = paraphraser_pipeline
        self.__detector = detector_pipeline
        self.__ai_label = detector_ai_label

    def score_text(self, text: str) -> float:
        scores = self.__detector(text, max_length=512, truncation=True, top_k=None)
        score_frame = pd.DataFrame(scores)
        return score_frame[score_frame.label == self.__ai_label]["score"].values[0]

    def attack_text(self, text: str, difference_score=0.5):
        sentences = sent_tokenize(text)
        added_sentences = list()
        current_ai_score = self.score_text(text)
        for sentence in sentences:
            added_sentences.append(sentence)
            ai_score = self.score_text(" ".join(added_sentences))
            lowest_ai_score = ai_score
            best_sentence = sentence

            prompts_to_check = [f"{prompt}{sentence}" for prompt in REWRITTEN_PROMPTS]

            # batch function for rewriting
            def rewritten_prompts():
                for prompt in prompts_to_check:
                    yield prompt

            outputted_texts = list()
            for result in self.__paraphraser(rewritten_prompts(), max_new_tokens=128, do_sample=True, top_k=60,
                                             temperature=1):
                outputted_texts.append(result[0]["generated_text"])

            # batch function for fixing grammar
            fix_grammar_prompts = [f"{REWRITTEN_PROMPTS[0]}{new_sentence}" for new_sentence in outputted_texts]

            def fix_grammar():
                for prompt in fix_grammar_prompts:
                    yield prompt

            new_sentences = list()
            for result in self.__paraphraser(fix_grammar(), max_new_tokens=128):
                new_sentences.append(result[0]["generated_text"].strip())

            # batch sentences for predictions
            batch_texts = list()
            for new_sentence in new_sentences:
                added_sentences[-1] = new_sentence
                batch_texts.append(" ".join(added_sentences))

            def get_new_texts():
                for new_text in batch_texts:
                    yield new_text

            ai_scores = list()
            for result in self.__detector(get_new_texts(), max_length=512, truncation=True, top_k=None):
                score_frame = pd.DataFrame(result)
                ai_scores.append(score_frame[score_frame.label == self.__ai_label]["score"].values[0])

            min_ai_score_index = np.argmin(ai_scores)
            new_lowest_ai_score = ai_scores[min_ai_score_index]
            # only update if revised sentences more human compared to previous
            if new_lowest_ai_score < lowest_ai_score:
                lowest_ai_score = new_lowest_ai_score
                best_sentence = new_sentences[min_ai_score_index]

            added_sentences[-1] = best_sentence

            if (lowest_ai_score > current_ai_score) and (lowest_ai_score - current_ai_score) >= difference_score:
                # remove sentence if it somehow contributes to increasing chances of detection
                added_sentences.pop()
            else:
                current_ai_score = lowest_ai_score

        return " ".join(added_sentences)

