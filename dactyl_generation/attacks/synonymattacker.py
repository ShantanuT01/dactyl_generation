import numpy as np
import shap
import polars as pl


class SynonymAttacker:
    def __init__(self, classifier_pipeline, mlm_model, space_char="Ġ"):
        self.__mlm_model = mlm_model
        self.__tokenizer = classifier_pipeline.tokenizer
        self.__space_char = space_char
        self.__pipeline = classifier_pipeline

    def attack_text(self, tokens_and_scores, max_edits=10, cutoff_score=0.5):
        index_order = tokens_and_scores[:, 1].argsort()[::-1][0:min(max_edits, len(tokens_and_scores))]
        # print(index_order)
        # [0:min(max_edits, len(tokens_and_scores))]
        old_string = self.__tokenizer.convert_tokens_to_string(tokens_and_scores[:, 0])
        initial_baseline = self.__pipeline(old_string)
        min_score = pl.DataFrame(initial_baseline[0]).filter(pl.col("label") == "ai").get_column("score").to_list()[0]
        if min_score < cutoff_score:
            return old_string

        print(self.__tokenizer.convert_tokens_to_string(tokens_and_scores[:, 0]), min_score)
        for index in index_order:

            old_string = self.__tokenizer.convert_tokens_to_string(tokens_and_scores[:, 0])

            token = tokens_and_scores[index, 0]
            shapley_val = tokens_and_scores[index, 1]

            if shapley_val < 0:
                break
            space_index = token.find(self.__space_char)
            rest_of_string = token[space_index + 1:]
            tokens_and_scores[index, 0] = token.replace(rest_of_string, "[MASK]")

            mask_string = self.__tokenizer.convert_tokens_to_string(tokens_and_scores[:, 0])
            outputs = self.__mlm_model(mask_string)
            possibilities = pl.DataFrame(outputs)

            possibilities = possibilities.filter(pl.col("sequence") != old_string)
            scores = self.__pipeline(possibilities.get_column("sequence").to_list())
            best_token = ""
            for i in range(len(scores)):
                new_score = pl.DataFrame(scores[i]).filter(pl.col("label") == "ai").get_column("score").to_list()[0]
                if new_score < min_score:
                    best_token = possibilities.get_column("token_str").to_list()[i]
                    min_score = new_score
            # print("Best:", best_token,  min_score)
            if best_token == "":
                tokens_and_scores[index, 0] = token
            else:
                tokens_and_scores[index, 0] = best_token

            print(self.__tokenizer.convert_tokens_to_string(tokens_and_scores[:, 0]), min_score)
            if min_score < cutoff_score:
                break

        return self.__tokenizer.convert_tokens_to_string(tokens_and_scores[:, 0])
