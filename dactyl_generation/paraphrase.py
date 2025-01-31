# Use a pipeline as a high-level helper
from tqdm import tqdm




def paraphrase(paraphrase_pipeline, text, temperature, top_p, max_new_tokens):
    return paraphrase_pipeline(f"Paraphrase this: {text}", do_sample=True, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)[0]["generated_text"]


def paraphrase_texts(queries, paraphrase_pipeline):
    paraphrased_texts = list()
    def data():
        for i in range(len(queries)):
            print(queries[i])
            yield queries[i]

    for result in tqdm(paraphrase_pipeline(**dict(data()), batch_size=4), total=len(queries)):
        paraphrased_texts.append(result[0]["generated_text"])

    return paraphrased_texts
