from gensim.models.word2vec import Word2Vec
import pickle
import os


def data_process(path="/dataset/poetry_data.txt", train_num=10000):
    with open(path, "r", encoding="utf-8") as f:
        naive_data = f.readlines()[:train_num]
        data = "".join(naive_data)
        data_split = " ".join(data)
    with open("./poetry_split.txt", "w", encoding="utf-8") as f:
        f.write(data_split)
    return data_split.strip(), naive_data


def get_wordvec(sentences, embidding_size=128):
    model = Word2Vec(
        sentences=sentences,
        vector_size=embidding_size,
        min_count=1,
    )
    if os.path.exists("Word2Vec.pkl"):
        return pickle.load(open("Word2Vec.pkl", "rb"))
    with open("Word2Vec.pkl", "wb") as f:
        pickle.dump((model.syn1neg, model.wv.key_to_index,
                    model.wv.index_to_key), f)
    return model.syn1neg, model.wv.key_to_index, model.wv.index_to_key
