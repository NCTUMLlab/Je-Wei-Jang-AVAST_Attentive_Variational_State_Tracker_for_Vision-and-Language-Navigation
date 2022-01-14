import numpy as np
from tqdm import tqdm


class Glove():
    def __init__(
        self,
        path: str
    ) -> None:
        super().__init__()
        self._words, self._word2idx, self._vectors, self._glove = [], {}, [], {}
        self._load(path)
        self.feature_dim = len(self._vectors[0])
        return

    def _load(
        self,
        path
    ) -> None:
        idx, words, word2idx, vectors = 0, [], {}, []
        with open(path, 'rb') as file_name:
            with tqdm(total=sum(1 for _ in file_name)) as pbar:
                file_name.seek(0)
                for line in file_name:
                    line_decode = line.decode().split()
                    word = line_decode[0]
                    words.append(word)
                    word2idx[word] = idx
                    idx += 1
                    vect = np.array(line_decode[1:]).astype(np.float)
                    vectors.append(vect)
                    pbar.update(1)
        self.set_words(words)
        self.set_word2idx(word2idx)
        self.set_vectors(vectors)
        self.set_glove({w: vectors[word2idx[w]] for w in words})
        return

    def w2v(
        self,
        word: str
    ) -> np.ndarray:
        return self._glove[word]

    @property
    def words(
        self
    ) -> list:
        return self._words

    def set_words(
        self,
        new_words: list
    ) -> None:
        self._words = new_words
        return

    @property
    def word2idx(
        self
    ) -> dict:
        return self._word2idx

    def set_word2idx(
        self,
        new_word2idx: dict
    ) -> None:
        self._word2idx = new_word2idx
        return

    @property
    def vectors(
        self
    ) -> list:
        return self._vectors

    def set_vectors(
        self,
        new_vectors: list
    ) -> None:
        self._vectors = new_vectors
        return

    @property
    def glove(
        self
    ) -> dict:
        return self._glove

    def set_glove(
        self,
        new_glove: dict
    ) -> None:
        self._glove = new_glove
        return


def main():
    return


if __name__ == '__main__':
    main()
