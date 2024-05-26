import logging
import pickle
from collections import defaultdict, Counter

import numpy as np
from scipy.spatial.distance import euclidean


def preprocess_text(text):
    words = text.lower().split()
    return words


def calculate_ngrams(words, n):
    return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]


def calculate_centroid(cluster):
    if not cluster:
        return {}
    total = Counter()
    for freq_dict in cluster:
        total.update(freq_dict)
    centroid = {word: count / len(cluster) for word, count in total.items()}
    return centroid


class TextClustering:
    def __init__(self, threshold=0.5):
        self.clusters = defaultdict(list)
        self.centroids = {}
        self.threshold = threshold
        self.vocabulary = set()
        logging.info("Modelo TextClustering inicializado con umbral %s", threshold)

    def calculate_frequencies(self, words):
        unigrams = words
        bigrams = calculate_ngrams(words, 2)
        trigrams = calculate_ngrams(words, 3)

        all_ngrams = unigrams + bigrams + trigrams
        frequencies = Counter(all_ngrams)
        self.vocabulary.update(frequencies.keys())
        return frequencies

    def vectorize_frequencies(self, frequencies):
        vector = np.array([frequencies.get(word, 0) for word in self.vocabulary])
        return vector

    def update_centroid(self, name):
        self.centroids[name] = calculate_centroid(self.clusters[name])

    def add_text(self, text, name=None):
        words = preprocess_text(text)
        frequencies = self.calculate_frequencies(words)
        vectorized_frequencies = self.vectorize_frequencies(frequencies)

        logging.info("Top 5 palabras/ngramas más repetidas: %s", frequencies.most_common(5))

        if not self.centroids:
            if not name:
                name = input("Introduce el nombre de la persona: ")
            self.clusters[name].append(frequencies)
            self.update_centroid(name)
            return name
        else:
            distances = {name: euclidean(vectorized_frequencies, self.vectorize_frequencies(self.centroids[name]))
                         for name in self.centroids}
            logging.info("Distancias a los centroides: %s", distances)

            closest_name = min(distances, key=distances.get)
            if distances[closest_name] < self.threshold:
                self.clusters[closest_name].append(frequencies)
                self.update_centroid(closest_name)
                logging.info("Texto añadido al cluster '%s'", closest_name)
                return closest_name
            else:
                manual_clustering = not name
                if manual_clustering:
                    name = input("Introduce el nombre de la persona: ")
                self.clusters[name].append(frequencies)
                self.update_centroid(name)
                logging.info("Texto añadido al cluster '%s'%s", name,
                             "" if manual_clustering else ", aunque no estuviese dentro del mínimo")
                return name

    def predict(self, text):
        words = preprocess_text(text)
        frequencies = self.calculate_frequencies(words)
        vectorized_frequencies = self.vectorize_frequencies(frequencies)

        distances = {name: euclidean(vectorized_frequencies, self.vectorize_frequencies(self.centroids[name]))
                     for name in self.centroids}
        logging.info("Distancias calculadas: %s", distances)
        closest_name = min(distances, key=distances.get)
        if distances[closest_name] < self.threshold:
            logging.info("Texto identificado como '%s' con una distancia de %s", closest_name, distances[closest_name])
            return closest_name
        else:
            logging.info("Texto no reconocido. Distancia mínima: %s", distances[closest_name])
            return "Persona no reconocida"

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        logging.info("Modelo guardado en %s", filename)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        logging.info("Modelo cargado desde %s", filename)
        return model
