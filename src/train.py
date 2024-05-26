import os
import logging
from text_clustering import TextClustering


def train_from_files(files, threshold):
    text_clustering = TextClustering(threshold=threshold)
    for filepath in files:
        name = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
            for line in text.strip().split('\n'):
                if line.strip():
                    text_clustering.add_text(line, name=name)
                    logging.info("Texto añadido al entrenamiento del archivo %s", filepath)
    return text_clustering
