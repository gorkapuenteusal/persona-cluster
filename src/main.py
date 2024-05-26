import os
import argparse
import logging
from text_clustering import TextClustering
from train import train_from_files
from utils import setup_logging


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Text Clustering and Prediction")
    parser.add_argument('--model', type=str, help='Nombre del archivo del modelo a cargar (sin extensión)')
    parser.add_argument('--train', nargs='*', help='Archivos de entrenamiento')
    parser.add_argument('--threshold', type=float, help='Threshold para el clustering')
    args = parser.parse_args()

    if (args.model or args.train) and args.threshold is None:
        parser.error('--threshold es obligatorio cuando se usa --model o --train')

    if args.model and not args.train:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', f"{args.model}.pkl")
        if os.path.exists(filepath):
            text_clustering = TextClustering.load_model(filepath)
            text_clustering.threshold = args.threshold  # Set threshold for inference
            logging.info("Threshold para inferencia: %s", args.threshold)
            print(f"Modelo {args.model} cargado.")
            while True:
                text = input("Introduce un texto (o 'exit' para terminar): ")
                if text.lower() == 'exit':
                    break
                person = text_clustering.predict(text)
                print(f"El texto parece ser de: {person}")
        else:
            print(f"El archivo {filepath} no existe.")
    elif args.train:
        train_files = [os.path.join(os.path.dirname(__file__), '..', 'data', 'train', file) for file in args.train]
        text_clustering = train_from_files(train_files, threshold=args.threshold)
        filename = input("Introduce el nombre del archivo para guardar el modelo (sin extensión): ")
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', f"{filename}.pkl")
        text_clustering.save_model(filepath)
    else:
        text_clustering = TextClustering(threshold=3.)
        logging.info("Threshold para entrenamiento manual: %s", args.threshold)
        while True:
            text = input("Introduce un texto (o 'exit' para terminar): ")
            if text.lower() == 'exit':
                filename = input("Introduce el nombre del archivo para guardar el modelo (sin extensión): ")
                filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', f"{filename}.pkl")
                text_clustering.save_model(filepath)
                break
            person = text_clustering.add_text(text)
            print(f"El texto parece ser de: {person}")


if __name__ == "__main__":
    main()
