# -*- coding: utf-8 -*-
"""
src/data_processing/loader.py

This module is responsible for loading and preprocessing all datasets required for the project.
It handles fetching data from various online sources (like Hugging Face Datasets) and provides
a fallback mechanism with local data to ensure robustness.
"""

import random
from typing import List, Dict

# Import project-specific modules
from src import config

class RealDatasetLoader:
    """
    Loads standard kNN-MT datasets (WMT19, Multi30K, OPUS, etc.) from online sources
    and local fallbacks. It also performs quality filtering and deduplication.
    """

    def __init__(self):
        """Initializes the dataset loader, pulling parameters from the config file."""
        print("[INFO] Initializing Real Dataset Loader...")
        loader_params = config.DATA_LOADER_PARAMS
        self.max_samples_total = loader_params["max_samples_total"]
        self.max_samples_per_dataset = loader_params["max_samples_per_dataset"]
        print(f"\t- Target total samples: {self.max_samples_total}")
        print(f"\t- Max samples per dataset: {self.max_samples_per_dataset}")

    def load_opus_datasets(self) -> List[Dict[str, str]]:
        """Loads datasets from the OPUS collection."""
        samples = []
        opus_configs = [
            ("Helsinki-NLP/opus-100", "de-en"),
            ("tatoeba", "de-en"),
        ]

        for dataset_name, lang_pair in opus_configs:
            try:
                print(f"\t[LOADING...] Trying to load dataset \"{dataset_name}\"...")
                from datasets import load_dataset
                dataset = load_dataset(dataset_name, lang_pair, split="train", streaming=False)

                count = 0
                for example in dataset:
                    if count >= self.max_samples_per_dataset or len(samples) >= self.max_samples_total:
                        break
                    
                    if 'translation' in example:
                        translation = example['translation']
                        de_text = translation.get('de', '').strip()
                        en_text = translation.get('en', '').strip()

                        if (de_text and en_text and
                            5 <= len(de_text.split()) <= 35 and
                            5 <= len(en_text.split()) <= 35):
                            samples.append({
                                'source_text': de_text, 'target_text': en_text,
                                'source_lang': 'de', 'target_lang': 'en',
                                'dataset': dataset_name.split('/')[-1], 'domain': 'mixed'
                            })
                            count += 1
                if count > 0:
                    print(f"\t[SUCCESS] Fetched {count} samples from \"{dataset_name}\"")
            except Exception as e:
                print(f"\t[FAILURE] Could not load \"{dataset_name}\": {e}")
        return samples

    def load_wmt_datasets(self) -> List[Dict[str, str]]:
        """Loads datasets from the WMT series."""
        samples = []
        wmt_configs = [("wmt19", "de-en")]

        for dataset_name, lang_pair in wmt_configs:
            try:
                print(f"\t[LOADING...] Trying to load dataset \"{dataset_name}\"...")
                from datasets import load_dataset
                dataset = load_dataset(dataset_name, lang_pair, split="train")

                count = 0
                for example in dataset:
                    if count >= self.max_samples_per_dataset or len(samples) >= self.max_samples_total:
                        break
                    
                    if 'translation' in example:
                        translation = example['translation']
                        de_text = translation.get('de', '').strip()
                        en_text = translation.get('en', '').strip()

                        if (de_text and en_text and 5 <= len(de_text.split()) <= 35):
                            samples.append({
                                'source_text': de_text, 'target_text': en_text,
                                'source_lang': 'de', 'target_lang': 'en',
                                'dataset': dataset_name, 'domain': 'news'
                            })
                            count += 1
                if count > 0:
                    print(f"\t[SUCCESS] Fetched {count} samples from {dataset_name}")
            except Exception as e:
                print(f"\t[FAILURE] Could not load {dataset_name}: {e}")
        return samples

    def load_multi30k_dataset(self) -> List[Dict[str, str]]:
        """Loads the Multi30K dataset (image description translations)."""
        samples = []
        try:
            print(f"\t[LOADING...] Trying to load dataset \"Multi30K\"...")
            from datasets import load_dataset
            dataset = load_dataset("bentrevett/multi30k", split="train")

            count = 0
            for example in dataset:
                if count >= self.max_samples_per_dataset or len(samples) >= self.max_samples_total:
                    break

                de_text = example.get('de', '').strip()
                en_text = example.get('en', '').strip()

                if (de_text and en_text and 5 <= len(de_text.split()) <= 35):
                    samples.append({
                        'source_text': de_text, 'target_text': en_text,
                        'source_lang': 'de', 'target_lang': 'en',
                        'dataset': 'multi30k', 'domain': 'description'
                    })
                    count += 1
            if count > 0:
                print(f"\t[SUCCESS] Fetched {count} samples from \"Multi30K\"")
        except Exception as e:
            print(f"\t[FAILURE] Could not load \"Multi30K\": {e}")
        return samples

    def load_all_datasets(self) -> List[Dict[str, str]]:
        """
        Loads all available data sources, combines them, filters for quality,
        and removes duplicates.
        """
        all_samples = []
        print("[BEGIN] Starting to load all datasets...")

        all_samples.extend(self.load_opus_datasets())
        all_samples.extend(self.load_wmt_datasets())
        all_samples.extend(self.load_multi30k_dataset())

        if len(all_samples) < self.max_samples_total:
            print("\t[WARNING] Online data is insufficient! Using local fallback data...")
            fallback_samples = self.get_fallback()
            random.shuffle(fallback_samples)
            needed = self.max_samples_total - len(all_samples)
            all_samples.extend(fallback_samples[:needed])

        # Deduplication process
        unique_samples = []
        seen_pairs = set()
        for sample in all_samples:
            source = sample['source_text'].strip().lower()
            target = sample['target_text'].strip().lower()
            if (source, target) not in seen_pairs and len(source) > 10:
                seen_pairs.add((source, target))
                unique_samples.append(sample)

        print(f"[COMPLETED] Loaded a total of {len(unique_samples)} unique, deduplicated samples.")

        # Display final statistics
        if unique_samples:
            dataset_counts = {}
            domain_counts = {}
            for sample in unique_samples:
                dataset = sample['dataset']
                domain = sample['domain']
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            print("\tFinal Dataset Distribution:")
            for dataset, count in dataset_counts.items():
                print(f"\t\t- {dataset}: {count} samples")
            print("\tFinal Domain Distribution:")
            for domain, count in domain_counts.items():
                print(f"\t\t- {domain}: {count} samples")

        return unique_samples

    def get_fallback(self) -> List[Dict[str, str]]:
        """Provides a hardcoded list of diverse, high-quality samples as a fallback."""
        # This list is a direct copy from the user's data_generation.py script.
        return [
            # News
            # No. Samples: 40
            {'source_text': "Die deutsche Wirtschaft wächst trotz globaler Unsicherheiten weiter.", 'target_text': "The German economy continues to grow despite global uncertainties.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Forscher entwickeln neue Behandlungsmethoden gegen Alzheimer.", 'target_text': "Researchers develop new treatment methods against Alzheimer's disease.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Der Klimawandel beeinflusst die Landwirtschaft in Europa erheblich.", 'target_text': "Climate change significantly affects agriculture in Europe.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Künstliche Intelligenz revolutioniert die moderne Medizin.", 'target_text': "Artificial intelligence is revolutionizing modern medicine.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Erneuerbare Energien decken bereits dreißig Prozent des Strombedarfs.", 'target_text': "Renewable energy already covers thirty percent of electricity demand.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Die Europäische Union plant strengere Umweltgesetze für Unternehmen.", 'target_text': "The European Union plans stricter environmental laws for companies.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Digitale Technologien verändern die Art, wie wir arbeiten und lernen.", 'target_text': "Digital technologies are changing the way we work and learn.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Die Raumfahrtindustrie investiert Milliarden in die Mars-Erforschung.", 'target_text': "The space industry invests billions in Mars exploration.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Neue Studien zeigen positive Auswirkungen von Sport auf die Gesundheit.", 'target_text': "New studies show positive effects of exercise on health.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Die Bevölkerung in den Städten wächst schneller als auf dem Land.", 'target_text': "The population in cities is growing faster than in rural areas.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Wissenschaftler entdecken neue Arten im Amazonas-Regenwald.", 'target_text': "Scientists discover new species in the Amazon rainforest.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Die Arbeitslosigkeit sinkt auf den niedrigsten Stand seit Jahren.", 'target_text': "Unemployment falls to the lowest level in years.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Banken investieren verstärkt in nachhaltige Finanzprodukte.", 'target_text': "Banks increasingly invest in sustainable financial products.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Die Regierung kündigt neue Bildungsreformen für Schulen an.", 'target_text': "The government announces new educational reforms for schools.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},
            {'source_text': "Moderne Architektur prägt das Stadtbild vieler Metropolen.", 'target_text': "Modern architecture shapes the cityscape of many metropolises.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_news', 'domain': 'news'},

            # Conversations
            # No. Samples: 40
            {'source_text': "Guten Morgen! Wie geht es Ihnen heute?", 'target_text': "Good morning! How are you doing today?", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Können Sie mir bitte dabei helfen, das Problem zu lösen?", 'target_text': "Can you please help me solve this problem?", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Entschuldigung, wo finde ich den nächsten Supermarkt?", 'target_text': "Excuse me, where can I find the nearest supermarket?", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Ich hätte gerne zwei Kaffee und ein Stück Kuchen, bitte.", 'target_text': "I would like two coffees and a piece of cake, please.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Wann fährt der nächste Zug nach München ab?", 'target_text': "When does the next train to Munich depart?", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Vielen Dank für Ihre Hilfe, das war sehr freundlich.", 'target_text': "Thank you very much for your help, that was very kind.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Könnten Sie bitte langsamer sprechen? Ich verstehe nicht alles.", 'target_text': "Could you please speak more slowly? I don't understand everything.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Wie viel kostet eine Fahrkarte nach Berlin?", 'target_text': "How much does a ticket to Berlin cost?", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Ist dieses Restaurant heute Abend noch geöffnet?", 'target_text': "Is this restaurant still open tonight?", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Das Wetter ist heute wirklich schön und warm.", 'target_text': "The weather is really nice and warm today.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Haben Sie vielleicht eine Empfehlung für ein gutes Hotel?", 'target_text': "Do you perhaps have a recommendation for a good hotel?", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Ich möchte gerne einen Tisch für vier Personen reservieren.", 'target_text': "I would like to reserve a table for four people.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Können Sie mir sagen, wie ich zum Bahnhof komme?", 'target_text': "Can you tell me how to get to the train station?", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Es tut mir leid, ich habe Sie nicht verstanden.", 'target_text': "I'm sorry, I didn't understand you.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},
            {'source_text': "Wie lange dauert es, bis wir dort ankommen?", 'target_text': "How long does it take until we arrive there?", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_conversation', 'domain': 'conversation'},

            # Business/Technology
            # No. Samples: 30
            {'source_text': "Die neue Software wird nächste Woche implementiert werden.", 'target_text': "The new software will be implemented next week.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_business', 'domain': 'business'},
            {'source_text': "Unser Team arbeitet an einem innovativen Projekt.", 'target_text': "Our team is working on an innovative project.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_business', 'domain': 'business'},
            {'source_text': "Der Bericht muss bis Freitag fertiggestellt werden.", 'target_text': "The report must be completed by Friday.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_business', 'domain': 'business'},
            {'source_text': "Die Präsentation war sehr überzeugend und gut strukturiert.", 'target_text': "The presentation was very convincing and well-structured.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_business', 'domain': 'business'},
            {'source_text': "Wir müssen unsere Verkaufsstrategie überdenken und verbessern.", 'target_text': "We need to rethink and improve our sales strategy.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_business', 'domain': 'business'},
            {'source_text': "Das Unternehmen hat seine Gewinnziele übertroffen.", 'target_text': "The company has exceeded its profit targets.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_business', 'domain': 'business'},
            {'source_text': "Die Qualitätskontrolle ist ein wesentlicher Teil unseres Prozesses.", 'target_text': "Quality control is an essential part of our process.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_business', 'domain': 'business'},
            {'source_text': "Unsere Kunden schätzen den hervorragenden Service sehr.", 'target_text': "Our customers greatly appreciate the excellent service.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_business', 'domain': 'business'},
            {'source_text': "Die Produktivität hat sich im letzten Quartal deutlich erhöht.", 'target_text': "Productivity has increased significantly in the last quarter.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_business', 'domain': 'business'},
            {'source_text': "Wir benötigen eine effizientere Lösung für dieses Problem.", 'target_text': "We need a more efficient solution for this problem.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_business', 'domain': 'business'},

            # Descriptive/Narrative
            # No. Samples: 30
            {'source_text': "Ein junger Mann liest ein Buch im sonnigen Park.", 'target_text': "A young man reads a book in the sunny park.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_description', 'domain': 'description'},
            {'source_text': "Die Kinder spielen fröhlich auf dem großen Spielplatz.", 'target_text': "The children play happily on the large playground.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_description', 'domain': 'description'},
            {'source_text': "Eine elegante Frau trägt ein rotes Kleid und hohe Schuhe.", 'target_text': "An elegant woman wears a red dress and high heels.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_description', 'domain': 'description'},
            {'source_text': "Der alte Leuchtturm steht majestätisch an der felsigen Küste.", 'target_text': "The old lighthouse stands majestically on the rocky coast.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_description', 'domain': 'description'},
            {'source_text': "Bunte Blumen blühen wunderschön in dem gepflegten Garten.", 'target_text': "Colorful flowers bloom beautifully in the well-maintained garden.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_description', 'domain': 'description'},
            {'source_text': "Der erfahrene Koch bereitet ein leckeres Abendessen zu.", 'target_text': "The experienced chef prepares a delicious dinner.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_description', 'domain': 'description'},
            {'source_text': "Touristen fotografieren die beeindruckende historische Architektur.", 'target_text': "Tourists photograph the impressive historic architecture.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_description', 'domain': 'description'},
            {'source_text': "Der fleißige Student arbeitet intensiv an seiner Masterarbeit.", 'target_text': "The diligent student works intensively on his master's thesis.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_description', 'domain': 'description'},
            {'source_text': "Ein großer Hund läuft schnell durch den grünen Wald.", 'target_text': "A large dog runs quickly through the green forest.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_description', 'domain': 'description'},
            {'source_text': "Die moderne Bibliothek bietet viele interessante Bücher und Zeitschriften.", 'target_text': "The modern library offers many interesting books and magazines.", 'source_lang': 'de', 'target_lang': 'en', 'dataset': 'fallback_description', 'domain': 'description'},
        ]
