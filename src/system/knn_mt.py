# -*- coding: utf-8 -*-
"""
src/system/knn_mt.py

This module contains the core implementation of the kNN-MT system. It is responsible
for loading models, managing the kNN datastore and FAISS indexes, computing all
state vectors (Error, Pressure, Context), and performing the translation process.
"""

import time
import faiss
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import MarianMTModel, MarianTokenizer
from transformers.cache_utils import EncoderDecoderCache
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple, Optional, Any
from difflib import SequenceMatcher
from sacrebleu import sentence_bleu

# Import project-specific modules
from src import config
from src.core import ErrorStateVector, GenerativeContextVector, Action
from src.system.decoding import BeamSearchDecoder, BeamHypothesis


class kNNMTSystem:
    """
    The core kNN-MT system, handling translation, state computation, and kNN retrieval.
    This class integrates the NMT model with a FAISS-based datastore to simulate
    a production-like environment.
    """

    def __init__(self):
        """Initializes the entire kNN-MT system."""
        print("[START] Initializing kNN-MT System...")

        # --- 1. Load Models and Tokenizers from config ---
        self.tokenizer = MarianTokenizer.from_pretrained(config.MODEL_NAMES["translation_model"])
        self.model = MarianMTModel.from_pretrained(config.MODEL_NAMES["translation_model"])
        self.sentence_encoder = SentenceTransformer(config.MODEL_NAMES["sentence_encoder"])
        self.lm_tokenizer = MarianTokenizer.from_pretrained(config.MODEL_NAMES["fluency_lm"])
        self.lm_model = MarianMTModel.from_pretrained(config.MODEL_NAMES["fluency_lm"])

        # Set models to evaluation mode
        self.model.eval()
        self.lm_model.eval()

        # --- 2. Configure Model and Datastore Dimensions from config ---
        self.model_hidden_size = self.model.config.d_model
        self.datastore_size = config.KNN_SYSTEM_PARAMS["datastore_size"]
        self.embedding_dim = config.KNN_SYSTEM_PARAMS["embedding_dim"]

        # --- 3. Initialize Query Projection Layer ---
        # This layer projects the decoder's hidden state into the kNN query space.
        self.query_projection = nn.Linear(self.model_hidden_size, self.embedding_dim)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.zeros_(self.query_projection.bias)

        # --- 4. Create Simulated kNN Datastore ---
        # In a real system, these would be pre-computed from a large training corpus.
        np.random.seed(config.RANDOM_SEED)
        self.datastore_embeddings = np.random.randn(self.datastore_size, self.embedding_dim).astype('float32')

        # Simulate a more realistic token distribution (frequent tokens are more common)
        vocab_size = self.model.config.vocab_size
        token_probs = np.exp(-np.arange(vocab_size) / 1000)
        token_probs /= np.sum(token_probs)
        self.datastore_values = np.random.choice(vocab_size, size=self.datastore_size, p=token_probs)

        # --- 5. Initialize FAISS Indexes ---
        self.exact_index = faiss.IndexFlatIP(self.embedding_dim)
        self.hnsw_index = faiss.IndexHNSWFlat(self.embedding_dim, 32) # 32 connections per node
        nlist = min(100, self.datastore_size // 10) # Number of IVF cells
        self.ivf_index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(self.embedding_dim), self.embedding_dim, nlist
        )

        # Train and populate the indexes
        self._initialize_indexes()
        
        # Loading spaCy NER models for faithfulness calculation
        try:
            from packaging import version
            import spacy

            if version.parse(spacy.__version__) >= version.parse("3.8.0"):
                ner_en_spacy_ver = 'en_core_web_trf'
            else:
                ner_en_spacy_ver = 'en_core_web_lg'
            self.ner_de = spacy.load('de_core_news_lg')
            self.ner_en = spacy.load(ner_en_spacy_ver)
            print("[SUCCESS] spaCy models loaded.")
        except ImportError:
            raise ImportError("[ERROR] spaCy not installed. Please run: pip install spacy")
        except OSError:
            raise OSError(f'[ERROR] spaCy models not found/no longer supported. Please run: python -m spacy download de_core_news_lg {ner_en_spacy_ver}')

        print("[SUCCESS] kNN-MT system initialization complete.")
        print(f"\t- Model: {config.MODEL_NAMES['translation_model']}")
        print(f"\t- Datastore size: {self.datastore_size}")
        print(f"\t- Embedding dimension: {self.embedding_dim}")

    def _initialize_indexes(self):
        """
        Private helper to normalize embeddings, train IVF index, and add data to all indexes.
        """
        print("[INFO] Building FAISS indexes...")
        # Normalizing embeddings is crucial for inner product (cosine similarity) search
        embeddings_normalized = self.datastore_embeddings.copy()
        faiss.normalize_L2(embeddings_normalized)

        # Add data to flat and HNSW indexes
        self.exact_index.add(embeddings_normalized)
        self.hnsw_index.add(embeddings_normalized)

        # The IVF index requires a training step to define the clusters (cells)
        if self.datastore_size >= self.ivf_index.nlist:
            self.ivf_index.train(embeddings_normalized)
            self.ivf_index.add(embeddings_normalized)
        else:
            print(f"[WARNING] Datastore size ({self.datastore_size}) is too small to train IVF index. Skipping.")

        print("[SUCCESS] FAISS indexes are ready.")

    def project_to_query_embedding(self, hidden_state: torch.Tensor) -> np.ndarray:
        """
        Projects the NMT decoder's last hidden state to the kNN query embedding space.
        """
        with torch.no_grad():
            # Ensure we are using the hidden state of the last token
            last_hidden_vec = hidden_state[0, -1, :] if len(hidden_state.shape) == 3 else hidden_state
            
            query_embedding = self.query_projection(last_hidden_vec).cpu().numpy()
            
            # L2 normalize the query to match the datastore's normalized embeddings
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            
            return query_embedding.astype('float32')

    def _align_entities(self, source_entities: List[str], generated_entities: List[str]) -> int:
        """
        Aligns entities from source to generated text using direct and semantic matching.
        """
        if not source_entities or not generated_entities:
            return 0

        aligned_count = 0
        used_generated_indices = set()

        # 1. Direct Matching (case-insensitive)
        gen_entities_lower = [e.lower() for e in generated_entities]
        for src_ent in source_entities:
            try:
                idx = gen_entities_lower.index(src_ent.lower())
                if idx not in used_generated_indices:
                    aligned_count += 1
                    used_generated_indices.add(idx)
                    continue # Move to next source entity
            except ValueError:
                pass # Not found, proceed to semantic matching

        # 2. Semantic Matching for remaining entities
        remaining_src_ents = [se for se in source_entities if se.lower() not in [ge.lower() for i, ge in enumerate(generated_entities) if i in used_generated_indices]]
        remaining_gen_ents_indices = [i for i, ge in enumerate(generated_entities) if i not in used_generated_indices]
        
        if not remaining_src_ents or not remaining_gen_ents_indices:
            return aligned_count

        remaining_gen_ents = [generated_entities[i] for i in remaining_gen_ents_indices]
        
        src_embs = self.sentence_encoder.encode(remaining_src_ents, convert_to_tensor=True)
        gen_embs = self.sentence_encoder.encode(remaining_gen_ents, convert_to_tensor=True)
        
        cosine_scores = util.pytorch_cos_sim(src_embs, gen_embs)

        for i in range(len(remaining_src_ents)):
            # Find the best match for the current source entity
            best_match_idx = torch.argmax(cosine_scores[i]).item()
            
            if cosine_scores[i, best_match_idx] > 0.8: # Similarity threshold
                original_gen_idx = remaining_gen_ents_indices[best_match_idx]
                if original_gen_idx not in used_generated_indices:
                    aligned_count += 1
                    used_generated_indices.add(original_gen_idx)
                    # To prevent one generated entity from matching multiple source entities,
                    # we can zero out its column in the similarity matrix.
                    cosine_scores[:, best_match_idx] = -1 
        
        return aligned_count

    def compute_error_state(
        self, source_text: str,
        generated_prefix_words: List[str], reference_text: Optional[str] = None
    ) -> ErrorStateVector:
        """
        Computes the ErrorStateVector based on the current generated prefix.
        Corresponds to Section 3.1 of the theoretical framework.
        """
        if not generated_prefix_words:
            return ErrorStateVector(0.0, 0.0, 0.0)

        generated_text = " ".join(generated_prefix_words)

        # 1. Semantic Drift Error (ε_sem)
        try:
            source_emb = self.sentence_encodeen_core_web_trfr.encode([source_text], convert_to_tensor=True)
            generated_emb = self.sentence_encoder.encode([generated_text], convert_to_tensor=True)
            cos_sim = F.cosine_similarity(source_emb, generated_emb).item()
            semantic_drift = np.clip(1.0 - cos_sim, 0.0, 2.0)
        except Exception:
            semantic_drift = 0.5 # Fallback value

        # 2. Fluency Degradation Error (ε_flu)
        try:
            inputs = self.lm_tokenizer(generated_text, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.lm_model(**inputs, labels=inputs["input_ids"])
                # perplexity = torch.exp(outputs.loss).item()
                fluency_degradation = outputs.loss.item()
            # Normalize perplexity to a reasonable error scale
            # fluency_degradation = min(1.0, np.log(perplexity + 1) / 10.0)
        except Exception:
            fluency_degradation = 0.5 # Fallback value

        # 3. Faithfulness Mismatch Error (ε_fth)
        try:
            if not self.ner_de or not self.ner_en:
                raise NotImplementedError("spaCy models not loaded, falling back.")
            source_doc = self.ner_de(source_text)
            source_entities = [ent.text for ent in source_doc.ents]

            if not source_entities:
                faithfulness_mismatch = 0.0
            else:
                generated_doc = self.ner_en(generated_text)
                generated_entities = [ent.text for ent in generated_doc.ents]
                
                aligned_count = self._align_entities(source_entities, generated_entities)
                
                faithfulness_mismatch = 1.0 - (aligned_count / len(source_entities))
        except Exception:
            # Fallback to word overlap if NER fails
            if reference_text:
                ref_tokens = set(reference_text.lower().split())
                gen_tokens = set(generated_text.lower().split())
                if ref_tokens:
                    overlap = len(ref_tokens & gen_tokens) / len(ref_tokens)
                    faithfulness_mismatch = 1.0 - overlap
                else:
                    faithfulness_mismatch = 0.0
            else:
                faithfulness_mismatch = 0.5 # Default fallback
        # faithfulness_mismatch = 0.5 # Default fallback
        # if reference_text:
        #     try:
        #         # Use SacreBLEU for a robust score
        #         bleu = sentence_bleu(generated_text, [reference_text])
        #         faithfulness_mismatch = max(0.0, 1.0 - bleu.score / 100.0)
        #     except Exception:
        #         # Fallback to simple word overlap
        #         ref_tokens = set(reference_text.lower().split())
        #         gen_tokens = set(generated_text.lower().split())
        #         if ref_tokens:
        #             overlap = len(ref_tokens & gen_tokens) / len(ref_tokens)
        #             faithfulness_mismatch = 1.0 - overlap
        # else:
        #     # Estimate based on length ratio if no reference is available
        #     source_len = len(source_text.split())
        #     generated_len = len(generated_prefix_words)
        #     expected_len = source_len * 1.1  # Heuristic for DE->EN
        #     length_ratio = generated_len / max(1, expected_len)
        #     faithfulness_mismatch = min(1.0, abs(1.0 - length_ratio))

        return ErrorStateVector(
            semantic_drift=semantic_drift,
            fluency_degradation=fluency_degradation,
            faithfulness_mismatch=np.clip(faithfulness_mismatch, 0.0, 1.0)
        )

    def perform_knn_retrieval(self, query_embedding: np.ndarray, action: Action) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Performs kNN retrieval using the appropriate FAISS index based on the chosen action.
        """
        if action.k == 0:
            return np.array([]), np.array([]), 0.0

        start_time = time.time()
        
        index_map = {
            'exact': self.exact_index,
            'hnsw': self.hnsw_index,
            'ivf_pq': self.ivf_index
        }
        index = index_map.get(action.index_type, self.exact_index)

        try:
            distances, indices = index.search(query_embedding.reshape(1, -1), action.k)
            retrieved_values = self.datastore_values[indices[0]]
        except Exception as e:
            print(f"[WARNING] kNN retrieval failed: {e}. Returning empty results.")
            return np.array([]), np.array([]), time.time() - start_time

        retrieval_time = time.time() - start_time
        return distances[0], retrieved_values, retrieval_time

    def compute_context_state(self, hidden_state: torch.Tensor, query_embedding: np.ndarray) -> GenerativeContextVector:
        """
        Computes the GenerativeContextVector from the decoder's internal state.
        Corresponds to Section 3.3 of the theoretical framework.
        """
        # 1. & 2. Predictive Uncertainty (H_unc) and Confidence (H_cnf)
        try:
            with torch.no_grad():
                last_hidden = hidden_state[0, -1, :] if len(hidden_state.shape) == 3 else hidden_state
                vocab_logits = self.model.lm_head(last_hidden)
                vocab_probs = F.softmax(vocab_logits, dim=-1)
                
                log_probs = torch.log(vocab_probs + 1e-10)
                entropy = -torch.sum(vocab_probs * log_probs).item()
                max_prob = torch.max(vocab_probs).item()
        except Exception:
            entropy = 2.5 # Fallback
            max_prob = 0.5 # Fallback

        # 3. Retrieval Relevance (H_rel)
        try:
            # Use the fastest index (exact but small datastore) to get the best possible relevance score
            distances, _ = self.exact_index.search(query_embedding.reshape(1, -1), k=1)
            min_distance = distances[0][0]
            # Convert distance to relevance score in [0, 1]
            relevance = 1.0 / (1.0 + min_distance)
        except Exception:
            relevance = 0.5 # Fallback

        return GenerativeContextVector(
            predictive_uncertainty=entropy,
            predictive_confidence=max_prob,
            retrieval_relevance=relevance
        )

    def _tokens_to_words(self, tokens: List[int]) -> List[str]:
        """Helper function to decode a list of token IDs to a list of words."""
        # Use batch_decode for efficiency and correct handling of special tokens
        text = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return text.split()

    def translate_with_knn_beam_search(self, source_text: str, reference_text: Optional[str] = None,
                                      max_length: int = 64, num_beams: int = 3,
                                      length_penalty: float = 1.0) -> Dict[str, Any]:
        """
        Performs translation using beam search, collecting state-action-reward data at each step.
        """

        # --- 1. Encode Source Text ---
        inputs = self.tokenizer(source_text, return_tensors="pt", max_length=256, truncation=True)
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True
            )

        # --- 2. Initialize Beam Search ---
        decoder = BeamSearchDecoder(
            beam_size=num_beams,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            length_penalty=length_penalty
        )
        
        # Start with a single hypothesis containing the PAD token as the start-of-sequence token
        initial_beams = [
            BeamHypothesis(
                tokens=[self.tokenizer.pad_token_id],   # Start with the PAD token
                log_prob=0.0,                           # Initial log probability
                past_key_values=None                    # Past key values for caching
            )
        ]

        # --- 3. Define the Step Function for the Decoder ---
        def beam_step(beams: List[BeamHypothesis], step: int) -> List[BeamHypothesis]:
            """Performs one step of beam search expansion."""
            all_candidates = []
            for beam in beams:
                
                # --- Get NMT predictions ---
                decoder_input_ids = torch.tensor([[beam.tokens[-1]]]).long()
                with torch.no_grad():
                    decoder_outputs = self.model.get_decoder()(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=encoder_outputs.last_hidden_state,
                        encoder_attention_mask=inputs["attention_mask"],
                        past_key_values=beam.past_key_values
                    )
                    last_hidden_state = decoder_outputs.last_hidden_state
                    # Transformers v4.38+ a `Cache` object is returned, otherwise a tuple
                    new_past_key_values = decoder_outputs.past_key_values
                    
                    lm_logits = self.model.lm_head(last_hidden_state[:, -1, :])
                    log_probs = F.log_softmax(lm_logits, dim=-1)[0]
                
                # --- Compute shared states for this beam ---
                query_embedding = self.project_to_query_embedding(last_hidden_state)
                context_state = self.compute_context_state(last_hidden_state, query_embedding)

                # --- Expand with top-k candidates ---
                topk_log_probs, topk_indices = torch.topk(log_probs, k=num_beams)
                for i in range(num_beams):
                    new_token = topk_indices[i].item()
                    new_tokens = beam.tokens + [new_token]
                    new_log_prob = beam.log_prob + topk_log_probs[i].item()
                    
                    # Compute error state for this new, specific path
                    new_generated_words = self._tokens_to_words(new_tokens[1:]) # Exclude start token
                    new_error_state = self.compute_error_state(source_text, new_generated_words, reference_text)
                    
                    # Create the new hypothesis
                    new_hypothesis = BeamHypothesis(
                        tokens=new_tokens,
                        log_prob=new_log_prob,
                        hidden_states=beam.hidden_states + [last_hidden_state],
                        context_states=beam.context_states + [context_state],
                        error_states=beam.error_states + [new_error_state],
                        query_embeddings=beam.query_embeddings + [query_embedding],
                        past_key_values=new_past_key_values
                    )
                    all_candidates.append(new_hypothesis)

            # --- Prune the beams ---
            all_candidates.sort(key=lambda h: h.score(length_penalty), reverse=True)
            return all_candidates[:num_beams]

        # --- 4. Run the Search ---
        try:
            completed_hypotheses = decoder.search(initial_beams, beam_step, max_length)
            
            if not completed_hypotheses:
                raise ValueError("Beam search returned no hypotheses.")
            
            best_hypothesis = completed_hypotheses[0]
            
            # --- 5. Construct Trajectory from the Best Hypothesis ---
            trajectory = []
            for step in range(len(best_hypothesis.tokens) -1): # -1 because we have N tokens but N-1 steps
                step_data = {
                    'step': step,
                    'generated_tokens': self._tokens_to_words(best_hypothesis.tokens[1:step+2]),
                    'error_state': best_hypothesis.error_states[step],
                    'context_state': best_hypothesis.context_states[step],
                    'query_embedding': best_hypothesis.query_embeddings[step],
                    'beam_score': best_hypothesis.score(length_penalty),
                    'num_active_beams': num_beams,
                }
                trajectory.append(step_data)

            final_translation = " ".join(self._tokens_to_words(best_hypothesis.tokens[1:]))

        except Exception as e:
            import traceback
            traceback.print_exc(); exit()
            print(f"[ERROR] Translation failed: {e}. Using fallback.")
            final_translation = self._simple_translation_fallback(source_text)
            trajectory = []
        
        print(f"[INFO] Translating (beam_size={num_beams}): \"{source_text[:50]}{"..." if len(source_text) > 50 else ''}\"")
        print(f'[INFO] Final Translation:{" "*int(9+num_beams//10)}\"{final_translation[:50]}{"..." if len(final_translation) > 50 else ''}\"')

        return {
            'translation': final_translation,
            'trajectory': trajectory,
            'num_steps': len(trajectory),
            'source_text': source_text,
            'reference_text': reference_text,
            'beam_size': num_beams,
            'decoding_strategy': 'beam_search',
        }

    def _simple_translation_fallback(self, source_text: str) -> str:
        """A very basic, non-ML fallback method for translation."""
        # This is a placeholder for robustness. In a real system, this might
        # be a call to a different, more stable translation service.
        basic_dict = {"guten": "good", "morgen": "morning", "danke": "thank", "bitte": "please"}
        words = source_text.lower().split()
        translated_words = [basic_dict.get(word.strip(".,!?"), word) for word in words]
        return " ".join(translated_words)
