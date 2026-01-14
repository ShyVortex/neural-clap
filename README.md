# neural-clap
### Neural Crowd Listener for App Release Planning

> **Evolution of the CLAP project:** reimagining mobile app review classification using Large Language Models (LLMs).

---

## About
**neural-clap** is an advanced classification suite designed to analyze user feedback from mobile app stores.
Building upon the methodology introduced in [*"Listening to the Crowd for the Release Planning of Mobile Apps"*](https://ieeexplore.ieee.org/document/8057860) (*Scalabrino et al., 2017*),
this repository replaces the original Random Forest and manual NLP pipelines with state-of-the-art LLMs.

While the original **CLAP** tool relied on statistical methods, stemming, and custom negation handling,
**neural-clap** leverages the semantic reasoning of Generative AI to categorize reviews with greater nuance, specifically
targeting complex categories like usability and security where statistical models traditionally struggled.

## Key Features
* **Hybrid Model Architecture:** run classifications using local privacy-focused models (via [**Ollama**](https://github.com/ollama/ollama))
  or separately with powerful cloud-based models (e.g. *[ChatGPT](https://chatgpt.com/), [Gemini](https://gemini.google.com/), [Mistral](https://chat.mistral.ai/chat), [Qwen](https://qwen.ai/)*).
* **7-Category Standard:** automatically sorts reviews into the rigorous taxonomy defined in the original research paper:
    * **BUG**: crashes, broken functionality, errors.
    * **FEATURE**: requests for new capabilities.
    * **PERFORMANCE**: lag, slow loading, freezes.
    * **SECURITY**: privacy concerns, hacks, permissions.
    * **ENERGY**: battery drain, overheating.
    * **USABILITY**: UI difficulties, accessibility, confusing design.
    * **OTHER**: non-informative reviews, praise, or noise.
* **Classification Arena:** includes scripts to **compare and benchmark** results between CLAP and different models
  to evaluate accuracy against the original baseline.

## CLAP vs. Neural-CLAP
| Feature               | CLAP (2017)                           | Neural-CLAP (2026)                    |
|:----------------------|:--------------------------------------|:--------------------------------------|
| **Core Engine**       | Random Forest (Statistical)           | Large Language Models (Generative)    |
| **Preprocessing**     | Heavy (Stop-words, Stemming, N-grams) | Minimal / None (Raw Text)             |
| **Negation Handling** | Custom State Machine / Parser         | Native Semantic Understanding         |
| **Multilingual**      | Failed (~50% accuracy loss)           | Native Multilingual Support (via LLM) |

## Installation

```bash
git clone https://github.com/ShyVortex/neural-clap.git
cd neural-clap
pip install -r requirements.txt
```