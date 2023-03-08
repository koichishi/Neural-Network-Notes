#### Fields of NLP that are yet to explore
- **Retrieval augmented in-context learning** (RAICL)
- Better benchmarks
- "Last mile" for productive application
- Faithful, human-interpretable explanations

## The Rise of In-context Learning
- #### Standard supervision vs. In-context Learning
| Standard supervision for *nervous anticipation* | In-context Learning |
|---|---|
|Need a dataset of positive/negative example of certain phenomenon | Model must learning the **meanings** of the terms and **intensions** |
|Specific models surves different tasks| A huge, giant model that can serve all tasks |
- #### Key Architecture: Transformers and attentions!
	Burning area: why does this simple model works so well?
- #### Self-supervision
	1. The model's *only* objective is to learn co-occurrence patterns in the sequences it is trained on
	2. Alternatively: to asssign high probability to attested sequences
	3. Generation then involves *sampling* from the model
	4. The sequences can contain anything - code, natural languages, etc.
	5. The objective can't mention specific symbols or relations between symbols (no standard supervision)
- #### Large-scale pretraining
	GloVe, BERT, GPT, GPT-3, ...
- #### Learning from human feedback (Don't overlook!)
	From ChatGPT blog, we see that besides self-supervision, humans participate in
	- Binary classification on good/bad generation
	- Ranking of generations 
		to be fed to light-weighted reinforcement learning
- #### Step-by-step and chain-of-thought reasoning
	Can models reasons about negations?
	E.g. Step-by-step prompting style giving to ChatGPT:
	- Logical and commonsense reasoning exam
	- Explain your reasoning in detail, then answer with Yes or No. Your answers should follow this 4-line format
	**Much better results!**

## Retrieval augmented in-context learning
- #### Large language models for everything?
	E.g. "When was UCSD founded" -> HUGE MODEL -> "UCSD was found in ..."
	- Efficiency :(
	- Updateability :(
	- Provenance (Is source of the answer believable?) :(
	- Effectiveness :(
	- Synthesis :(
- #### Retrieval-based NLP
	Scores documents from the database and sythesize them into one answer
	- Efficiency!
	- Updateability!
	- Provenance!
	- Effectiveness!
	- Synthesis!
- #### The present: Wrangling pretrained components
	Specify specific task parameters that meant to *tie* the integrated models 
	- Hard to design and debug!
- #### Models can communicate in natural language
- #### Few-shot OpenQA
	- Hindsight retrival - use retriever recursively to get questions and answers as prompt
	- 