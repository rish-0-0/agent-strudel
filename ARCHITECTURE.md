# Architecture of this repo

This project basically has 2 agents which aid in development of a fine tuned model ready for generating strudel code to run music

Agents:

1. Training data generator agent.
2. Fine tuning agent.

Frontend:


## Training Data Generator Agent

This agent uses Claude's Agentic SDK framework to scaffold itself as an agent which makes useful strudel scripts on which a baseline model will fine tune itself to write strudel scripts.

The pipeline is simple:

There is one LLM: Sonnet-5 with low effort would do.

It generates a strudel script
A bash nodejs call is made to check if everything compiles and runs.

If it runs successfully

Another LLM invocation judges the generated snippet and confirms whether this script will be a useful addition to the training dataset.

If the evaluator LLM call decides "yes", we add the generated "music" to the dataset along with a label giving a description to the kind of music generated, so we get a labelled dataset, and while generating music we can be trained on different kinds of music, else we ask the agent to generate another snippet.

The context will keep growing and to avoid that there will be a summary step every 20 generations to compact the context and generate a summary into the state of the model.

I think a good limit on the number of generations would be 2000 and for fine tuning we can use 80% and the remaining 20% can be used to self train the model.

## Fine tuning agent

Another folder in the root level for this agent. We are going to be fine tuning SmolLM2-135M Model and experimenting with how well it can run on the web and generate strudel code.

