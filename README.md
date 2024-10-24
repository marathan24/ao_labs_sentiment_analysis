## AO Reference Design #1
# Sentiment Analysis Streamlit Demo
Maintainer: [spolisar](https://github.com/spolisar), shane@aolabs.ai

A streamlit app where users can train an weightless neural network agent to do sentiment analysis tasks.

AO labs is building AI agents than can learn after training. 

In arch__sentiment.py, you can view the agent's particular neural architecture and try your hand at how different configurations affect performance.


## Installation & Setup

You can run this app in a docker container (recommended) or directly on your local environment. You'll need to `pip install` from our private repo ao_core, which is currently in private beta-- say hi on our [discord](https://discord.com/invite/nHuJc4Y4n7) for access!


### Docker Installation

1) Generate a GitHub Personal Access Token to ao_core    
    Go to https://github.com/settings/tokens?type=beta

2) Clone this repo and create a `.env` file in your local clone where you'll add the PAT as follows:
    `ao_github_PAT=token_goes_here`
    No spaces! See `.env_example`.

3) In a Git Bash terminal, build and run the Dockerfile with these commands:
```shell
export DOCKER_BUILDKIT=1

docker build --secret id=env,src=.env -t "ao_app" .

docker run -p 8501:8501 streamlit
```
You're done! Access the app at `localhost:8501` in your browser.

### Local Environment Installation

To install in a local conda or venv environment and run the app, use these commands:

```shell
pip install -r requirements.txt

streamlit run main.py
```
*Important:* You'll first need to uncomment lines 4 & 5 in the requirements.txt file.

You're done! Access the app at `localhost:8501` in your browser.


## How Do These Agents Work?
Agents are weightless neural network state machines made up of 3 layers, an input layer, an inner state, and output state. 

The input layer takes in the 128x8 bits, as an an input array of 1024 binary digits .

The inner state layer is a representation of how the agent 'understands' its input.

The output layer is 2 binary digits representing the agent's prediction, which is converted into an integer to match the sentiment labels.

You can click on Training and Testing icon to train and test in the streamlit app.

## File Structure
- arch__sentiment.py - defines how the agent's neural architecture (how many neurons and how they're connected)
- shuffled_embeddings.json - Embeddings converted to int8 using "mixedbread-ai/mxbai-embed-large-v1" model and then converting to int8 using quantization

## Future Work
- We may add the ability to download and upload trained agents in the future
- There are a couple cells in the Font training sets that seem to be causing errors without breaking anything, identifying and fixing those cells would be good

## Contributing
Fork the repo, make your changes and submit a pull request. Join our [discord](https://discord.com/invite/nHuJc4Y4n7) and say hi!