import streamlit as st
import json
import numpy as np
import pandas as pd
import ao_core as ao
from arch__sentiment import arch
# from config import openai_api_key

# Initialize global variables
if "training_history" not in st.session_state:
    st.session_state.training_history = []
if "test_history" not in st.session_state:
    st.session_state.test_history = []
if "agent" not in st.session_state:
    print("-------creating agent-------")
    st.session_state.agent = ao.Agent(arch, notes="Sentiment Analysis Agent")
    
    # Initialize the agent with random inputs to seed training
    for _ in range(4):
        random_input = np.random.randint(0, 2, arch.Q__flat.shape, dtype=np.int8)
        random_label = np.random.randint(0, 2, arch.Z__flat.shape, dtype=np.int8)
        st.session_state.agent.reset_state()
        st.session_state.agent.next_state(INPUT=random_input, LABEL=random_label)

# Load data from JSON file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['train'], data['test']

train_data, test_data = load_data('shuffled_embeddings.json')

# Function to convert int8 embeddings to binary
def convert_embeddings_to_binary(embeddings):
    embeddings_np = np.array(embeddings, dtype=np.int8)
    binary_embeddings = np.unpackbits(embeddings_np.view(np.uint8))
    print(binary_embeddings.shape)
    return binary_embeddings

# Function to encode labels into binary
def encode_label(label):
    label_mapping = {
        "positive": [1, 1],
        "negative": [0, 1],
        "neutral": [0, 0],
        "no_sentiment": [1, 0]
    }
    return label_mapping.get(label.lower(), [0, 0])

# Training function
def train_agent():
    count = 0
    for sample in train_data[50:100]:
        binary_input = convert_embeddings_to_binary(sample['embeddings'])
        label = encode_label(sample['label'])
        # Pad the label to match arch_z size if necessary
        count+=1
        print("The data point is getting trained. Data Point :   ", count)
        # if len(label) < arch.Z__flat.size:
        #     label += [0] * (arch.Z__flat.size - len(label))
        label = np.array(label, dtype=np.int8)
        st.session_state.agent.reset_state()
        st.session_state.agent.next_state(INPUT=binary_input, LABEL=label, print_result=False)
        # Record training history
        st.session_state.training_history.append({
            "text": sample['text'],
            "label": sample['label']
        })

# Testing function
def test_agent():
    correct = 0
    total = len(test_data)
    for sample in test_data[50:100]:
        binary_input = convert_embeddings_to_binary(sample['embeddings'])
        true_label = sample['label']
        print("--------------")
        print(true_label)
        st.session_state.agent.reset_state()
        st.session_state.agent.next_state(INPUT=binary_input, print_result=False)
        response = st.session_state.agent.story[st.session_state.agent.state-1, arch.Z__flat]
        print(response)
        predicted_label = interpret_response(response)
        st.session_state.test_history.append({
            "text": sample['text'],
            "true_label": true_label,
            "predicted_label": predicted_label
        })
        if predicted_label.lower() == true_label.lower():
            correct += 1
        print(f'Corrected samples are {correct} out of 50')
    accuracy = (correct / 50) * 100
    return accuracy

# Function to interpret agent's response
def interpret_response(response):
    # Assuming response is a binary array of length 4
    # Map binary to sentiment
    sentiment_mapping = {
        (1,1): "positive",
        (0,1): "negative",
        (0,0): "neutral",
        (1,0): "no_sentiment"
    }
    response_tuple = tuple(response)
    return sentiment_mapping.get(response_tuple, "unknown")

# Streamlit UI Configuration
st.set_page_config(
    page_title="Sentiment Analysis with AO Labs",
    page_icon="misc/ao_favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://discord.gg/Zg9bHPYss5",
        "Report a bug": "mailto:eng@aolabs.ai",
        "About": "AO Labs builds next-gen AI models that learn after training; learn more at docs.aolabs.ai/docs/mnist-benchmark",
    },
)

# Sidebar for Agent Management
with st.sidebar:
    st.write("## Current Active Agent:")
    st.write(st.session_state.agent.notes)

    st.write("---")
    st.write("## Load Agent:")

    def load_pickle_files(directory):
        import os
        pickle_files = [
            f[:-10] for f in os.listdir(directory) if f.endswith(".ao.pickle")
        ]
        return pickle_files

    import os
    directory = os.path.dirname(os.path.abspath(__file__))

    if directory:
        pickle_files = load_pickle_files(directory)

        if pickle_files:
            selected_file = st.selectbox(
                "Choose from saved Agents:", options=pickle_files
            )

            if st.button(f"Load {selected_file}"):
                file_path = os.path.join(directory, selected_file + ".ao.pickle")
                st.session_state.agent = ao.Agent.unpickle(
                    file=file_path, custom_name=selected_file
                )
                st.session_state.agent._update_neuron_data()
                st.write("Agent loaded")
        else:
            st.warning("No Agents saved yet-- be the first!")

    st.write("---")
    st.write("## Save Agent:")

    agent_name = st.text_input(
        "## *Optional* Rename active Agent:", value=st.session_state.agent.notes
    )
    st.session_state.agent.notes = agent_name

    @st.dialog("Save successful!")
    def save_modal_dialog():
        st.write("Agent saved to your local disk (in the same directory as this app).")

    agent_name_clean = agent_name.split("\\")[-1].split(".")[0]
    if st.button("Save " + agent_name_clean):
        st.session_state.agent.pickle(agent_name_clean)
        save_modal_dialog()

    st.write("---")
    st.write("## Download/Upload Agents:")

    @st.dialog("Upload successful!")
    def upload_modal_dialog():
        st.write(
            "Agent uploaded and ready as *Newly Uploaded Agent*, which you can rename during saving."
        )

    uploaded_file = st.file_uploader(
        "Upload .ao.pickle files here", label_visibility="collapsed"
    )
    if uploaded_file is not None:
        if st.button("Confirm Agent Upload"):
            st.session_state.agent = ao.Agent.unpickle(
                uploaded_file, custom_name="Newly Uploaded Agent", upload=True
            )
            st.session_state.agent._update_neuron_data()
            upload_modal_dialog()

    @st.dialog("Download ready")
    def download_modal_dialog(agent_pickle):
        st.write(
            "The Agent's .ao.pickle file will be saved to your default Downloads folder."
        )

        # Create a download button
        st.download_button(
            label="Download Agent: " + st.session_state.agent.notes,
            data=agent_pickle,
            file_name=st.session_state.agent.notes + ".ao.pickle",
            mime="application/octet-stream",
        )

    if st.button("Prepare Active Agent for Download"):
        agent_pickle = st.session_state.agent.pickle(download=True)
        download_modal_dialog(agent_pickle)

# Main Application Layout
st.title("AO Labs - Sentiment Analysis with Weightless Neural Networks")
st.write("### *A preview by [aolabs.ai](https://www.aolabs.ai/)*")

left_col, right_col = st.columns([0.4, 0.6], gap="large")

with left_col:
    st.header("Train the Agent")
    if st.button("Start Training"):
        train_agent()
        st.success("Training completed!")
        st.write(f"Total training samples: {len(st.session_state.training_history)}")

    st.divider()
    st.header("Test the Agent")
    if st.button("Run Testing"):
        accuracy = test_agent()
        st.success(f"Testing completed! Accuracy: {accuracy:.2f}%")
        st.write(f"Total test samples: {len(st.session_state.test_history)}")

    st.divider()
    with st.expander("### Training History"):
        if st.session_state.training_history:
            df_train = pd.DataFrame(st.session_state.training_history)
            st.dataframe(df_train)
        else:
            st.write("No training history available.")

    with st.expander("### Test Results"):
        if st.session_state.test_history:
            df_test = pd.DataFrame(st.session_state.test_history)
            st.dataframe(df_test)
        else:
            st.write("No test results available.")

with right_col:
    st.header("Agent Status")
    st.write("**Current Active Agent:** ", st.session_state.agent.notes)
    st.write("**Architecture:** ", arch.description)
    st.write("**Input Length:** ", sum(arch_i := [8 for _ in range(128)]))
    st.write("**Output Classes:** Positive, Negative, Neutral, No Sentiment")

    st.divider()
    with st.expander("### Agent Overview"):
        st.write("""
            This sentiment analysis agent uses a weightless neural network architecture to classify text based on precomputed embeddings.
            The embeddings are binary-encoded and fed into the agent for training and testing.
        """)

    st.divider()
    with st.expander("### Upload New Data"):
        st.write("Currently, the system uses preloaded embeddings from `data.json`.")
        st.write("To use new data, update the `data.json` file accordingly.")

# Footer
st.write("---")
footer_md = """
[View & fork the code behind this application here.](https://github.com/aolabsai/Recommender) \n
To learn more about Weightless Neural Networks and the new generation of AI we're developing at AO Labs, [visit our docs.aolabs.ai.](https://docs.aolabs.ai/)\n
\n
We eagerly welcome contributors and hackers at all levels! [Say hi on our discord.](https://discord.gg/Zg9bHPYss5)
"""
st.markdown(footer_md)
st.image("misc/aolabs-logo-horizontal-full-color-white-text.png", width=300)
