import streamlit as st
import lorax
from lorax.types import Parameters, MERGE_STRATEGIES, ADAPTER_SOURCES, MAJORITY_SIGN_METHODS, MergedAdapters

LORAX_PORT = 8080
HOST = f"http://localhost:{LORAX_PORT}"

def add_merge_adapter(i):
    merge_adapter_id = st.text_input(f"Merge Adapter ID {i+1}", key=f"merge_adapter_id_{i}", value=None, placeholder="Merge Adapter Id")
    merge_adapter_weight = st.number_input(f"Merge Adapter Weight {i+1}", key=f"merge_adapter_weight_{i}", value=None, placeholder="Merge Adapter weight")
    st.divider()
    return merge_adapter_id, merge_adapter_weight

def render_parameters(params: Parameters):
    with st.expander("Request parameters"):
        max_new_tokens = st.slider("Max new tokens", 0, 256, 20)
        repetition_penalty = st.number_input(
            "Repetition penalty",
            help="The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.",
            min_value=0.0,
            value=None,
            placeholder="No repition penalty by default",
        )
        return_full_text = st.checkbox("Return full text", help="Whether to return the full text or just the generated part")
        stop_tokens = st.text_input("Stop tokens", help="A comma seperated list of tokens where generation is stopped if the model encounters any of them")
        stop_sequences = stop_tokens.split(",") if stop_tokens else []
        seed = st.number_input("Seed", help="Random seed for generation", min_value=0, value=None, placeholder="Random seed for generation")
        temperature = st.number_input("Temperature", help="The value used to module the next token probabilities", min_value=0.0, value=None, placeholder="Temperature for generation")
        best_of = st.number_input("Best of", help="The number of independently computed samples to generate and then pick the best from", value=None, placeholder="Best of for generation")
        best_of_int = int(best_of) if best_of else None
        watermark = st.checkbox("Watermark", help="Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)")
        decoder_input_details = st.checkbox("Decoder input details", help="Whether to return the decoder input details")

        do_sample_val = st.checkbox("Do Sample", help="Whether to use sampling or greedy decoding for text generation")
        top_k_int, top_p, typical_p = None, None, None
        if do_sample_val:
            top_k = st.number_input(
                "Top K",
                help="The number of highest probability vocabulary tokens to keep for top-k-filtering",
                value=None,
                placeholder="Top K for generation",
                format="%d"
            )
            top_k_int = int(top_k) if top_k else None
            top_p = st.number_input("Top P", help="The cumulative probability of parameter for top-p-filtering", value=None, placeholder="Top P for generation")
            typical_p = st.number_input(
                "Typical P",
                help="The typical decoding mass. See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information", 
                value=None,
                placeholder="Typical P for generation"
            )
        params.max_new_tokens = max_new_tokens
        params.repetition_penalty = repetition_penalty
        params.return_full_text = return_full_text
        params.stop_sequences = stop_sequences
        params.seed = seed
        params.temperature = temperature
        params.best_of = best_of_int
        params.watermark = watermark
        params.decoder_input_details = decoder_input_details
        params.do_sample = do_sample_val
        params.top_k = top_k_int
        params.top_p = top_p
        params.typical_p = typical_p
        st.write(params)


    with st.expander("Adapter Configuration"):
        adapter_id = st.text_input("Adapter ID", value=None, placeholder="Adapter id")
        adapter_source = st.selectbox("Adapter Source", options=ADAPTER_SOURCES, index=None)
        api_token = st.text_input("API Token", value=None, placeholder="API token")
        if st.checkbox("Merged Adapters"):
            num_adapters = st.slider("Number of Merge Adapters", value=1, min_value=1, max_value=10)
            merge_strategy = st.selectbox("Merge Strategy", options=MERGE_STRATEGIES, index=None)
            majority_sign_method = st.selectbox("Majority Sign Method", options=MAJORITY_SIGN_METHODS, index=None)
            density = st.number_input("Density", value=0.0, placeholder="Density")
            st.divider()
            merge_adapters_list = [add_merge_adapter(i) for i in range(num_adapters)]
            merge_adapter_ids, merge_adapter_weights = zip(*merge_adapters_list)
            merge_adapters = MergedAdapters(
                ids=merge_adapter_ids,
                weights=merge_adapter_weights,
                density=density,
                merge_strategy=merge_strategy,
                majority_sign_method=majority_sign_method,
            )
            params.merged_adapters = merge_adapters
        params.adapter_id = adapter_id
        params.adapter_source = adapter_source
        params.api_token = api_token
        st.write(params)


def main():
    st.markdown(
        r"""
        <style>
        .stDeployButton {
                visibility: hidden;
            }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("Lorax GUI")
    params = Parameters()
    render_parameters(params)

    txt = st.text_area("Enter prompt", "Type Here ...")
    client = lorax.Client(HOST)
    if st.button("Generate"):
        resp = client.generate(prompt=txt, **params.dict())
        st.write(resp)


if __name__ == "__main__":
    main()
