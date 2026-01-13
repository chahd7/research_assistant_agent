import streamlit as st
from agent import ResearchAgent

# Load agent once
@st.cache_resource
def load_agent():
    return ResearchAgent()

agent = load_agent()

# App title
st.title("Research Agent Explorer")
st.write("Ask anything and get summarized results from your agent!")

# Query input
query = st.text_input("Enter your research question:")
run_query = st.button("Run Query")

if run_query and query:
    with st.spinner("Running your query..."):
        out = agent.run(query)

    # Display top passages in collapsible sections
    st.subheader("Top Passages")
    for i, p in enumerate(out["passages"], 1):
        with st.expander(f"Passage {i} | Score: {p['score']:.3f} | Source: {p['url']}"):
            st.write(p['passage'])

    # Display summary
    st.subheader("Extractive Summary")
    st.text_area("Summary", value=out["summary"], height=150)

    # Processing time
    st.success(f"Query processed in {out['time']:.1f}s")
