import streamlit as st
from stuff import summarize_with_stuff
from map import summarize_with_map

def main():
    st.title("📄 Blog Summarizer with LangChain & Groq")

    url = st.text_input(
        "Enter a blog/article URL:",
        "https://lilianweng.github.io/posts/2023-06-23-agent/"
    )

    # ✅ Choose which summarizer(s) to run
    st.subheader("Choose summarization methods:")
    use_stuff = st.checkbox("Stuff Method")
    use_map = st.checkbox("Map-Reduce Method")

    if st.button("Summarize"):
        if not url:
            st.warning("⚠ Please enter a valid URL")
            return
        
        if not (use_stuff or use_map):
            st.warning("⚠ Please select at least one summarization method")
            return

        results = {}

        # ✅ Run Stuff Method only if selected
        if use_stuff:
            with st.spinner("⏳ Summarizing using Stuff Method..."):
                try:
                    results["Stuff Method"] = summarize_with_stuff(url)
                except Exception as e:
                    results["Stuff Method"] = f"❌ Error: {str(e)}"

        # ✅ Run Map-Reduce Method only if selected
        if use_map:
            with st.spinner("⏳ Summarizing using Map-Reduce Method..."):
                try:
                    results["Map-Reduce Method"] = summarize_with_map(url)
                except Exception as e:
                    results["Map-Reduce Method"] = f"❌ Error: {str(e)}"

        # ✅ Show Results in Columns if Both Selected
        if len(results) == 2:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Stuff Method Summary")
                st.write(results["Stuff Method"])
            with col2:
                st.subheader("Map-Reduce Summary")
                st.write(results["Map-Reduce Method"])
        else:
            # ✅ Show single result if only one method selected
            for method, summary in results.items():
                st.subheader(method)
                st.write(summary)

if __name__ == "__main__":
    main()
