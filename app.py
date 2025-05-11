import streamlit as st
import json
from code_analyzer import load_model, generate_summary, generate_ast, CodeAnalyzer
from ast_generator import analyze_ast
import ast

def main():
    st.title("Python Code Documentation Generator")
    
    # Initialize the model
    @st.cache_resource
    def get_model():
        return load_model()
    
    model = get_model()
    
    # Code input
    code = st.text_area("Enter Python Code:", height=200)
    
    if st.button("Generate Documentation"):
        if code:
            # Generate documentation
            with st.spinner("Generating documentation..."):
                summary = generate_summary(code, model)
                st.subheader("Generated Documentation:")
                tree = ast.parse(code)
                for node in tree.body:
                    st.write(analyze_ast(node))
                st.write(summary)
            
            # Generate and display AST
            with st.spinner("Generating AST..."):
                ast_dict = generate_ast(code)
                st.subheader("Abstract Syntax Tree:")
                st.json(ast_dict)
        else:
            st.warning("Please enter some code first!")

if __name__ == "__main__":
    main()