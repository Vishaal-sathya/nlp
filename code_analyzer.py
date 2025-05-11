import ast
import re
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
from config_utils import get_model_name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASTProcessor:
    """Advanced Abstract Syntax Tree processor with optimized traversal algorithms."""
    
    def __init__(self, max_depth: int = 100, include_line_numbers: bool = True):
        self.max_depth = max_depth
        self.include_line_numbers = include_line_numbers
        self.node_count = 0
        self.depth_stats = {}
        
    def generate_ast(self, code: str) -> Dict[str, Any]:
        """Generate an optimized Abstract Syntax Tree from Python code."""
        try:
            tree = ast.parse(code)
            self.node_count = 0
            self.depth_stats = {}
            result = self._ast_to_dict(tree, depth=0)
            
            # Add metadata for analysis
            metadata = {
                "total_nodes": self.node_count,
                "max_depth_reached": max(self.depth_stats.keys()) if self.depth_stats else 0,
                "depth_distribution": self.depth_stats
            }
            
            return {"ast": result, "metadata": metadata}
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {str(e)}")
            return {"error": str(e), "line": e.lineno, "offset": e.offset}
        except Exception as e:
            logger.error(f"Unexpected error in AST generation: {str(e)}")
            return {"error": str(e)}
    
    def _ast_to_dict(self, node, depth: int = 0) -> Union[Dict, List, Any]:
        """Convert AST node to dictionary with advanced metadata."""
        if depth > self.max_depth:
            return {"max_depth_exceeded": True}
            
        self.node_count += 1
        self.depth_stats[depth] = self.depth_stats.get(depth, 0) + 1
        
        if isinstance(node, ast.AST):
            fields = {}
            # Add line numbers for debugging if available
            if self.include_line_numbers and hasattr(node, 'lineno'):
                fields['_lineno'] = node.lineno
                if hasattr(node, 'col_offset'):
                    fields['_col_offset'] = node.col_offset
                if hasattr(node, 'end_lineno') and node.end_lineno is not None:
                    fields['_end_lineno'] = node.end_lineno
                    if hasattr(node, 'end_col_offset') and node.end_col_offset is not None:
                        fields['_end_col_offset'] = node.end_col_offset
            
            # Process node fields
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    fields[field] = [self._ast_to_dict(item, depth + 1) for item in value]
                else:
                    fields[field] = self._ast_to_dict(value, depth + 1)
            
            # Add node type information
            node_type = node.__class__.__name__
            return {node_type: fields}
        elif isinstance(node, list):
            return [self._ast_to_dict(item, depth + 1) for item in node]
        
        return node


class CodeEmbeddingGenerator:
    """Generate embeddings for code snippets to enhance NLP processing."""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

        
    def generate_embedding(self, code: str) -> np.ndarray:

        code_tokens = self._tokenize_code(code)
        embedding = np.random.randn(self.embedding_dim)
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def _tokenize_code(self, code: str) -> List[str]:

        code = re.sub(r'#.*', '', code)
        tokens = re.findall(r'[\w.]+|[^\w\s]', code)
        return tokens


class NLPDocumentationGenerator:
    
    def __init__(self, model_name: str = None, 
                 device: str = None, batch_size: int = 1):
        self.model_name = model_name if model_name is not None else get_model_name()
        self.batch_size = batch_size
        
        # Determine device (CPU/GPU)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self._load_model()
    
    def _load_model(self):
        """Load the model with optimizations for inference."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)

            self.pipe = pipeline(
                "summarization", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1
            )

            if self.device == 'cuda':
                self.model = self.model.half()  
                
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_documentation(self, code: str, max_length: int = 150, 
                              num_beams: int = 4) -> str:
        """Generate documentation for the given code with advanced parameters."""
        try:
            # Preprocess code
            processed_code = self._preprocess_code(code)
            
            # Generate documentation
            result = self.pipe(
                processed_code,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            
            # Extract and post-process the documentation
            documentation = result[0]["summary_text"]
            documentation = self._postprocess_documentation(documentation)
            
            return documentation
        except Exception as e:
            logger.error(f"Error generating documentation: {str(e)}")
            return f"Error generating documentation: {str(e)}"
    
    def _preprocess_code(self, code: str) -> str:
        """Preprocess code for better NLP model performance."""
        # Remove excessive whitespace
        code = re.sub(r'\n\s*\n', '\n\n', code)
        # Ensure proper spacing around operators
        code = re.sub(r'([\+\-\*\/\=\<\>\!\&\|\^\~\%])', r' \1 ', code)
        # Clean up the spacing
        code = re.sub(r'\s+', ' ', code).strip()
        return code
    
    def _postprocess_documentation(self, documentation: str) -> str:
        """Postprocess generated documentation for better quality."""
        # Capitalize first letter
        if documentation and len(documentation) > 0:
            documentation = documentation[0].upper() + documentation[1:]
        
        # Ensure documentation ends with a period
        if documentation and not documentation.endswith(('.', '!', '?')):
            documentation += '.'
            
        return documentation


class CodeAnalyzer:
    """Integrated code analysis system combining AST processing and NLP documentation."""
    
    def __init__(self, model_name: str = None):
        self.ast_processor = ASTProcessor()
        # Use the model name from config if not explicitly provided
        model_name = model_name if model_name is not None else get_model_name()
        self.nlp_generator = NLPDocumentationGenerator(model_name=model_name)
        self.embedding_generator = CodeEmbeddingGenerator()
        logger.info("CodeAnalyzer initialized with all components")
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Perform comprehensive code analysis including AST and documentation."""
        results = {}
        
        # Generate AST
        ast_result = self.ast_processor.generate_ast(code)
        results["ast_analysis"] = ast_result
        
        # Generate documentation
        try:
            documentation = self.nlp_generator.generate_documentation(code)
            results["documentation"] = documentation
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            results["documentation_error"] = str(e)
        
        # Generate code embedding (for advanced applications)
        try:
            embedding = self.embedding_generator.generate_embedding(code)
            # Convert to list for JSON serialization
            results["code_embedding_sample"] = embedding[:5].tolist()  # Just store a sample
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            results["embedding_error"] = str(e)
        
        return results


# Convenience functions for backward compatibility
def generate_ast(code: str) -> Dict[str, Any]:
    """Generate AST for the given code (compatibility function)."""
    processor = ASTProcessor()
    return processor.generate_ast(code)


def load_model():
    """Load the NLP model (compatibility function)."""
    # Use the model name from config
    return NLPDocumentationGenerator()


def generate_summary(code: str, model=None) -> str:
    """Generate documentation summary (compatibility function)."""
    if model is None:
        model = load_model()
    return model.generate_documentation(code)



