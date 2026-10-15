"""Code splitter."""
import re
from typing import Any, List
from loguru import logger
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.node_parser.interface import TextSplitter

DEFAULT_COST = 10
DEFAULT_LINES_OVERLAP = 15
DEFAULT_MAX_WINDOW_SIZE = 512 * 4 / 100
types_related_to_class = ['class_definition', 'class_specifier', 'class_declaration', 'enum_specifier',
                          'interface_declaration', 'struct_type']
types_related_to_function = ['function_definition', 'function_declaration', 'generator_function_declaration',
                             'method_definition']
types_of_interest = types_related_to_function + types_related_to_class
type_cost = {item: 3 for item in types_related_to_class}
type_cost.update({item: 5 for item in types_related_to_function})


class DynamicCodeSplitter():
    def __init__(self, language: str, general_cost: int = DEFAULT_COST, max_window_size: int = DEFAULT_MAX_WINDOW_SIZE,
                 **data: Any):
        self.language = language
        self.general_cost = general_cost
        self.max_window_size = max_window_size

    @classmethod
    def class_name(cls) -> str:
        return "DynamicCodeSplitter"

    def find_nodes(self, root_node, text):
        splits = self.extract_nodes(root_node)
        text_arrays = text.split("\n")
        splits = self._min_cost_split(splits=splits, max_length=len(text_arrays),
                                      max_window_size=self.max_window_size,
                                      general_cost=self.general_cost)
        text_chunks = []
        for start, end in splits[0]:
            text_chunks.append("\n".join(text_arrays[start:end + 1]))
        return text_chunks

    @staticmethod
    def _min_cost_split(splits, max_length, max_window_size, general_cost):
        # Convert splits to a dictionary for fast access
        split_cost = {point: cost for point, cost in splits}

        # Initialize DP array for storing minimum cost and corresponding break point
        dp = [(0, -1)] * (max_length + 1)  # (cost, break point)

        # Calculate minimum cost for all subproblems
        for i in range(1, max_length + 1):
            min_cost = float('inf')
            break_point = -1

            # Consider all possible previous breakpoints
            for j in range(max(i - max_window_size, 0), i):
                # Calculate the cost for breaking at j
                cost = dp[j][0] + split_cost.get(i, general_cost)

                if cost < min_cost:
                    min_cost = cost
                    break_point = j

            # Update DP array with the minimum cost and break point for i
            dp[i] = (min_cost, break_point)

        # Backtrack to find the break points and calculate total cost
        breakpoints = []
        current = max_length
        total_cost = dp[max_length][0]  # total cost is the minimum cost to split the entire text
        while current > 0:
            prev_break = dp[current][1]
            breakpoints.append((prev_break, current - 1))  # -1 because end is inclusive
            current = prev_break

        breakpoints.reverse()  # to display from start to end
        return breakpoints, total_cost

    def extract_nodes(self, node, result=None, depth=0):
        if result is None:
            result = []
        if depth > 2000:  # Check if the current recursion depth exceeds the maximum allowed depth
            return result  # Stop recursion if the maximum depth is reached

        # Check if the node is a type of interest and not an anonymous function
        if node.type in types_of_interest and 'lambda' not in node.text.decode('utf8'):
            line_no = node.start_point[0] + 1  # Adjusting because line numbers are zero-indexed
            node_cost = type_cost[node.type]
            result.append((line_no, node_cost))
        # Recursively visit children
        for child in node.children:
            self.extract_nodes(child, result, depth + 1)  # Increment the depth
        return result

    def split_text(self, text: str) -> List[str]:
        """Split incoming code and return chunks using the AST."""
        try:
            import tree_sitter_languages
        except ImportError:
            raise ImportError(
                "Please install tree_sitter_languages to use CodeSplitter."
            )

        try:
            parser = tree_sitter_languages.get_parser(self.language)
        except Exception:
            print(
                f"Could not get parser for language {self.language}. Check "
                "https://github.com/grantjenks/py-tree-sitter-languages#license "
                "for a list of valid languages."
            )
            raise

        tree = parser.parse(bytes(text, "utf-8"))

        if (
                not tree.root_node.children
                or tree.root_node.children[0].type != "ERROR"
        ):
            chunks = [chunk.strip() for chunk in self.find_nodes(tree.root_node, text)]

            return chunks
        else:
            logger.error(f"Could not parse code with language {self.language}.")
            return chunk_text(text, 2000)


def chunk_text(text: str, chunk_size: int = 2000) -> List[str]:
    """
    Chunk text into smaller parts of specified size.

    Args:
    text (str): The string of text to be chunked.
    chunk_size (int): The size of each chunk, default is 4000 characters.

    Returns:
    List[str]: A list of text chunks.
    """
    # Initialize an empty list to hold chunks
    chunks = []

    # Loop through the text, stepping by chunk_size each time
    for i in range(0, len(text), chunk_size):
        # Append a chunk of text to the list
        chunks.append(text[i: i + chunk_size])

    return chunks


class CommentCleaner:
    def __init__(self, language):
        self.language = language
        try:
            import tree_sitter_languages
        except ImportError:
            raise ImportError(
                "Please install tree_sitter_languages to use CommentCleaner."
            )

        try:
            self.parser = tree_sitter_languages.get_parser(self.language)
        except Exception:
            print(
                f"Could not get parser for language {self.language}. Check "
                "https://github.com/grantjenks/py-tree-sitter-languages#license "
                "for a list of valid languages."
            )
            raise

    @staticmethod
    def merge_newlines(text):
        # Replace multiple newlines with a single newline
        return re.sub(r"\n+", "\n", text)
        # parts = filter(None, text.split('\n'))
        #
        # # Join the non-empty parts back together with a single newline
        # return '\n'.join(parts)

    @staticmethod
    def extract_comments(root_node, code):
        comments = []
        stack = [root_node]  # Initialize stack with the root node

        while stack:  # Process nodes until the stack is empty
            node = stack.pop()  # Pop a node from the stack
            if "comment" in node.type:
                comments.append(code[node.start_byte:node.end_byte])
            # Add child nodes to the stack to be processed
            stack.extend(node.children[::-1])  # Reverse children to maintain original order

        return comments

    def remove_comments_from_code(self, code):
        comments = self.extract_comments(self.parser.parse(bytes(code, "utf-8")).root_node, code)
        for comment in sorted(comments, key=len, reverse=True):
            comment = comment.replace("\n", "").strip()
            if (self.language == 'python' and len(comment) > 2) or self.language != 'python':
                code = code.replace(comment.replace("\n", "").strip(), '')
        code = self.merge_newlines(code)
        return code


if __name__ == "__main__":
    splitter = DynamicCodeSplitter(language='java', general_cost=50, max_window_size=30)
    source_code = """"""

    # temp = splitter.split_text(text=source_code)
    # print(temp)
    cleaner = CommentCleaner(language='python')
    print(cleaner.remove_comments_from_code(source_code))
