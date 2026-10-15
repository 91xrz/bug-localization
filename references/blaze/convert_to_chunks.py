from datasets import Dataset, load_dataset, load_from_disk
from copy import deepcopy
from utils.processor import DynamicCodeSplitter, CommentCleaner
from tqdm import tqdm
import os

if __name__ == "__main__":
    ds = load_dataset("oss_bugs_with_content")
    df = ds['train'].to_pandas()
    splitter_map = dict()
    cleaner_map = dict()
    extension_map = {
        "python": [".py"],
        "java": [".java"],
        "cpp": [".cpp", ".cxx", ".cc", ".h"],
        "javascript": [".js", ".ts"]
    }
    all_data = []
    for item in tqdm(df.iterrows(), total=len(df)):
        text_content = item[1]['file_content']
        file_extension = os.path.splitext(item[1]['updated_file'])
        language = item[1]['language']
        if len(file_extension[1]) >= 2 and file_extension[1] in extension_map[language]:
            if language in splitter_map.keys():
                splitter = splitter_map[language]
                cleaner = cleaner_map[language]
            else:
                splitter = DynamicCodeSplitter(language=language, general_cost=50, max_window_size=30)
                cleaner = CommentCleaner(language=language)
                splitter_map[language] = splitter
                cleaner_map[language] = cleaner
            cleaned_content = cleaner.remove_comments_from_code(item[1]['file_content'])
            chunk_list = splitter.split_text(cleaned_content)
            for chunk in chunk_list:
                generic_chunk = deepcopy(item[1].to_dict())
                cleaned_content = cleaner.remove_comments_from_code(chunk)
                generic_chunk['chunk_content'] = chunk
                del generic_chunk['file_content']
                all_data.append(generic_chunk)
    new_ds = Dataset.from_list(all_data)
    new_ds.push_to_hub("oss_bugs_with_chunks")