"""Explicit, offline Tree-sitter grammar registry for source file extensions."""

from functools import lru_cache
from importlib import import_module
from importlib.metadata import version
from pathlib import PurePosixPath

GRAMMARS = {
    'python': ('tree_sitter_python', 'language'),
    'java': ('tree_sitter_java', 'language'),
    'javascript': ('tree_sitter_javascript', 'language'),
    'typescript': ('tree_sitter_typescript', 'language_typescript'),
    'tsx': ('tree_sitter_typescript', 'language_tsx'),
    'c': ('tree_sitter_c', 'language'),
    'cpp': ('tree_sitter_cpp', 'language'),
    'csharp': ('tree_sitter_c_sharp', 'language'),
    'go': ('tree_sitter_go', 'language'),
    'rust': ('tree_sitter_rust', 'language'),
}
EXTENSIONS = {
    '.py': 'python', '.pyi': 'python', '.java': 'java',
    '.js': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript', '.jsx': 'javascript',
    '.ts': 'typescript', '.mts': 'typescript', '.cts': 'typescript', '.tsx': 'tsx',
    '.c': 'c', '.h': 'c', '.cc': 'cpp', '.cpp': 'cpp', '.cxx': 'cpp',
    '.hpp': 'cpp', '.hh': 'cpp', '.hxx': 'cpp', '.cs': 'csharp', '.go': 'go', '.rs': 'rust',
}
CLASS_NODES = {
    'python': {'class_definition'},
    'java': {'class_declaration', 'interface_declaration', 'enum_declaration', 'record_declaration', 'annotation_type_declaration'},
    'javascript': {'class_declaration', 'class'},
    'typescript': {'class_declaration', 'class', 'abstract_class_declaration', 'interface_declaration', 'enum_declaration'},
    'tsx': {'class_declaration', 'class', 'abstract_class_declaration', 'interface_declaration', 'enum_declaration'},
    'c': {'struct_specifier', 'union_specifier', 'enum_specifier'},
    'cpp': {'class_specifier', 'struct_specifier', 'union_specifier', 'enum_specifier'},
    'csharp': {'class_declaration', 'struct_declaration', 'interface_declaration', 'record_declaration', 'enum_declaration'},
    'go': {'type_spec'},
    'rust': {'impl_item', 'trait_item', 'struct_item', 'enum_item'},
}
FUNCTION_NODES = {
    'python': {'function_definition'},
    'java': {'method_declaration', 'constructor_declaration', 'compact_constructor_declaration'},
    'javascript': {'function_declaration', 'function_expression', 'generator_function_declaration', 'generator_function', 'method_definition', 'arrow_function'},
    'typescript': {'function_declaration', 'function_expression', 'generator_function_declaration', 'generator_function', 'method_definition', 'arrow_function', 'method_signature', 'abstract_method_signature'},
    'tsx': {'function_declaration', 'function_expression', 'generator_function_declaration', 'generator_function', 'method_definition', 'arrow_function', 'method_signature', 'abstract_method_signature'},
    'c': {'function_definition'}, 'cpp': {'function_definition'},
    'csharp': {'method_declaration', 'constructor_declaration', 'local_function_statement', 'destructor_declaration', 'operator_declaration'},
    'go': {'function_declaration', 'method_declaration'},
    'rust': {'function_item', 'function_signature_item'},
}


def language_for_path(path, overrides=None):
    extension = PurePosixPath(path).suffix.lower()
    language = (overrides or {}).get(extension, EXTENSIONS.get(extension))
    if language not in GRAMMARS:
        raise ValueError('No registered AST grammar for ' + str(path))
    return language


def validate_language_config(extensions, overrides):
    if not isinstance(overrides, dict) or any(
            not isinstance(ext, str) or not ext.startswith('.') or ext != ext.lower() or language not in GRAMMARS
            for ext, language in overrides.items()):
        raise ValueError('language_overrides must map lowercase extensions to registered grammar names')
    for extension in extensions:
        language_for_path('source' + extension, overrides)


@lru_cache(maxsize=None)
def parser_for_language(language):
    from tree_sitter import Language, Parser
    module, factory = GRAMMARS[language]
    try:
        grammar = getattr(import_module(module), factory)()
    except ImportError as error:
        raise ImportError('Install ' + module.replace('_', '-') + ' from requirements.txt') from error
    return Parser(Language(grammar))


def grammar_versions():
    return {module.replace('_', '-'): version(module.replace('_', '-'))
            for module, _ in set(GRAMMARS.values())}
