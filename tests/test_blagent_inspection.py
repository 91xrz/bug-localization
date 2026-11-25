"""Exercise the actual LangChain error-observation and bounded-loop behavior."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/p) for p in ('.runtime/blagent-deps','.runtime/rrl-deps','.runtime/igsil-deps','src')]
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_core.language_models.fake import FakeListLLM
from bug_localization.blagent_recovery import inspection_required_parser,guarded_tool


class InspectionTests(unittest.TestCase):
    def execute(self,folder,replies):
        inspected=[]
        tool=Tool(name='ReadFileSkeleton',description='Read a candidate skeleton',
                  func=guarded_tool(lambda p:'def actual_function()', ['a.py'],inspected,[],folder))
        agent=initialize_agent([tool],FakeListLLM(responses=replies),agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                               handle_parsing_errors=True,max_iterations=3)
        agent.agent.output_parser=inspection_required_parser(agent.agent.output_parser,inspected,folder)
        return agent.invoke({'input':'Inspect a.py and rank it.'})['output'],inspected

    def test_early_final_recovers_through_real_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            output,inspected=self.execute(folder,['Final Answer: {"a.py":9}',
                'Thought: inspect\nAction: ReadFileSkeleton\nAction Input: a.py','Final Answer: {"a.py":8}'])
            self.assertEqual(json.loads(output),{'a.py':8})
            self.assertEqual(inspected,[{'file_path':'a.py','skeleton':'def actual_function()'}])
            self.assertEqual(len(json.loads((folder/'early_final_rejections.json').read_text())),1)

    def test_repeated_early_final_still_honors_iteration_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            output,inspected=self.execute(folder,['Final Answer: {"a.py":9}'])
            self.assertIn('iteration limit',output)
            self.assertEqual(inspected,[])
            self.assertEqual(len(json.loads((folder/'early_final_rejections.json').read_text())),3)

if __name__=='__main__':unittest.main()
