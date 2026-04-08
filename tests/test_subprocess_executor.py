"""
SubprocessCodeExecutor 单元测试
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from automl_react.utils.subprocess_executor import SubprocessCodeExecutor, SubprocessExecutionResult


class TestSubprocessCodeExecutor(unittest.TestCase):
    """SubprocessCodeExecutor 测试"""

    def setUp(self):
        self.executor = SubprocessCodeExecutor(timeout=30)

    def test_simple_code_execution(self):
        """测试简单代码执行"""
        result = self.executor.execute(
            code='x = 1 + 2\nprint(f"result: {x}")',
        )
        self.assertTrue(result.success)
        self.assertIn("result: 3", result.output)
        self.assertEqual(result.return_code, 0)
        self.assertFalse(result.timed_out)

    def test_variable_passback(self):
        """测试变量回传"""
        result = self.executor.execute(
            code='result = {"a": 1, "b": [2, 3]}',
            required_output_names=["result"],
        )
        self.assertTrue(result.success)
        self.assertIn("result", result.variables)
        self.assertEqual(result.variables["result"], {"a": 1, "b": [2, 3]})

    def test_context_injection(self):
        """测试上下文变量注入"""
        result = self.executor.execute(
            code='result = data_path + "/output.csv"',
            context={"data_path": "/tmp/test"},
            required_output_names=["result"],
        )
        self.assertTrue(result.success)
        self.assertEqual(result.variables["result"], "/tmp/test/output.csv")

    def test_timeout(self):
        """测试超时终止"""
        executor = SubprocessCodeExecutor(timeout=2)
        result = executor.execute(
            code='import time\nwhile True:\n    time.sleep(0.1)',
        )
        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)
        self.assertIn("超时", result.error)

    def test_syntax_error(self):
        """测试语法错误捕获"""
        result = self.executor.execute(
            code='def foo(\n    # missing closing paren',
        )
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertFalse(result.timed_out)

    def test_runtime_error(self):
        """测试运行时错误捕获"""
        result = self.executor.execute(
            code='x = 1 / 0',
        )
        self.assertFalse(result.success)
        self.assertIn("ZeroDivisionError", result.error)

    def test_import_error(self):
        """测试导入错误"""
        result = self.executor.execute(
            code='import nonexistent_module_abc123',
        )
        self.assertFalse(result.success)
        self.assertIn("ModuleNotFoundError", result.error)

    def test_csv_file_output(self):
        """测试子进程写文件，主进程能读到"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "output.csv")
            code = f'''
import pandas as pd
df = pd.DataFrame({{"a": [1, 2, 3], "b": [4, 5, 6]}})
df.to_csv("{csv_path}", index=False)
print(f"Saved {{len(df)}} rows")
'''
            result = self.executor.execute(code=code)
            self.assertTrue(result.success)
            self.assertTrue(os.path.isfile(csv_path))
            import pandas as pd
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 3)
            self.assertIn("Saved 3 rows", result.output)

    def test_pandas_dataframe_passback(self):
        """测试 DataFrame 通过 pickle 回传"""
        result = self.executor.execute(
            code='import pandas as pd\ndf = pd.DataFrame({"x": [1, 2, 3]})',
        )
        self.assertTrue(result.success)
        self.assertIn("df", result.variables)
        import pandas as pd
        self.assertIsInstance(result.variables["df"], pd.DataFrame)
        self.assertEqual(len(result.variables["df"]), 3)

    def test_numpy_array_passback(self):
        """测试 numpy 数组回传"""
        result = self.executor.execute(
            code='import numpy as np\nX = np.array([1, 2, 3])',
            required_output_names=["X"],
        )
        self.assertTrue(result.success)
        self.assertIn("X", result.variables)

    def test_multiple_context_types(self):
        """测试多种类型的上下文变量"""
        result = self.executor.execute(
            code='result = f"{name}_{count}_{flag}"',
            context={
                "name": "test",
                "count": 42,
                "flag": True,
            },
            required_output_names=["result"],
        )
        self.assertTrue(result.success)
        self.assertEqual(result.variables["result"], "test_42_True")

    def test_stderr_capture(self):
        """测试 stderr 捕获"""
        result = self.executor.execute(
            code='import sys\nprint("err msg", file=sys.stderr)\nraise ValueError("test error")',
        )
        self.assertFalse(result.success)
        self.assertIn("ValueError", result.error)

    def test_empty_code(self):
        """测试空代码执行"""
        result = self.executor.execute(code='pass')
        self.assertTrue(result.success)

    def test_result_type(self):
        """测试返回类型"""
        result = self.executor.execute(code='x = 1')
        self.assertIsInstance(result, SubprocessExecutionResult)


class TestSubprocessCodeExecutorWithWorkingDir(unittest.TestCase):
    """测试工作目录功能"""

    def test_working_dir(self):
        """测试指定工作目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = SubprocessCodeExecutor(timeout=10)
            result = executor.execute(
                code='import os\ncwd = os.getcwd()',
                required_output_names=["cwd"],
                working_dir=tmpdir,
            )
            self.assertTrue(result.success)
            # macOS /var -> /private/var symlink
            self.assertEqual(
                os.path.realpath(result.variables["cwd"]),
                os.path.realpath(tmpdir),
            )


if __name__ == "__main__":
    unittest.main()
