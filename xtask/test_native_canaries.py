import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from native_canaries import case_assertions, retain_admission_files


class NativeResults(unittest.TestCase):
    def test_cleanup_retains_exact_admission_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            keep = ["CMakeCache.txt", "linked-archives.txt", "lib/libmlx.a", "canary"]
            discard = ["obj/array.o", "upstream-tests/tests", "metallib/mlx.metallib"]
            for name in keep + discard:
                path = build / name
                path.parent.mkdir(exist_ok=True)
                path.write_text(name)
            report = {"linked_exports": {"archives": [{"path": str(build / "lib/libmlx.a")}],
                                         "canary": {"path": str(build / "canary")}}}
            retain_admission_files(build, report)
            self.assertEqual(sorted(str(p.relative_to(build)) for p in build.rglob("*") if p.is_file()), sorted(keep))
            self.assertFalse((build / "obj").exists())

    def result(self):
        return ET.fromstring('''<doctest><TestSuite>
          <TestCase name="other" skipped="true"/>
          <TestCase name="test grad"/>
          </TestSuite><OverallResultsTestCases successes="1" failures="0" skipped="250"/>
          <OverallResultsAsserts successes="31" failures="0"/></doctest>''')

    def test_accepts_only_the_named_case_with_assertions(self):
        self.assertEqual(case_assertions(self.result(), "test grad"), 31)

    def test_rejects_missing_renamed_skipped_duplicate_empty_and_failed_cases(self):
        for field, attribute, value in [
            ("TestSuite/TestCase[@name='test grad']", "name", "renamed"),
            ("TestSuite/TestCase[@name='test grad']", "skipped", "true"),
            ("OverallResultsAsserts", "successes", "0"),
            ("OverallResultsAsserts", "failures", "1"),
            ("OverallResultsTestCases", "failures", "1"),
        ]:
            with self.subTest(field=field, attribute=attribute):
                result = self.result()
                result.find(field).set(attribute, value)
                with self.assertRaises(ValueError):
                    case_assertions(result, "test grad")
        for result in [ET.fromstring("<doctest/>"), self.result()]:
            ET.SubElement(result.find("TestSuite") if result.find("TestSuite") is not None
                          else result, "TestCase", name="test grad")
            with self.assertRaises(ValueError):
                case_assertions(result, "test grad")


if __name__ == "__main__":
    unittest.main()
