"""Actual pinned scheduler copy callbacks remain unchanged under the v3 patch."""
import importlib.util,subprocess,tempfile
from pathlib import Path
from helpers import ROOT
from test_c2_fake_backend import FakeBackendTests as OriginalTests
class TimingBackendTests(OriginalTests):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='c2-v3-backend-');cls.path=Path(cls.temp.name)
        spec=importlib.util.spec_from_file_location('v3patch',ROOT/'instrumentation/v3/build_patch.py')
        p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
        values=p.patched()
        for name,value in values.items():
            if name.startswith('ggml/src/') and '/' not in name[len('ggml/src/'):]: (cls.path/Path(name).name).write_text(value)
        source=cls.path/'ggml-backend.cpp';cls.binary=cls.path/'fake';cls.original=cls.path/'original'
        for implementation,binary in [(source,cls.binary),(ROOT/'llama.cpp/ggml/src/ggml-backend.cpp',cls.original)]:
            subprocess.run(['c++','-std=c++17','-O1','-pthread','-DHETEROLLM_CPU_TRACE=1','-DHETEROLLM_PATCHED_SOURCE="'+str(implementation)+'"',
                '-I',str(ROOT/'llama.cpp/ggml/include'),'-I',str(ROOT/'llama.cpp/ggml/src'),str(ROOT/'tests/fixtures/c2_fake_backend.cpp'),
                '-L',str(ROOT/'llama.cpp/build-mixed/bin'),'-Wl,-rpath,'+str(ROOT/'llama.cpp/build-mixed/bin'),'-lggml-base','-o',str(binary)],check=True,capture_output=True,text=True,timeout=45)
del OriginalTests
