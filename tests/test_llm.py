"""LLM 封装的单元测试（chat / chat_stream / 计量）。"""
from agent_project import llm



class TestChatStream:
    def test_stream_yields_tokens_and_records_usage(self, monkeypatch):
        """chat_stream 逐块产出文本，末块 usage 照常落计量。"""
        class _Delta:
            def __init__(self, c): self.content = c
        class _Choice:
            def __init__(self, c): self.delta = _Delta(c)
        class _Usage:
            prompt_tokens = 10
            completion_tokens = 2
            total_tokens = 12
        class _Chunk:
            def __init__(self, c=None, usage=None):
                self.choices = [_Choice(c)] if c else []
                self.usage = usage

        chunks = [_Chunk("你好"), _Chunk("呀"), _Chunk(usage=_Usage())]

        class _Completions:
            @staticmethod
            def create(**kwargs):
                assert kwargs.get("stream") is True
                assert kwargs.get("stream_options", {}).get("include_usage") is True
                return iter(chunks)

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        monkeypatch.setattr(llm, "_client", _Client())
        llm.reset_usage()
        out = "".join(llm.chat_stream([{"role": "user", "content": "hi"}]))
        assert out == "你好呀"
        assert llm.usage_summary()["total_tokens"] == 12
