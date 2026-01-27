"""LLM을 사용한 RAG 답변 생성을 위한 생성 서비스."""

import os
from typing import Optional

from src.config import settings


class GenerationService:
    """로컬 LLM을 사용하여 답변을 생성하는 서비스."""

    def __init__(self, model_path: str | None = None):
        """생성 서비스를 초기화합니다.

        인수:
            model_path: GGUF 모델 파일의 경로. None인 경우 설정을 사용합니다.
        """
        self.model_path = model_path or settings.llm_model_path
        self._model = None

    def _load_model(self):
        """LLM 모델을 지연 로드합니다."""
        if self._model is not None:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for generation. "
                "Install with: pip install llama-cpp-python"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Please download the Qwen2.5-3B-Instruct model."
            )

        self._model = Llama(
            model_path=self.model_path,
            n_ctx=4096,  # Context window
            n_threads=4,  # CPU threads
            n_gpu_layers=0,  # CPU only
            verbose=False,
        )

    @property
    def model(self):
        """로드된 모델을 가져옵니다 (지연 로드)."""
        self._load_model()
        return self._model

    @property
    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return self._model is not None

    def generate_answer(
        self,
        question: str,
        context: list[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """질문과 컨텍스트를 기반으로 답변을 생성합니다.

        인수:
            question: 사용자의 질문
            context: 문서에서 추출한 관련 텍스트 청크 목록
            max_tokens: 생성된 답변의 최대 토큰 수
            temperature: 샘플링 온도 (0.0 = 결정적, 1.0 = 창의적)

        반환값:
            생성된 답변 텍스트
        """
        # RAG 프롬프트 생성
        prompt = self._build_rag_prompt(question, context)

        # 응답 생성
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False,
        )

        # 생성된 텍스트 추출
        generated_text = response["choices"][0]["text"].strip()

        return generated_text

    def _build_rag_prompt(self, question: str, context: list[str]) -> str:
        """컨텍스트와 질문으로 RAG 프롬프트를 생성합니다.

        Qwen 채팅 템플릿 형식을 사용합니다.

        인수:
            question: 사용자의 질문
            context: 관련 텍스트 청크 목록

        반환값:
            포맷된 프롬프트 문자열
        """
        # Combine context chunks
        context_text = "\n\n---\n\n".join(context) if context else "관련 컨텍스트를 찾을 수 없습니다."

        # Qwen2.5 채팅 템플릿
        prompt = f"""<|im_start|>system
당신은 제공된 컨텍스트를 기반으로 질문에 답변하는 도움이 되는 어시스턴트입니다.
컨텍스트에 있는 정보만 사용하여 답변하세요. 컨텍스트에
관련 정보가 포함되어 있지 않다면, 명확하게 말하세요. 질문과 동일한 언어로 답변하세요.
<|im_end|>
<|im_start|>user
Context:
{context_text}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""
        return prompt

    def generate_no_context_response(self, question: str) -> str:
        """관련 컨텍스트를 찾을 수 없을 때 응답을 생성합니다.

        인수:
            question: 사용자의 질문

        반환값:
            관련 정보를 찾을 수 없음을 나타내는 응답
        """
        # 질문이 한국어인지 감지
        is_korean = any('\uac00' <= char <= '\ud7a3' for char in question)

        if is_korean:
            return "죄송합니다. 업로드된 문서에서 해당 질문과 관련된 내용을 찾을 수 없습니다."
        else:
            return "죄송합니다. 업로드된 문서에서 질문에 대한 관련 정보를 찾을 수 없습니다."
