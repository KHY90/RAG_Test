"""LLM을 사용한 RAG 답변 생성을 위한 생성 서비스."""

from pathlib import Path

from src.config import settings


class GenerationService:
    """llama-cpp-python을 사용하여 GGUF 모델로 답변을 생성하는 서비스."""

    def __init__(self, model_path: str | None = None):
        """생성 서비스를 초기화합니다.

        인수:
            model_path: GGUF 모델 파일 경로. None인 경우 설정을 사용합니다.
        """
        self.model_path = model_path or settings.llm_model_path
        self._llm = None

    def _load_model(self):
        """GGUF 모델을 로드합니다. GPU 우선, 실패 시 CPU 폴백."""
        if self._llm is not None:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python이 필요합니다. "
                "Install with: pip install llama-cpp-python"
            )

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"GGUF 모델 파일을 찾을 수 없습니다: {self.model_path}\n"
                f"모델을 다운로드하여 해당 경로에 저장해주세요."
            )

        # 1차: GPU 시도 (n_gpu_layers=-1)
        try:
            print("GPU 모드로 모델 로드 시도 중...")
            self._llm = Llama(
                model_path=str(model_file),
                n_ctx=settings.llm_context_length,
                n_gpu_layers=-1,  # 모든 레이어 GPU
                verbose=False,
            )
            print("✓ GPU 모드로 모델 로드 완료")
            return
        except Exception as e:
            print(f"GPU 로드 실패: {e}")

        # 2차: CPU 폴백 (n_gpu_layers=0)
        print("CPU 모드로 모델 로드 중...")
        self._llm = Llama(
            model_path=str(model_file),
            n_ctx=settings.llm_context_length,
            n_gpu_layers=0,  # CPU only
            verbose=False,
        )
        print("✓ CPU 모드로 모델 로드 완료")

    @property
    def model(self):
        """로드된 모델을 가져옵니다 (지연 로드)."""
        self._load_model()
        return self._llm

    @property
    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return self._llm is not None

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
        )

        # 생성된 텍스트 추출
        generated_text = response["choices"][0]["text"].strip()
        return generated_text

    def _build_rag_prompt(self, question: str, context: list[str]) -> str:
        """컨텍스트와 질문으로 RAG 프롬프트를 생성합니다.

        인수:
            question: 사용자의 질문
            context: 관련 텍스트 청크 목록

        반환값:
            프롬프트 문자열
        """
        context_text = "\n\n---\n\n".join(context) if context else "관련 컨텍스트를 찾을 수 없습니다."

        system_prompt = (
            "당신은 제공된 컨텍스트를 기반으로 질문에 답변하는 도움이 되는 어시스턴트입니다. "
            "컨텍스트에 있는 정보만 사용하여 답변하세요. "
            "컨텍스트에 관련 정보가 포함되어 있지 않다면, 명확하게 말하세요. "
            "질문과 동일한 언어로 답변하세요."
        )

        # Qwen2.5 ChatML 형식
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nContext:\n{context_text}\n\nQuestion: {question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

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
