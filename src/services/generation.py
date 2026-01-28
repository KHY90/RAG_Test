"""LLM을 사용한 RAG 답변 생성을 위한 생성 서비스."""

from typing import Optional

from src.config import settings


class GenerationService:
    """Hugging Face transformers를 사용하여 답변을 생성하는 서비스."""

    def __init__(self, model_name: str | None = None):
        """생성 서비스를 초기화합니다.

        인수:
            model_name: Hugging Face 모델 이름. None인 경우 설정을 사용합니다.
        """
        self.model_name = model_name or settings.llm_model_name
        self._pipeline = None
        self._tokenizer = None

    def _load_model(self):
        """LLM 모델을 로드합니다."""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers와 torch가 필요합니다. "
                "Install with: pip install transformers torch"
            )

        # GPU 사용 가능 여부 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading LLM model '{self.model_name}' on {device}...")

        # 토크나이저 로드
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 파이프라인 생성
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self._tokenizer,
            device=device if device == "cuda" else -1,  # -1 for CPU
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        print(f"LLM model loaded successfully on {device}.")

    @property
    def model(self):
        """로드된 파이프라인을 가져옵니다 (지연 로드)."""
        self._load_model()
        return self._pipeline

    @property
    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return self._pipeline is not None

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
        # RAG 메시지 생성
        messages = self._build_rag_messages(question, context)

        # 응답 생성
        response = self.model(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        # 생성된 텍스트 추출 (assistant 응답만)
        generated_text = response[0]["generated_text"]

        # 마지막 assistant 메시지 추출
        if isinstance(generated_text, list):
            # 메시지 형식인 경우
            for msg in reversed(generated_text):
                if msg.get("role") == "assistant":
                    return msg.get("content", "").strip()

        return str(generated_text).strip()

    def _build_rag_messages(self, question: str, context: list[str]) -> list[dict]:
        """컨텍스트와 질문으로 RAG 메시지를 생성합니다.

        인수:
            question: 사용자의 질문
            context: 관련 텍스트 청크 목록

        반환값:
            메시지 리스트
        """
        # 컨텍스트 결합
        context_text = "\n\n---\n\n".join(context) if context else "관련 컨텍스트를 찾을 수 없습니다."

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 제공된 컨텍스트를 기반으로 질문에 답변하는 도움이 되는 어시스턴트입니다. "
                    "컨텍스트에 있는 정보만 사용하여 답변하세요. "
                    "컨텍스트에 관련 정보가 포함되어 있지 않다면, 명확하게 말하세요. "
                    "질문과 동일한 언어로 답변하세요."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}",
            },
        ]
        return messages

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
