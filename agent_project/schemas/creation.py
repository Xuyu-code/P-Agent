"""Structured creation-request slots and prompt assembly."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .. import config


class GenerationParams(BaseModel):
    """User-facing controls for the external image-generation service."""

    k: int = Field(config.DEFAULT_K, ge=1, le=8)
    steps: int = Field(config.DEFAULT_STEPS, ge=1, le=150)
    guidance_scale: float = config.DEFAULT_GUIDANCE_SCALE
    conditioning_scale: float = config.DEFAULT_CONDITIONING_SCALE
    seed: int = config.DEFAULT_SEED
    postprocess: bool = config.DEFAULT_POSTPROCESS


class CreationSlots(BaseModel):
    """What the agent knows about the puppet head the user wants.

    All descriptive fields are optional: the LLM extractor fills what the
    user stated, and `missing_fields()` decides whether to ask a follow-up.
    `lineart_path` is the only hard requirement (the generation service is
    condition-driven — without a lineart there is nothing to generate from).
    """

    role: str | None = None          # 角色，如 女角 / 老生
    facing: str | None = None        # 朝向，如 左侧面 / 右侧面
    headwear: str | None = None      # 冠饰，如 凤冠
    color_palette: str | None = None # 色彩，如 暖黄色
    pattern: str | None = None       # 纹样，如 镂空云纹
    extra_notes: str | None = None   # 其他自由描述
    negative_prompt: str | None = None
    lineart_path: str | None = None  # local path under agent data/uploads/
    generated_once: bool = False     # 已用当前槽位成功生成过——用户已默认接受现有信息，不再追问缺省字段
    params: GenerationParams = Field(default_factory=GenerationParams)

    def missing_fields(self) -> list[str]:
        """Descriptive fields still empty; role matters most for a usable prompt."""
        missing = []
        if not self.role:
            missing.append("role")
        if not self.headwear:
            missing.append("headwear")
        if not self.color_palette:
            missing.append("color_palette")
        return missing

    def merge(self, other: "CreationSlots") -> "CreationSlots":
        """Later user turns refine earlier slots; only non-None fields override."""
        data = self.model_dump()
        for key, value in other.model_dump().items():
            if key in ("params", "generated_once"):
                continue
            if value is not None:
                data[key] = value
        data["params"] = self.params
        return CreationSlots(**data)


# extra_notes 里这类词只是复述风格前缀，拼进 prompt 会产生“…，皮影风格”的冗余
_REDUNDANT_NOTES = {"皮影", "皮影风格", "河湟皮影", "河湟皮影风格", "河湟皮影风格头像"}


def build_prompt(slots: CreationSlots) -> str:
    """Assemble the generation prompt from slots (Chinese, comma-separated)."""
    extra = slots.extra_notes
    if extra and extra.strip().strip("，。,. ") in _REDUNDANT_NOTES:
        extra = None
    parts = [
        "河湟皮影风格头像",
        slots.facing,
        slots.role,
        slots.headwear and f"佩戴{slots.headwear}",
        slots.color_palette and f"{slots.color_palette}色调",
        slots.pattern and f"{slots.pattern}纹样",
        extra,
    ]
    return "，".join(p for p in parts if p)
