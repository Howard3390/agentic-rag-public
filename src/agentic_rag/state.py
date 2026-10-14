from typing import Literal, NotRequired, TypedDict


Stage = Literal["dense", "linear", "light"]


class Evidence(TypedDict):
    evidence_id: str
    content: str
    source_id: str
    source_ref: str
    retrieval_stage: Stage
    reference_id: NotRequired[str]


class AgentState(TypedDict):
    query: str
    current_stage: NotRequired[Stage]
    current_evidence: NotRequired[list[Evidence]]
    stage_evidence: NotRequired[dict[Stage, list[Evidence]]]
    evidence_pool: NotRequired[list[Evidence]]
    stage_history: NotRequired[list[dict]]
    retrieval_error: NotRequired[str | None]
    sufficient: NotRequired[bool]
    judge_reason: NotRequired[str]
    missing_info: NotRequired[list[str]]
    supporting_ids: NotRequired[list[int]]
    judge_error: NotRequired[str | None]
    answer: NotRequired[str | None]
    answer_evidence: NotRequired[list[Evidence]]
    answer_source_ids: NotRequired[list[str]]
    answer_error: NotRequired[str | None]
