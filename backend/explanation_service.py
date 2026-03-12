from typing import Any, Dict, List, Optional


def _clamp(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _normalize_prediction(value: Any) -> str:
    label = str(value or "").upper()
    if label == "AIGC":
        return "AIGC"
    return "REAL"


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _build_summary_text(prediction: str, confidence: float, real_prob: float, aigc_prob: float) -> str:
    gap = abs(real_prob - aigc_prob)
    confidence_percent = int(round(confidence * 100))
    if gap < 0.1:
        return f"当前模型判断结果为{prediction}，但真实与AIGC概率接近，整体存在明显不确定性。置信度约为{confidence_percent}%，建议结合图像来源与人工复核共同判断。"
    if prediction == "AIGC":
        return f"当前模型判断结果为AIGC，且AIGC概率显著高于真实概率。置信度约为{confidence_percent}%，说明多分支信号对“AI生成”方向提供了较一致支持。"
    return f"当前模型判断结果为REAL，且真实概率显著高于AIGC概率。置信度约为{confidence_percent}%，说明模型在本次样本上更倾向于自然图像判定。"


def _build_overall_assessment(prediction: str, real_prob: float, aigc_prob: float) -> Dict[str, Any]:
    authenticity_score = int(round(real_prob * 100))
    gap = abs(real_prob - aigc_prob)
    if gap < 0.1:
        risk_level = "Medium"
        assessment_text = "真实与AIGC概率接近，当前证据呈边界状态，建议谨慎解释并结合更多外部信息复核。"
    elif prediction == "AIGC":
        risk_level = "High"
        assessment_text = "模型更倾向该图像为AI生成，当前风险等级为高，建议在关键业务场景中进行二次核验。"
    else:
        risk_level = "Low"
        assessment_text = "模型更倾向该图像为真实拍摄，当前风险等级为低，但仍不应将单次模型结果视为绝对结论。"
    return {
        "risk_level": risk_level,
        "authenticity_score": authenticity_score,
        "assessment_text": assessment_text,
    }


def _branch_description(
    branch_name_cn: str,
    enabled: bool,
    contribution: Any,
    is_top: bool,
    prediction: str,
    branch_type: str,
) -> str:
    if not enabled:
        return f"{branch_name_cn}分支在本次推理中未启用，因此不参与本次结论。"
    if contribution is None:
        return f"{branch_name_cn}分支已启用，但当前未返回贡献度数值，系统采用兼容模式展示，暂无法量化其影响强弱。"
    score = _clamp(contribution)
    score_percent = int(round(score * 100))
    focus_map = {
        "rgb": "该分支主要从语义结构与纹理一致性角度提供证据。",
        "noise": "该分支主要从残差与噪声一致性角度提供证据。",
        "frequency": "该分支主要从频域分布与谱结构角度提供证据。",
    }
    if is_top:
        return f"{branch_name_cn}分支贡献约为{score_percent}%，是本次判定的主要依据。{focus_map[branch_type]}其方向与最终结果（{prediction}）保持一致。"
    return f"{branch_name_cn}分支贡献约为{score_percent}%，为辅助证据来源。{focus_map[branch_type]}"


def _cross_branch_description(
    enabled_branches: List[str],
    branch_contribution: Dict[str, Any],
    prediction: str,
) -> str:
    enabled_set = {name.lower() for name in enabled_branches}
    active_values: List[float] = []
    for key in ["rgb", "noise", "frequency"]:
        if key in enabled_set and branch_contribution.get(key) is not None:
            parsed_value = _clamp(branch_contribution.get(key), default=None)
            if parsed_value is not None:
                active_values.append(parsed_value)
    if not active_values:
        return "当前缺少可用的分支贡献度数据，无法判断跨分支是否一致支持最终结论。"
    spread = max(active_values) - min(active_values)
    if spread < 0.12:
        return f"各启用分支贡献度较为接近，说明模型采用多分支联合证据完成判定，整体对{prediction}结论呈协同支持。"
    return f"分支贡献存在明显主次差异，模型主要依赖高贡献分支完成{prediction}判定，其余分支提供辅助支持。"


def generate_explanation_report(inference: Dict[str, Any]) -> Dict[str, Any]:
    prediction = _normalize_prediction(inference.get("prediction"))
    confidence = _clamp(inference.get("confidence"))

    probabilities = inference.get("probabilities") or {}
    real_prob = _clamp(probabilities.get("real"), default=None)
    aigc_prob = _clamp(probabilities.get("aigc"), default=None)

    if real_prob is None or aigc_prob is None:
        legacy_probability = _clamp(inference.get("probability"))
        if prediction == "AIGC":
            aigc_prob = legacy_probability
            real_prob = 1 - legacy_probability
        else:
            real_prob = legacy_probability
            aigc_prob = 1 - legacy_probability

    branch_contribution = inference.get("branch_contribution") or {}
    enabled_branches = list((inference.get("metadata") or {}).get("enabled_branches") or ["rgb", "noise", "frequency"])
    enabled_set = {name.lower() for name in enabled_branches}

    rgb_value = branch_contribution.get("rgb")
    noise_value = branch_contribution.get("noise")
    freq_value = branch_contribution.get("frequency")

    scored_candidates = []
    for name, value in [("rgb", rgb_value), ("noise", noise_value), ("frequency", freq_value)]:
        if value is not None:
            scored_candidates.append((name, _clamp(value)))
    top_branch = max(scored_candidates, key=lambda x: x[1])[0] if scored_candidates else None

    artifacts = inference.get("artifacts") or {}
    has_noise = bool(artifacts.get("noise_residual"))
    has_spectrum = bool(artifacts.get("frequency_spectrum"))
    has_grad_cam = bool(artifacts.get("grad_cam"))

    confidence_level = _confidence_label(confidence)
    if confidence_level == "high":
        confidence_text = "当前置信度较高，表示模型内部信号整体较一致，但这并不等于绝对正确，仍建议结合来源信息综合判断。"
    elif confidence_level == "medium":
        confidence_text = "当前置信度处于中等水平，说明结果有一定支持但仍存在不确定性，适合结合可视化证据共同解读。"
    else:
        confidence_text = "当前置信度较低，说明模型信号存在混合或边界特征，建议将本次结果视为提示信息而非最终定论。"

    summary_text = _build_summary_text(prediction, confidence, real_prob, aigc_prob)
    overall_assessment = _build_overall_assessment(prediction, real_prob, aigc_prob)

    return {
        "report_title": "AI Image Forensic Report",
        "summary": {
            "final_result": prediction,
            "confidence_score": confidence,
            "summary_text": summary_text,
        },
        "overall_assessment": overall_assessment,
        "forensic_analysis": {
            "rgb_analysis": {
                "title": "RGB Semantic Analysis",
                "description": _branch_description("RGB", "rgb" in enabled_set, rgb_value, top_branch == "rgb", prediction, "rgb"),
            },
            "noise_analysis": {
                "title": "Noise Residual Analysis",
                "description": _branch_description("噪声", "noise" in enabled_set, noise_value, top_branch == "noise", prediction, "noise"),
            },
            "frequency_analysis": {
                "title": "Frequency Domain Analysis",
                "description": _branch_description("频域", "frequency" in enabled_set, freq_value, top_branch == "frequency", prediction, "frequency"),
            },
            "cross_branch_conclusion": {
                "title": "Cross-Branch Conclusion",
                "description": _cross_branch_description(enabled_branches, branch_contribution, prediction),
            },
        },
        "visual_evidence": [
            {
                "title": "Prediction Probability",
                "description": f"该图展示REAL与AIGC的概率对比。当前REAL约为{real_prob:.3f}，AIGC约为{aigc_prob:.3f}，用于衡量模型的类别倾向强弱。",
                "available": True,
            },
            {
                "title": "Branch Contribution Analysis",
                "description": "该图展示RGB、Noise、Frequency三分支对最终决策的相对贡献，用于识别本次判定的关键证据来源。",
                "available": True,
            },
            {
                "title": "Noise Residual Evidence",
                "description": "已提供噪声残差证据图，可用于辅助观察噪声一致性与残差结构。" if has_noise else "当前未提供该证据图，噪声残差维度仅能基于数值结果解释。",
                "available": has_noise,
            },
            {
                "title": "Frequency Spectrum Evidence",
                "description": "已提供频谱证据图，可用于辅助观察频域分布与潜在异常模式。" if has_spectrum else "当前未提供该证据图，频域维度仅能基于数值结果解释。",
                "available": has_spectrum,
            },
            {
                "title": "Attention Heatmap",
                "description": "已提供注意力热图，可用于定位模型重点关注的判别区域。" if has_grad_cam else "当前未提供该证据图，暂无法展示模型空间注意力区域。",
                "available": has_grad_cam,
            },
        ],
        "key_indicators": [
            f"最终预测为{prediction}，置信度约为{confidence:.3f}",
            f"概率分布：REAL={real_prob:.3f}，AIGC={aigc_prob:.3f}",
            "分支贡献用于定位关键证据来源，缺失证据图已按兼容模式降级展示",
        ],
        "confidence_explanation": {
            "title": "Confidence Interpretation",
            "description": confidence_text,
        },
        "user_friendly_conclusion": {
            "title": "Final Interpretation",
            "description": f"综合当前可用证据，系统将该图像判定为{prediction}。建议在高风险场景中结合图像来源、上下文与人工复核共同决策，以避免单模型误判带来的业务风险。",
        },
    }
