import React from 'react';
import { motion } from 'framer-motion';
import { FileText, ShieldAlert, Microscope, Image as ImageIcon, Gauge, MessageSquare } from 'lucide-react';

const textMap = {
  'AI Image Forensic Report': 'AI图像取证报告',
  'RGB Semantic Analysis': 'RGB语义分析',
  'Noise Residual Analysis': '噪声残差分析',
  'Frequency Domain Analysis': '频域分析',
  'Cross-Branch Conclusion': '跨分支结论',
  'Prediction Probability': '预测概率',
  'Branch Contribution Analysis': '分支贡献分析',
  'Noise Residual Evidence': '噪声残差证据',
  'Frequency Spectrum Evidence': '频谱证据',
  'Attention Heatmap': '注意力热力图',
  'Confidence Interpretation': '置信度解释',
  'Final Interpretation': '最终解读',
  'REAL': '真实图像',
  'AIGC': 'AI生成',
  'Available': '可用',
  'Missing': '缺失',
  'Low': '低',
  'Medium': '中',
  'High': '高',
};

// 简单的中文到英文翻译函数
const translateContent = (text, language) => {
  if (language !== 'en' || !text) return text;
  
  // 常见中文短语的翻译
  const translations = {
    '当前模型判断结果为REAL': 'Current model judgment result is REAL',
    '且真实概率显著高于AIGC概率': 'and the real probability is significantly higher than the AIGC probability',
    '置信度约为': 'Confidence is approximately',
    '说明模型在本次样本上更倾向于自然图像判定': 'indicating that the model is more inclined to judge natural images in this sample',
    '模型更倾向该图像为真实拍摄': 'The model is more inclined to believe this image is a real shot',
    '当前风险等级为低': 'Current risk level is low',
    '但仍不应将单次模型结果视为绝对结论': 'but single model results should still not be considered absolute conclusions',
    'RGB分支贡献约为': 'RGB branch contribution is approximately',
    '为辅助证据来源': 'as an auxiliary evidence source',
    '该分支主要从语义结构与纹理一致性角度提供证据': 'This branch mainly provides evidence from the perspective of semantic structure and texture consistency',
    '噪声分支贡献约为': 'Noise branch contribution is approximately',
    '是本次判定的主要依据': 'is the main basis for this judgment',
    '该分支主要从残差与噪声一致性角度提供证据': 'This branch mainly provides evidence from the perspective of residual and noise consistency',
    '其方向与最终结果（REAL）保持一致': 'Its direction is consistent with the final result (REAL)',
    '频域分支贡献约为': 'Frequency branch contribution is approximately',
    '该分支主要从频域分布与谱结构角度提供证据': 'This branch mainly provides evidence from the perspective of frequency domain distribution and spectral structure',
    '分支贡献存在明显主次差异': 'There are obvious primary and secondary differences in branch contributions',
    '模型主要依赖高贡献分支完成REAL判定': 'The model mainly relies on high-contribution branches to complete REAL judgment',
    '其余分支提供辅助支持': 'Other branches provide auxiliary support',
    '该图展示REAL与AIGC的概率对比': 'This chart shows the probability comparison between REAL and AIGC',
    '当前REAL约为': 'Current REAL is approximately',
    'AIGC约为': 'AIGC is approximately',
    '用于衡量模型的类别倾向强弱': 'used to measure the strength of the model\'s category tendency',
    '该图展示RGB、Noise、Frequency三分支对最终决策的相对贡献': 'This chart shows the relative contributions of the three branches (RGB, Noise, Frequency) to the final decision',
    '用于识别本次判定的关键证据来源': 'used to identify the key evidence sources for this judgment',
    '已提供噪声残差证据图': 'Noise residual evidence map has been provided',
    '可用于辅助观察噪声一致性与残差结构': 'can be used to help observe noise consistency and residual structure',
    '已提供频谱证据图': 'Frequency spectrum evidence map has been provided',
    '可用于辅助观察频域分布与潜在异常模式': 'can be used to help observe frequency domain distribution and potential abnormal patterns',
    '已提供注意力热图': 'Attention heatmap has been provided',
    '可用于定位模型重点关注的判别区域': 'can be used to locate the discriminative regions that the model focuses on',
    '最终预测为REAL': 'Final prediction is REAL',
    '概率分布': 'Probability distribution',
    '分支贡献用于定位关键证据来源': 'Branch contributions are used to locate key evidence sources',
    '缺失证据图已按兼容模式降级展示': 'Missing evidence maps have been displayed in compatibility mode',
    '当前置信度处于中等水平': 'Current confidence is at a medium level',
    '说明结果有一定支持但仍存在不确定性': 'indicating that the result has some support but still has uncertainty',
    '适合结合可视化证据共同解读': 'suitable for joint interpretation with visual evidence',
    '综合当前可用证据': 'Based on the currently available evidence',
    '系统将该图像判定为REAL': 'the system judges this image as REAL',
    '建议在高风险场景中结合图像来源、上下文与人工复核共同决策': 'It is recommended to combine image sources, context, and human review for joint decision-making in high-risk scenarios',
    '以避免单模型误判带来的业务风险': 'to avoid business risks caused by single model misjudgment',
  };
  
  let translatedText = text;
  for (const [chinese, english] of Object.entries(translations)) {
    if (translatedText.includes(chinese)) {
      translatedText = translatedText.replace(chinese, english);
    }
  }
  
  return translatedText;
};

const toZh = (value, fallback = '', language = 'zh') => {
  if (language !== 'zh') {
    return String(value || fallback);
  }
  const text = String(value || fallback);
  return textMap[text] || text;
};

const SectionCard = ({ title, children, icon: Icon }) => (
  <div className="glass-card p-6">
    <div className="flex items-center gap-2 mb-4">
      <Icon className="w-4 h-4 text-primary" />
      <h3 className="text-sm font-bold tracking-widest uppercase text-white">{title}</h3>
    </div>
    {children}
  </div>
);

const ExplanationReport = ({ explanation, language = 'zh' }) => {
  if (!explanation) return null;

  const summary = explanation.summary || {};
  const assessment = explanation.overall_assessment || {};
  const forensic = explanation.forensic_analysis || {};
  const evidence = Array.isArray(explanation.visual_evidence) ? explanation.visual_evidence : [];
  const indicators = Array.isArray(explanation.key_indicators) ? explanation.key_indicators : [];
  const confidence = explanation.confidence_explanation || {};
  const finalConclusion = explanation.user_friendly_conclusion || {};

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.35 }}
      className="space-y-5"
    >
      <div className="glass-card p-5 border border-white/10">
        <div className="flex items-center gap-2 mb-3">
          <FileText className="w-5 h-5 text-secondary" />
          <h2 className="text-xl font-bold text-white tracking-wide">{toZh(explanation.report_title, language === 'zh' ? 'AI图像取证报告' : 'AI Image Forensic Report', language)}</h2>
        </div>
        <p className="text-sm text-gray-300 leading-relaxed">{translateContent(summary.summary_text || '当前无可展示摘要。', language)}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <SectionCard title={language === 'zh' ? '摘要' : 'Summary'} icon={FileText}>
          <p className="text-sm text-gray-300 leading-relaxed mb-3">{translateContent(summary.summary_text || (language === 'zh' ? '当前无可展示摘要。' : 'No summary available.'), language)}</p>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="bg-white/5 border border-white/10 rounded-lg p-3">
              <div className="text-gray-400 uppercase tracking-wider">{language === 'zh' ? '最终结果' : 'Final Result'}</div>
              <div className="text-white font-semibold mt-1">{translateContent(summary.final_result || '-', language)}</div>
            </div>
            <div className="bg-white/5 border border-white/10 rounded-lg p-3">
              <div className="text-gray-400 uppercase tracking-wider">{language === 'zh' ? '置信度' : 'Confidence'}</div>
              <div className="text-white font-semibold mt-1">{typeof summary.confidence_score === 'number' ? `${Math.round(summary.confidence_score * 100)}%` : '-'}</div>
            </div>
          </div>
        </SectionCard>

        <SectionCard title={language === 'zh' ? '总体评估' : 'Overall Assessment'} icon={ShieldAlert}>
          <p className="text-sm text-gray-300 leading-relaxed mb-3">{translateContent(assessment.assessment_text || (language === 'zh' ? '当前无总体评估说明。' : 'No overall assessment available.'), language)}</p>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="bg-white/5 border border-white/10 rounded-lg p-3">
              <div className="text-gray-400 uppercase tracking-wider">{language === 'zh' ? '风险等级' : 'Risk Level'}</div>
              <div className="text-white font-semibold mt-1">{translateContent(assessment.risk_level || '-', language)}</div>
            </div>
            <div className="bg-white/5 border border-white/10 rounded-lg p-3">
              <div className="text-gray-400 uppercase tracking-wider">{language === 'zh' ? '真实性得分' : 'Authenticity Score'}</div>
              <div className="text-white font-semibold mt-1">{typeof assessment.authenticity_score === 'number' ? assessment.authenticity_score : '-'}</div>
            </div>
          </div>
        </SectionCard>
      </div>

      <SectionCard title={language === 'zh' ? '取证分析' : 'Forensic Analysis'} icon={Microscope}>
        <div className="space-y-4">
          <div className="bg-white/5 border border-white/10 rounded-lg p-4">
            <div className="text-xs uppercase tracking-wider text-primary mb-2">{toZh(forensic.rgb_analysis?.title, language === 'zh' ? 'RGB语义分析' : 'RGB Semantic Analysis', language)}</div>
            <p className="text-sm text-gray-300">{translateContent(forensic.rgb_analysis?.description || (language === 'zh' ? '当前无RGB分析。' : 'No RGB analysis available.'), language)}</p>
          </div>
          <div className="bg-white/5 border border-white/10 rounded-lg p-4">
            <div className="text-xs uppercase tracking-wider text-primary mb-2">{toZh(forensic.noise_analysis?.title, language === 'zh' ? '噪声残差分析' : 'Noise Residual Analysis', language)}</div>
            <p className="text-sm text-gray-300">{translateContent(forensic.noise_analysis?.description || (language === 'zh' ? '当前无噪声分析。' : 'No noise analysis available.'), language)}</p>
          </div>
          <div className="bg-white/5 border border-white/10 rounded-lg p-4">
            <div className="text-xs uppercase tracking-wider text-primary mb-2">{toZh(forensic.frequency_analysis?.title, language === 'zh' ? '频域分析' : 'Frequency Domain Analysis', language)}</div>
            <p className="text-sm text-gray-300">{translateContent(forensic.frequency_analysis?.description || (language === 'zh' ? '当前无频域分析。' : 'No frequency analysis available.'), language)}</p>
          </div>
          <div className="bg-white/5 border border-white/10 rounded-lg p-4">
            <div className="text-xs uppercase tracking-wider text-primary mb-2">{toZh(forensic.cross_branch_conclusion?.title, language === 'zh' ? '跨分支结论' : 'Cross-Branch Conclusion', language)}</div>
            <p className="text-sm text-gray-300">{translateContent(forensic.cross_branch_conclusion?.description || (language === 'zh' ? '当前无跨分支结论。' : 'No cross-branch conclusion available.'), language)}</p>
          </div>
        </div>
      </SectionCard>

      <SectionCard title={language === 'zh' ? '视觉证据' : 'Visual Evidence'} icon={ImageIcon}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {evidence.length === 0 && (
            <div className="text-sm text-gray-400">{language === 'zh' ? '当前无可展示证据说明。' : 'No visual evidence available.'}</div>
          )}
          {evidence.map((item, index) => (
            <div key={`${item.title}-${index}`} className="bg-white/5 border border-white/10 rounded-lg p-4">
              <div className="flex items-center justify-between gap-3 mb-2">
                <div className="text-xs uppercase tracking-wider text-secondary">{toZh(item.title, language === 'zh' ? '视觉证据' : 'Visual Evidence', language)}</div>
                <span className={`text-[10px] px-2 py-1 rounded-full border ${item.available ? 'text-green-300 border-green-400/30 bg-green-500/10' : 'text-yellow-300 border-yellow-400/30 bg-yellow-500/10'}`}>
                  {item.available ? (language === 'zh' ? '可用' : 'Available') : (language === 'zh' ? '缺失' : 'Missing')}
                </span>
              </div>
              <p className="text-sm text-gray-300">{translateContent(item.description, language)}</p>
            </div>
          ))}
        </div>
      </SectionCard>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <SectionCard title={language === 'zh' ? '关键指标' : 'Key Indicators'} icon={Gauge}>
          <div className="space-y-2">
            {indicators.length === 0 && <p className="text-sm text-gray-400">{language === 'zh' ? '当前无关键指标。' : 'No key indicators available.'}</p>}
            {indicators.map((item, index) => (
              <p key={`${item}-${index}`} className="text-sm text-gray-300">• {translateContent(item, language)}</p>
            ))}
          </div>
        </SectionCard>
        <SectionCard title={toZh(confidence.title, language === 'zh' ? '置信度解释' : 'Confidence Interpretation', language)} icon={MessageSquare}>
          <p className="text-sm text-gray-300 leading-relaxed">{translateContent(confidence.description || (language === 'zh' ? '当前无置信度解释。' : 'No confidence interpretation available.'), language)}</p>
        </SectionCard>
      </div>

      <SectionCard title={toZh(finalConclusion.title, language === 'zh' ? '最终解读' : 'Final Interpretation', language)} icon={MessageSquare}>
        <p className="text-sm text-gray-300 leading-relaxed">{translateContent(finalConclusion.description || (language === 'zh' ? '当前无最终解读。' : 'No final interpretation available.'), language)}</p>
      </SectionCard>
    </motion.div>
  );
};

export default ExplanationReport;
