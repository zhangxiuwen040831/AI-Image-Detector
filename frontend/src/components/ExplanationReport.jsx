import React from 'react';
import { motion } from 'framer-motion';
import { FileText, ShieldAlert, Microscope, Image as ImageIcon, Gauge, MessageSquare } from 'lucide-react';

const textMapZhToEn = {
  'AI图像取证报告': 'AI Image Forensic Report',
  'NTIRE模型取证报告': 'NTIRE Model Forensic Report',
  '全局语义分析': 'Global Semantic Analysis',
  '噪声伪迹分析': 'Noise Artifact Analysis',
  '频域伪迹分析': 'Frequency Artifact Analysis',
  '多模态融合决策': 'Multi-modal Fusion Decision',
  '预测概率': 'Prediction Probability',
  '分支贡献分析': 'Branch Contribution Analysis',
  '噪声残差证据': 'Noise Residual Evidence',
  '频谱证据': 'Frequency Spectrum Evidence',
  '融合证据三角图': 'Fusion Evidence Triangle',
  '置信度解释': 'Confidence Interpretation',
  '最终解读': 'Final Interpretation',
  '真实图像': 'REAL',
  'AI生成': 'AIGC',
  '可用': 'Available',
  '缺失': 'Missing',
  '低': 'Low',
  '中': 'Medium',
  '高': 'High',
};

const textMapEnToZh = {
  'AI Image Forensic Report': 'AI图像取证报告',
  'NTIRE Model Forensic Report': 'NTIRE模型取证报告',
  'Global Semantic Analysis': '全局语义分析',
  'Noise Artifact Analysis': '噪声伪迹分析',
  'Frequency Artifact Analysis': '频域伪迹分析',
  'Multi-modal Fusion Decision': '多模态融合决策',
  'Prediction Probability': '预测概率',
  'Branch Contribution Analysis': '分支贡献分析',
  'Noise Residual Evidence': '噪声残差证据',
  'Frequency Spectrum Evidence': '频谱证据',
  'Fusion Evidence Triangle': '融合证据三角图',
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

// 中文到英文和英文到中文的完整翻译函数
const translateContent = (text, language) => {
  if (!text) return text;
  
  // 常见短语的双向翻译映射
  const translations = {
    'NTIRE模型取证报告': 'NTIRE Model Forensic Report',
    '当前模型判断结果为REAL': 'Current model judgment result is REAL',
    '且真实概率显著高于AIGC概率': 'and the real probability is significantly higher than the AIGC probability',
    '且0.16的真实概率<AIGC概率0.84': 'and the real probability of 0.16 < AIGC probability of 0.84',
    '且': 'and',
    '的真实概率': 'real probability',
    'AIGC概率': 'AIGC probability',
    '置信度约为': 'Confidence is approximately',
    'Confidence is approximately0.84': 'Confidence is approximately 0.84',
    '说明模型在本次样本上更倾向于自然图像判定': 'indicating that the model is more inclined to judge natural images in this sample',
    '说明模型在本次样本上更倾向于AIGC判定': 'indicating that the model is more inclined to judge AI-generated in this sample',
    '模型更倾向该图像为真实拍摄': 'The model is more inclined to believe this image is a real shot',
    '当前风险等级为低': 'Current risk level is low',
    '但仍不应将单次模型结果视为绝对结论': 'but single model results should still not be considered absolute conclusions',
    '全局语义分支贡献约为': 'Global semantic branch contribution is approximately',
    'Global semantic branch contribution is approximately0.48': 'Global semantic branch contribution is approximately 0.48',
    '为辅助证据来源': 'as an auxiliary evidence source',
    '用于提供图像内容与高层语义一致性证据': 'to provide evidence of image content and high-level semantic consistency',
    '该分支对应 NTIRE HybridAIGCDetector 的 semantic_branch（ViT/CLIP 风格语义编码 + semantic_head），关注对象结构、语义布局与跨区域一致性线索。': 'This branch corresponds to the semantic_branch of NTIRE HybridAIGCDetector (ViT/CLIP-style semantic encoding + semantic_head), focusing on object structure, semantic layout, and cross-region consistency cues.',
    '该分支主要从图像内容一致性与高层语义结构角度提供证据': 'This branch mainly provides evidence from the perspective of global semantics and content consistency',
    '噪声分支贡献约为': 'Noise branch contribution is approximately',
    'Noise branch contribution is approximately0.26': 'Noise branch contribution is approximately 0.26',
    '是本次判定的主要依据': 'is the main basis for this judgment',
    '用于提供噪声统计与残差一致性证据': 'to provide evidence of noise statistics and residual consistency',
    '该分支对应 noise_branch + noise_head，重点捕捉合成管线可能引入的残差模式、去噪痕迹与局部噪声不一致，其证据方向与最终结果（AIGC）一致。': 'This branch corresponds to noise_branch + noise_head, focusing on capturing residual patterns, denoising traces, and local noise inconsistencies that may be introduced by the synthesis pipeline. Its evidence direction is consistent with the final result (AIGC).',
    '该分支主要从残差一致性与噪声统计角度提供证据': 'This branch mainly provides evidence from the perspective of residual consistency and noise statistics',
    '其方向与最终结果（REAL）保持一致': 'Its direction is consistent with the final result (REAL)',
    '其证据方向与最终结果（AIGC）一致': 'Its evidence direction is consistent with the final result (AIGC)',
    '频域分支贡献约为': 'Frequency branch contribution is approximately',
    'Frequency branch contribution is approximately0.26': 'Frequency branch contribution is approximately 0.26',
    '用于提供频域分布与谱结构证据': 'to provide evidence of frequency domain distribution and spectral structure',
    '该分支对应 frequency_branch + frequency_head，主要关注压缩、上采样与生成器纹理可能带来的谱异常与周期性模式。': 'This branch corresponds to frequency_branch + frequency_head, mainly focusing on spectral anomalies and periodic patterns that may be introduced by compression, upsampling, and generator textures.',
    '该分支主要从频域分布与谱结构角度提供证据': 'This branch mainly provides evidence from the perspective of frequency distribution and spectral structure',
    '分支贡献存在明显主次差异': 'There are obvious primary and secondary differences in branch contributions',
    '当前分支贡献存在主次差异': 'There are obvious primary and secondary differences in branch contributions',
    '模型主要依赖高贡献分支完成REAL判定': 'The model mainly relies on high-contribution branches to complete REAL judgment',
    '模型主要依赖高贡献分支完成AIGC判定': 'The model mainly relies on high-contribution branches to complete AIGC judgment',
    '融合层对高贡献分支赋予更高权重以完成 AIGC 判定': 'The fusion layer assigns higher weights to high-contribution branches to complete AIGC judgment',
    '其余分支提供一致性校验与辅助支持': 'Other branches provide consistency verification and auxiliary support',
    '其余分支提供辅助支持': 'Other branches provide auxiliary support',
    '该图展示REAL与AIGC的概率对比': 'This chart shows the probability comparison between REAL and AIGC',
    '当前REAL约为': 'Current REAL is approximately',
    'AIGC约为': 'AIGC is approximately',
    '用于衡量模型的类别倾向强弱': 'used to measure the strength of the model\'s category tendency',
    '该图展示语义、Noise、Frequency三分支对最终决策的相对贡献': 'This chart shows the relative contributions of the three branches (Semantic, Noise, Frequency) to the final decision',
    '用于识别本次判定的关键证据来源': 'used to identify the key evidence sources for this judgment',
    '已提供噪声残差证据图': 'Noise residual evidence map has been provided',
    '可用于辅助观察噪声一致性与残差结构': 'can be used to help observe noise consistency and residual structure',
    '已提供频谱证据图': 'Frequency spectrum evidence map has been provided',
    '可用于辅助观察频域分布与潜在异常模式': 'can be used to help observe frequency domain distribution and potential abnormal patterns',
    '已提供注意力热图': 'Attention heatmap has been provided',
    '可用于定位模型重点关注的判别区域': 'can be used to locate the discriminative regions that the model focuses on',
    '已提供融合证据三角图': 'Fusion evidence triangle map has been provided',
    '用于展示语义、频域、噪声三路证据在融合决策中的相对权重': 'to show the relative weights of semantic, frequency, and noise evidence in fusion decision-making',
    '最终预测为REAL': 'Final prediction is REAL',
    '概率分布': 'Probability distribution',
    '分支贡献用于定位关键证据来源': 'Branch contributions are used to locate key evidence sources',
    '缺失证据图已按兼容模式降级展示': 'Missing evidence maps have been displayed in compatibility mode',
    '当前置信度处于中等水平': 'Current confidence is at a medium level',
    '当前置信度处于高水平': 'Current confidence is at a high level',
    '说明结果有一定支持': 'indicating that the result has some support',
    '说明结果有一定支持但仍存在不确定性': 'indicating that the result has some support but still has uncertainty',
    '适合结合可视化证据共同解读': 'suitable for joint interpretation with visual evidence',
    '综合当前可用证据': 'Based on the currently available evidence',
    '系统将该图像判定为REAL': 'the system judges this image as REAL',
    '建议在高风险场景中结合图像来源、上下文与人工复核共同决策': 'It is recommended to combine image sources, context, and human review for joint decision-making in high-risk scenarios',
    '以避免单模型误判带来的业务风险': 'to avoid business risks caused by single model misjudgment',
    '当前模型判断结果为AIGC': 'Current model judgment result is AIGC',
    '且AIGC概率显著高于真实概率': 'and the AIGC probability is significantly higher than the real probability',
    '说明模型在本次样本上更倾向于AI生成判定': 'indicating that the model is more inclined to judge AI-generated in this sample',
    '模型更倾向该图像为AI生成': 'The model is more inclined to believe this image is AI-generated',
    '当前风险等级为高': 'Current risk level is high',
    '当前风险等级为中': 'Current risk level is medium',
    '是本次判定的重要依据': 'is an important basis for this judgment',
    '其方向与最终结果（AIGC）保持一致': 'Its direction is consistent with the final result (AIGC)',
    '模型主要依赖高贡献分支完成AIGC判定': 'The model mainly relies on high-contribution branches to complete AIGC judgment',
    '最终预测为AIGC': 'Final prediction is AIGC',
    '系统将该图像判定为AIGC': 'the system judges this image as AIGC',
    '当前无可展示摘要。': 'No summary available.',
    '当前无语义分析。': 'No semantic analysis available.',
    '当前无噪声伪迹分析。': 'No noise artifact analysis available.',
    '当前无频域伪迹分析。': 'No frequency artifact analysis available.',
    '当前无融合决策说明。': 'No fusion decision available.',
    '当前无总体评估说明。': 'No overall assessment available.',
    '当前无可展示证据说明。': 'No visual evidence available.',
    '当前无关键指标。': 'No key indicators available.',
    '当前无置信度解释。': 'No confidence interpretation available.',
    '当前无最终解读。': 'No final interpretation available.',
    '模型通过 fusion 模块对语义/频域/噪声三路证据进行门控融合，并由 classifier 输出最终判定。': 'The model performs gated fusion of semantic/frequency/noise evidence through the fusion module, and outputs the final judgment via the classifier.',
    '，': ', ',
    '。': '.',
    '（': '(',
    '）': ')',
    '、': ', ',
  };
  
  // 创建反向翻译映射（英文到中文）
  const reverseTranslations = {};
  for (const [chinese, english] of Object.entries(translations)) {
    reverseTranslations[english] = chinese;
  }
  
  let translatedText = text;
  
  if (language === 'en') {
    // 中文到英文翻译 - 按长度从长到短排序，避免部分匹配
    const sortedEntries = Object.entries(translations).sort((a, b) => b[0].length - a[0].length);
    
    for (const [chinese, english] of sortedEntries) {
      if (translatedText.includes(chinese)) {
        translatedText = translatedText.split(chinese).join(english);
      }
    }
    
    // 处理单个词的翻译
    if (textMapZhToEn[translatedText]) {
      translatedText = textMapZhToEn[translatedText];
    }
    
    // 清理中文标点符号
    translatedText = translatedText
      .replace(/，/g, ', ')
      .replace(/。/g, '.')
      .replace(/（/g, '(')
      .replace(/）/g, ')')
      .replace(/、/g, ', ');
      
    // 处理数字后面没有空格的情况（如 "approximately0.84" -> "approximately 0.84"）
    translatedText = translatedText.replace(/(approximately|is|of|than|to|by|in|on|at|for|with|from|about|around)(\d)/gi, '$1 $2');
    
  } else {
    // 英文到中文翻译 - 按长度从长到短排序，避免部分匹配
    const sortedEntries = Object.entries(reverseTranslations).sort((a, b) => b[0].length - a[0].length);
    
    for (const [english, chinese] of sortedEntries) {
      if (translatedText.includes(english)) {
        translatedText = translatedText.split(english).join(chinese);
      }
    }
    
    // 处理单个词的翻译
    if (textMapEnToZh[translatedText]) {
      translatedText = textMapEnToZh[translatedText];
    }
  }
  
  return translatedText;
};

const translateText = (value, fallbackZh, fallbackEn, language = 'zh') => {
  const text = String(value || '');
  if (language === 'zh') {
    return textMapEnToZh[text] || text || fallbackZh;
  } else {
    return textMapZhToEn[text] || text || fallbackEn;
  }
};

const toZh = (value, fallback = '', language = 'zh') => {
  return translateText(value, fallback, fallback, language);
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
          <h2 className="text-xl font-bold text-white tracking-wide">{translateText(explanation.report_title, 'AI图像取证报告', 'AI Image Forensic Report', language)}</h2>
        </div>
        <p className="text-sm text-gray-300 leading-relaxed">{translateContent(summary.summary_text || (language === 'zh' ? '当前无可展示摘要。' : 'No summary available.'), language)}</p>
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
            <div className="text-xs uppercase tracking-wider text-primary mb-2">{translateText(forensic.rgb_analysis?.title, '全局语义分析', 'Global Semantic Analysis', language)}</div>
            <p className="text-sm text-gray-300">{translateContent(forensic.rgb_analysis?.description || (language === 'zh' ? '当前无语义分析。' : 'No semantic analysis available.'), language)}</p>
          </div>
          <div className="bg-white/5 border border-white/10 rounded-lg p-4">
            <div className="text-xs uppercase tracking-wider text-primary mb-2">{translateText(forensic.noise_analysis?.title, '噪声伪迹分析', 'Noise Artifact Analysis', language)}</div>
            <p className="text-sm text-gray-300">{translateContent(forensic.noise_analysis?.description || (language === 'zh' ? '当前无噪声伪迹分析。' : 'No noise artifact analysis available.'), language)}</p>
          </div>
          <div className="bg-white/5 border border-white/10 rounded-lg p-4">
            <div className="text-xs uppercase tracking-wider text-primary mb-2">{translateText(forensic.frequency_analysis?.title, '频域伪迹分析', 'Frequency Artifact Analysis', language)}</div>
            <p className="text-sm text-gray-300">{translateContent(forensic.frequency_analysis?.description || (language === 'zh' ? '当前无频域伪迹分析。' : 'No frequency artifact analysis available.'), language)}</p>
          </div>
          <div className="bg-white/5 border border-white/10 rounded-lg p-4">
            <div className="text-xs uppercase tracking-wider text-primary mb-2">{translateText(forensic.cross_branch_conclusion?.title, '多模态融合决策', 'Multi-modal Fusion Decision', language)}</div>
            <p className="text-sm text-gray-300">{translateContent(forensic.cross_branch_conclusion?.description || (language === 'zh' ? '当前无融合决策说明。' : 'No fusion decision available.'), language)}</p>
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
                <div className="text-xs uppercase tracking-wider text-secondary">{translateText(item.title, '视觉证据', 'Visual Evidence', language)}</div>
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
        <SectionCard title={translateText(confidence.title, '置信度解释', 'Confidence Interpretation', language)} icon={MessageSquare}>
          <p className="text-sm text-gray-300 leading-relaxed">{translateContent(confidence.description || (language === 'zh' ? '当前无置信度解释。' : 'No confidence interpretation available.'), language)}</p>
        </SectionCard>
      </div>

      <SectionCard title={translateText(finalConclusion.title, '最终解读', 'Final Interpretation', language)} icon={MessageSquare}>
        <p className="text-sm text-gray-300 leading-relaxed">{translateContent(finalConclusion.description || (language === 'zh' ? '当前无最终解读。' : 'No final interpretation available.'), language)}</p>
      </SectionCard>
    </motion.div>
  );
};

export default ExplanationReport;
