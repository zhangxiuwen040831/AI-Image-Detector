import React from 'react';
import { motion } from 'framer-motion';
import { FileText } from 'lucide-react';

const copyMap = {
  zh: {
    title: '综合解释',
    mainPath: '三分支门控融合',
    auxiliary: '噪声残差分支说明',
    summary: '综合结论',
    mainPathText: '系统以语义结构分支、频域分布分支与噪声残差分支作为主要判定依据，综合考察内容级语义一致性、结构合理性、频谱分布特征以及残差一致性，形成对图像真实性的判断。',
    auxiliaryText: '噪声残差分支提供残差一致性与噪声统计特征方面的信息，用于增强解释性与证据完整性。',
    aigcConclusion: '当前样本在语义结构、频域特征与噪声残差证据的综合观察下，更呈现出与生成图像相一致的取证特征。',
    realConclusion: '当前样本在语义结构、频域特征与噪声残差证据的综合观察下，更接近真实图像的取证表现。',
    thresholdNote: '当前系统使用阈值判定，而不是 50% 多数判定。',
  },
  en: {
    title: 'Integrated Explanation',
    mainPath: 'Three-Branch Gated Fusion',
    auxiliary: 'Noise Residual Branch',
    summary: 'Integrated Conclusion',
    mainPathText: 'The system uses semantic structure, frequency distribution, and noise residual branches as the primary decision basis, integrating content-level consistency, structural plausibility, spectral forensic cues, and residual consistency to form an authenticity judgment.',
    auxiliaryText: 'The noise residual branch provides information on residual consistency and noise statistical characteristics to enhance interpretability and evidence completeness.',
    aigcConclusion: 'Taken together, the semantic, spectral, and noise residual cues of this sample are more consistent with the forensic characteristics of AI-generated imagery.',
    realConclusion: 'Taken together, the semantic, spectral, and noise residual cues of this sample are closer to the forensic characteristics of a real image.',
    thresholdNote: 'The current system uses threshold-based classification, not a 50% majority rule.',
  },
};

function TextColumn({ title, body }) {
  return (
    <div className="px-0 py-5 xl:px-6 xl:py-2">
      <div className="section-title mb-3">{title}</div>
      <p className="text-sm leading-8 text-muted">{body}</p>
    </div>
  );
}

const ExplanationReport = ({ result, language = 'zh' }) => {
  if (!result) return null;

  const copy = copyMap[language] || copyMap.zh;
  const isAIGC = result.prediction === 'AIGC';
  const conclusionText = `${isAIGC ? copy.aigcConclusion : copy.realConclusion} ${copy.thresholdNote}${result.decision_rule_text ? ` ${result.decision_rule_text}` : ''}`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="flex h-full flex-col rounded-[32px] border border-line bg-white p-6 shadow-card lg:p-7"
    >
      <div className="mb-5 flex flex-col gap-3 border-b border-line pb-5 md:flex-row md:items-end md:justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-panel">
            <FileText className="h-5 w-5 text-primary" />
          </div>
          <div>
            <p className="section-title">{copy.title}</p>
            <h2 className="text-2xl font-semibold tracking-tight text-ink">{copy.summary}</h2>
          </div>
        </div>
      </div>

      <div className="divide-y divide-line xl:grid xl:grid-cols-12 xl:divide-x xl:divide-y-0">
        <div className="xl:col-span-6">
          <TextColumn title={copy.mainPath} body={copy.mainPathText} />
        </div>
        <div className="xl:col-span-6">
          <TextColumn title={copy.summary} body={conclusionText} />
        </div>
      </div>
    </motion.div>
  );
};

export default ExplanationReport;
