import React from 'react';
import { motion } from 'framer-motion';
import {
  ArrowRight,
  BarChart3,
  Book,
  Cpu,
  Eye,
  Server,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Upload,
} from 'lucide-react';

const copy = {
  zh: {
    title: 'AI 图像检测器',
    intro:
      '本项目面向 NTIRE 2026 Robust AIGC Detection 任务，系统围绕语义结构、频域分布与噪声残差三类取证线索构建分析流程。当前部署版本为稳定推理模式：语义结构与频域分布参与最终判定，噪声残差分支作为真实辅助证据展示；完整三分支门控判定需重新训练后启用。',
    versionTitle: '当前交付版本',
    versionItems: [
      '当前部署采用三分支特征提取 + 稳定判定路径',
      '默认 checkpoint：checkpoints/best.pth',
      '默认阈值：balanced = 0.35',
      'recall-first = 0.20，precision-first = 0.55',
    ],
    architectureTitle: '模型主线',
    architectureIntro:
      '当前部署采用三分支特征提取框架：语义结构、频域分布与噪声残差线索都会真实计算，其中语义结构与频域分布负责稳定最终判定，噪声残差用于证据展示。',
    architectureItems: [
      'Semantic Branch：基于高层内容语义与结构一致性进行分析。',
      'Frequency Branch：关注频谱分布、生成伪影与频率异常特征。',
      'Integrated Decision：当前部署使用已训练的语义结构与频域分布稳定路径形成最终判定 logit。',
      'Noise Residual Branch：提供残差一致性与噪声统计特征，作为三分支证据体系中的真实辅助证据展示。',
    ],
    whyTitle: '当前部署思路',
    whyItems: [
      '部署重点在于兼容旧 checkpoint，避免未训练的新三分支分类头直接参与最终判定。',
      '语义结构、频域分布与噪声残差线索仍然全部 forward，并返回前端用于三分支解释。',
      '完整三分支门控最终判定保留在代码结构中，待重新训练后启用。',
    ],
    thresholdTitle: '推荐阈值',
    thresholds: [
      'recall-first：0.20，适合更注重召回的场景。',
      'balanced：0.35，当前默认部署阈值。',
      'precision-first：0.55，适合更严格确认 AIGC 的场景。',
    ],
    explainTitle: '结果页如何解读',
    explainItems: [
      '预测概率：显示最终 AIGC / REAL 概率。',
      '分支证据分析：展示三类证据分别提供了什么类型的取证信息，以及它们之间的互补关系。',
      '三角图：展示语义结构、频域分布与噪声残差证据在当前样本上的相对分布，而非单一权重值。',
      '噪声残差与频谱图：用于解释图像在残差与频域空间中的表现。',
    ],
    systemTitle: '系统结构',
    systemItems: [
      'React 前端：负责上传、展示结果与说明。',
      'FastAPI 后端：负责加载当前部署模型并提供推理接口。',
      'ForensicDetector：封装模型加载、后端阈值配置和输出结构；当前前端上传流程不提供独立阈值切换控件。',
    ],
    flowTitle: '检测流程',
    flowSteps: [
      ['1. 上传图像', '选择本地图片或拖拽上传。'],
      ['2. 模型推理', '系统综合语义结构、频域分布与噪声残差证据形成最终判断。'],
      ['3. 查看分析', '展示概率结果、分支证据分布与辅助取证解释。'],
    ],
  },
  en: {
    title: 'AI Image Detector',
    intro:
      'This project is built for the NTIRE 2026 Robust AIGC Detection task. The current deployment uses safe tri-branch inference: semantic, frequency, and noise branches all run, while the trained semantic-frequency path provides the final decision until a trained tri-fusion checkpoint is available.',
    versionTitle: 'Current Delivery',
    versionItems: [
      'Current deployment follows tri-branch feature extraction with a stable decision path',
      'Default checkpoint: checkpoints/best.pth',
      'Default threshold: balanced = 0.35',
      'recall-first = 0.20, precision-first = 0.55',
    ],
    architectureTitle: 'Primary Model Path',
    architectureIntro:
      'The deployed interface follows a tri-branch feature extraction framework: all three branches are computed, while semantic and frequency evidence provide the stable final decision and noise residual evidence is shown for interpretation.',
    architectureItems: [
      'Semantic Branch: analyzes high-level semantic consistency and structural plausibility.',
      'Frequency Branch: examines spectral distribution, generative artifacts, and frequency anomalies.',
      'Integrated Decision: the current deployment uses the trained semantic-frequency stable path for the final decision logit.',
      'Noise Branch: contributes real residual-consistency and noise-statistical evidence for the three-branch evidence view.',
    ],
    whyTitle: 'Current Deployment Rationale',
    whyItems: [
      'The deployment prioritizes compatibility with the existing checkpoint and avoids using an untrained tri-fusion classifier for final prediction.',
      'Semantic, frequency, and noise residual branches are all executed and returned for frontend explanation.',
      'Full tri-branch gated final prediction remains available after retraining with tri-fusion weights.',
    ],
    thresholdTitle: 'Recommended Thresholds',
    thresholds: [
      'recall-first: 0.20 for recall-oriented scenarios.',
      'balanced: 0.35, the current default deployment threshold.',
      'precision-first: 0.55 for stricter AIGC confirmation.',
    ],
    explainTitle: 'How To Read The Result Page',
    explainItems: [
      'Prediction Probability: final AIGC / REAL probability output.',
      'Branch Evidence Analysis: shows what kind of forensic information each branch contributes and how the branches complement one another.',
      'Triangle View: visualizes the relative distribution of semantic, frequency, and noise evidence for the current sample rather than a single weight value.',
      'Noise residual and spectrum views are auxiliary forensic displays that help interpret residual and frequency-domain behavior.',
    ],
    systemTitle: 'System Stack',
    systemItems: [
      'React Frontend: upload, result rendering, and project explanation.',
      'FastAPI Backend: loads the deployed model and serves inference APIs.',
      'ForensicDetector: wraps model loading, backend threshold configuration, and response formatting; the current upload UI does not expose a separate threshold switch.',
    ],
    flowTitle: 'Detection Flow',
    flowSteps: [
      ['1. Upload Image', 'Select a local image or drag and drop it.'],
      ['2. Run Inference', 'The system forms its main judgment from semantic and frequency evidence while extracting auxiliary residual evidence.'],
      ['3. Review Analysis', 'Probability, branch evidence distribution, and auxiliary forensic explanations are displayed.'],
    ],
  },
};

const Section = ({ icon: Icon, title, children }) => (
  <section className="rounded-[24px] border border-line bg-white p-6 shadow-card">
    <div className="mb-4 flex items-center gap-3">
      <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-panel">
        <Icon className="h-5 w-5 text-primary" />
      </div>
      <h2 className="text-lg font-semibold tracking-tight text-ink">{title}</h2>
    </div>
    {children}
  </section>
);

const BulletList = ({ items }) => (
  <div className="space-y-3">
    {items.map((item) => (
      <div key={item} className="flex items-start gap-3 text-sm leading-7 text-muted">
        <span className="mt-2 h-1.5 w-1.5 rounded-full bg-slate-400" />
        <p>{item}</p>
      </div>
    ))}
  </div>
);

const Documentation = ({ language = 'zh' }) => {
  const text = copy[language] || copy.zh;

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
      className="space-y-6"
    >
      <div className="rounded-[28px] border border-line bg-white p-6 shadow-card md:p-8">
        <div className="flex flex-col gap-5 md:flex-row md:items-start md:justify-between">
          <div className="max-w-3xl">
            <p className="section-title mb-3">{language === 'zh' ? '系统概览' : 'System overview'}</p>
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-panel">
                <Book className="h-6 w-6 text-primary" />
              </div>
              <h1 className="text-3xl font-semibold tracking-tight text-ink md:text-4xl">{text.title}</h1>
            </div>
            <p className="mt-5 text-sm leading-7 text-muted md:text-base">{text.intro}</p>
          </div>
          <div className="rounded-[24px] border border-line bg-panel p-5 md:min-w-[280px]">
            <p className="section-title mb-3">{language === 'zh' ? '部署信息' : 'Deployment'}</p>
            <div className="space-y-3 text-sm text-muted">
              <div className="flex items-center justify-between gap-3">
                <span>{language === 'zh' ? '前端' : 'Frontend'}</span>
                <span className="rounded-full bg-white px-3 py-1 text-xs font-medium text-ink">React + Vite</span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span>{language === 'zh' ? '后端' : 'Backend'}</span>
                <span className="rounded-full bg-white px-3 py-1 text-xs font-medium text-ink">FastAPI</span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span>{language === 'zh' ? '模式' : 'Mode'}</span>
                <span className="rounded-full bg-white px-3 py-1 text-xs font-medium text-ink">{language === 'zh' ? '三分支门控融合' : 'Tri-branch gated fusion'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Section icon={ShieldCheck} title={text.versionTitle}>
          <BulletList items={text.versionItems} />
        </Section>
        <Section icon={SlidersHorizontal} title={text.thresholdTitle}>
          <BulletList items={text.thresholds} />
        </Section>
      </div>

      <Section icon={Cpu} title={text.architectureTitle}>
        <p className="mb-4 text-sm leading-7 text-muted">{text.architectureIntro}</p>
        <BulletList items={text.architectureItems} />
      </Section>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Section icon={Sparkles} title={text.whyTitle}>
          <BulletList items={text.whyItems} />
        </Section>
        <Section icon={BarChart3} title={text.explainTitle}>
          <BulletList items={text.explainItems} />
        </Section>
      </div>

      <Section icon={Server} title={text.systemTitle}>
        <BulletList items={text.systemItems} />
      </Section>

      <Section icon={ArrowRight} title={text.flowTitle}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="rounded-[20px] border border-line bg-panel p-5">
            <Upload className="mb-4 h-5 w-5 text-primary" />
            <h3 className="text-base font-semibold text-ink">{text.flowSteps[0][0]}</h3>
            <p className="mt-2 text-sm leading-7 text-muted">{text.flowSteps[0][1]}</p>
          </div>
          <div className="rounded-[20px] border border-line bg-panel p-5">
            <Cpu className="mb-4 h-5 w-5 text-primary" />
            <h3 className="text-base font-semibold text-ink">{text.flowSteps[1][0]}</h3>
            <p className="mt-2 text-sm leading-7 text-muted">{text.flowSteps[1][1]}</p>
          </div>
          <div className="rounded-[20px] border border-line bg-panel p-5">
            <Eye className="mb-4 h-5 w-5 text-primary" />
            <h3 className="text-base font-semibold text-ink">{text.flowSteps[2][0]}</h3>
            <p className="mt-2 text-sm leading-7 text-muted">{text.flowSteps[2][1]}</p>
          </div>
        </div>
      </Section>
    </motion.div>
  );
};

export default Documentation;
