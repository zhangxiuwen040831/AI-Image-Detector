import React from 'react';
import { motion } from 'framer-motion';
import {
  Book,
  Cpu,
  ShieldCheck,
  SlidersHorizontal,
  BarChart3,
  Server,
  Sparkles,
  ArrowRight,
  Upload,
  Eye,
} from 'lucide-react';

const copy = {
  zh: {
    title: 'AI Image Detector',
    intro:
      '本项目是面向 NTIRE 2026 Robust AIGC Detection 任务的最终交付版本。当前正式部署的是 V10 base_only 路线：semantic 与 frequency 负责主判别，noise 分支仅保留为辅助诊断与可解释性展示。',
    versionTitle: '当前交付版本',
    versionItems: [
      '最终模型：V10 base_only',
      '默认 checkpoint：checkpoints/best.pth',
      '默认阈值：balanced = 0.35',
      'recall-first = 0.20，precision-first = 0.35',
    ],
    architectureTitle: '模型主线',
    architectureIntro:
      '最终模型不再采用旧版“hybrid 默认主导”的部署方式，而是使用 semantic + frequency 主路径完成最终分类。',
    architectureItems: [
      'Semantic Branch：基于 CLIP ViT，负责高层语义与结构一致性判断。',
      'Frequency Branch：负责频域分布、压缩痕迹与谱结构线索。',
      'Primary Fusion：融合 semantic + frequency，得到最终 base logit。',
      'Noise Branch：保留为辅助诊断，不作为默认最终判定主证据。',
    ],
    whyTitle: '为什么默认不是 Hybrid',
    whyItems: [
      'V10 Phase 2 之后，base_only 在 hard real 上的误报控制明显优于旧 hybrid 默认路线。',
      '最终工业候选重点是降低真实图误报，同时保护脆弱 AIGC 样本。',
      '因此当前部署默认走更稳定的 base_only，而不是把 noise 重新放回主投票路径。',
    ],
    thresholdTitle: '推荐阈值',
    thresholds: [
      'recall-first：0.20，适合更注重召回的场景。',
      'balanced：0.35，当前默认部署阈值。',
      'precision-first：0.35，当前最终模型下与 balanced 一致。',
    ],
    explainTitle: '结果页如何解读',
    explainItems: [
      '预测概率：显示最终 AIGC / REAL 概率。',
      '分支证据分析：当前展示的是样本级证据强度，不再只是旧式 gate 权重。',
      '三角图：展示语义/频域/噪声证据在当前样本上的分布。',
      '噪声残差与频谱图：作为辅助诊断视图，不等于最终分类依据本身。',
    ],
    systemTitle: '系统结构',
    systemItems: [
      'React Frontend：负责上传、展示结果与说明。',
      'FastAPI Backend：负责加载最终模型并提供推理接口。',
      'ForensicDetector：封装 V10 模型加载、阈值选择和输出结构。',
    ],
    flowTitle: '检测流程',
    flowSteps: [
      ['1. 上传图像', '选择本地图片或拖拽上传。'],
      ['2. 模型推理', '走 V10 base_only 主路径完成判定。'],
      ['3. 查看分析', '展示概率、证据分布和辅助可解释性结果。'],
    ],
  },
  en: {
    title: 'AI Image Detector',
    intro:
      'This is the final delivery build for the NTIRE 2026 Robust AIGC Detection task. The deployed model uses the V10 base-only route: semantic and frequency evidence drive the primary decision, while the noise branch is retained only for auxiliary diagnostics and explainability.',
    versionTitle: 'Current Delivery',
    versionItems: [
      'Final model: V10 base_only',
      'Default checkpoint: checkpoints/best.pth',
      'Default threshold: balanced = 0.35',
      'recall-first = 0.20, precision-first = 0.35',
    ],
    architectureTitle: 'Primary Model Path',
    architectureIntro:
      'The final deployment no longer uses the old hybrid-default route. Instead, the semantic + frequency primary path is responsible for the final classification.',
    architectureItems: [
      'Semantic Branch: CLIP ViT-based high-level semantic and structural reasoning.',
      'Frequency Branch: frequency distribution, compression traces, and spectral cues.',
      'Primary Fusion: fuses semantic + frequency into the final base logit.',
      'Noise Branch: retained for auxiliary diagnostics, not the default final evidence source.',
    ],
    whyTitle: 'Why Hybrid Is Not The Default',
    whyItems: [
      'After V10 Phase 2, base_only controlled hard-real false positives better than the legacy hybrid-default route.',
      'The final industrial candidate prioritized lower false positives while protecting fragile positive AIGC samples.',
      'So the deployment default is the more stable base_only route, not a noise-heavy voting path.',
    ],
    thresholdTitle: 'Recommended Thresholds',
    thresholds: [
      'recall-first: 0.20 for recall-oriented scenarios.',
      'balanced: 0.35, the current default deployment threshold.',
      'precision-first: 0.35, currently identical to balanced for the final model.',
    ],
    explainTitle: 'How To Read The Result Page',
    explainItems: [
      'Prediction Probability: final AIGC / REAL probability output.',
      'Branch Evidence Analysis: now shows sample-level evidence intensity instead of only legacy gate weights.',
      'Triangle View: shows the distribution of semantic / frequency / noise evidence for the current sample.',
      'Noise residual and spectrum views are auxiliary diagnostics rather than the final classification rule by themselves.',
    ],
    systemTitle: 'System Stack',
    systemItems: [
      'React Frontend: upload, result rendering, and project explanation.',
      'FastAPI Backend: loads the final model and serves inference APIs.',
      'ForensicDetector: wraps V10 model loading, threshold selection, and response formatting.',
    ],
    flowTitle: 'Detection Flow',
    flowSteps: [
      ['1. Upload Image', 'Select a local image or drag and drop it.'],
      ['2. Run Inference', 'The V10 base-only primary path produces the decision.'],
      ['3. Review Analysis', 'Probability, evidence distribution, and auxiliary explainability views are displayed.'],
    ],
  },
};

const Section = ({ icon: Icon, title, children }) => (
  <section className="glass-card p-6 rounded-2xl border border-white/10">
    <div className="flex items-center gap-2 mb-4">
      <Icon className="w-5 h-5 text-primary" />
      <h2 className="text-xl font-bold text-white">{title}</h2>
    </div>
    {children}
  </section>
);

const BulletList = ({ items }) => (
  <div className="space-y-3">
    {items.map((item) => (
      <div key={item} className="flex items-start gap-3 text-gray-300">
        <span className="mt-1 text-primary">•</span>
        <p className="leading-relaxed">{item}</p>
      </div>
    ))}
  </div>
);

const Documentation = ({ language = 'zh' }) => {
  const text = copy[language] || copy.zh;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8 }}
      className="max-w-5xl mx-auto px-6 py-12 space-y-6"
    >
      <div className="glass-card p-8 rounded-2xl border border-white/10">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center">
            <Book className="text-primary w-6 h-6" />
          </div>
          <h1 className="text-3xl font-bold text-white">{text.title}</h1>
        </div>
        <p className="text-gray-300 leading-relaxed text-base">{text.intro}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Section icon={ShieldCheck} title={text.versionTitle}>
          <BulletList items={text.versionItems} />
        </Section>
        <Section icon={SlidersHorizontal} title={text.thresholdTitle}>
          <BulletList items={text.thresholds} />
        </Section>
      </div>

      <Section icon={Cpu} title={text.architectureTitle}>
        <p className="text-gray-300 leading-relaxed mb-4">{text.architectureIntro}</p>
        <BulletList items={text.architectureItems} />
      </Section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="glass p-4 rounded-xl border border-primary/20">
            <Upload className="text-primary w-6 h-6 mb-3" />
            <h3 className="text-white font-semibold mb-2">{text.flowSteps[0][0]}</h3>
            <p className="text-sm text-gray-400 leading-relaxed">{text.flowSteps[0][1]}</p>
          </div>
          <div className="glass p-4 rounded-xl border border-secondary/20">
            <Cpu className="text-secondary w-6 h-6 mb-3" />
            <h3 className="text-white font-semibold mb-2">{text.flowSteps[1][0]}</h3>
            <p className="text-sm text-gray-400 leading-relaxed">{text.flowSteps[1][1]}</p>
          </div>
          <div className="glass p-4 rounded-xl border border-accent/20">
            <Eye className="text-accent w-6 h-6 mb-3" />
            <h3 className="text-white font-semibold mb-2">{text.flowSteps[2][0]}</h3>
            <p className="text-sm text-gray-400 leading-relaxed">{text.flowSteps[2][1]}</p>
          </div>
        </div>
      </Section>
    </motion.div>
  );
};

export default Documentation;
