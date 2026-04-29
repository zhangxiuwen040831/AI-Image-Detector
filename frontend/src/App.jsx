import React, { useState, useCallback, Suspense, lazy, useRef, useEffect } from 'react';
import { motion, AnimatePresence, useInView } from 'framer-motion';
import { ArrowUpRight, BookOpenText, Info, Languages, ShieldCheck, Zap, User, LogOut, Settings } from 'lucide-react';
import UploadPanel from './components/UploadPanel';
import Documentation from './components/Documentation';
import ShieldAnimation from './components/ShieldAnimation';
import LoginPage from './pages/LoginPage';
import UserProfile from './components/UserProfile';
import { getEvidenceImage, normalizeModeResult } from './utils/resultUtils';

const DetectionResult = lazy(() => import('./components/DetectionResult'));
const ProbabilityChart = lazy(() => import('./components/ProbabilityChart'));
const BranchContribution = lazy(() => import('./components/BranchContribution'));
const NoiseResidualViewer = lazy(() => import('./components/NoiseResidualViewer'));
const FrequencySpectrum = lazy(() => import('./components/FrequencySpectrum'));
const ExplanationReport = lazy(() => import('./components/ExplanationReport'));
const FusionEvidenceTriangle = lazy(() => import('./components/FusionEvidenceTriangle'));

const API_URL = (import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '')).trim();
const API_BASE = API_URL ? API_URL.replace(/\/$/, '') : '';
const DETECT_ENDPOINT = API_BASE ? `${API_BASE}/detect` : '/detect';
const SESSION_KEY = 'ai_image_detector_user';

const THRESHOLD_MODES = {
  recall: { key: 'recall', threshold: 0.20, labels: { zh: '高召回模式', en: 'High Recall Mode' } },
  standard: { key: 'standard', threshold: 0.35, labels: { zh: '标准模式', en: 'Standard Mode' } },
  precision: { key: 'precision', threshold: 0.55, labels: { zh: '高精度模式', en: 'High Precision Mode' } },
};

function RevealSection({ children, className = '', amount = 0.2 }) {
  const ref = useRef(null);
  const inView = useInView(ref, { amount, margin: '14% 0px 14% 0px' });

  return (
    <motion.section
      ref={ref}
      initial={{ opacity: 0.72, y: 12 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0.84, y: 8 }}
      transition={{ duration: 0.32, ease: 'easeOut' }}
      className={className}
    >
      {children}
    </motion.section>
  );
}

const parseAuthJson = async (response, language) => {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(language === 'zh' ? '服务器返回异常，请检查后端接口。' : 'Unexpected server response. Please check the backend API.');
  }
};

function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showDocumentation, setShowDocumentation] = useState(false);
  const [language, setLanguage] = useState('zh');
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [currentPage, setCurrentPage] = useState('main');
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [thresholdMode, setThresholdMode] = useState('standard');

  useEffect(() => {
    const savedUser = localStorage.getItem(SESSION_KEY);
    if (savedUser) {
      try {
        const parsed = JSON.parse(savedUser);
        const normalizedUser = (
          parsed &&
          (parsed.user_id || parsed.id) &&
          parsed.username &&
          parsed.user_type
        )
          ? {
              user_id: parsed.user_id ?? parsed.id,
              id: parsed.user_id ?? parsed.id,
              username: parsed.username,
              nickname: parsed.nickname || parsed.username,
              user_type: parsed.user_type,
              last_login_time: parsed.last_login_time ?? null,
              last_logout_time: parsed.last_logout_time ?? null,
            }
          : null;

        if (normalizedUser) {
          console.log('[Auth] restored user from localStorage', normalizedUser.username, normalizedUser.user_type);
          setCurrentUser(normalizedUser);
        } else {
          localStorage.removeItem(SESSION_KEY);
          localStorage.removeItem('token');
        }
      } catch (e) {
        localStorage.removeItem(SESSION_KEY);
        localStorage.removeItem('token');
      }
    } else {
      localStorage.removeItem('token');
    }
    setLoading(false);
  }, []);

  const getAuthHeaders = () => {
    const token = localStorage.getItem('token');
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    };
  };

  const handleLoginSuccess = (userData) => {
    setCurrentUser(userData);
    setCurrentPage('main');
  };

  const handlePasswordChange = async (oldPassword, newPassword) => {
    try {
      const response = await fetch('/auth/user/password', {
        method: 'PUT',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          old_password: oldPassword,
          new_password: newPassword,
        }),
      });

      const data = await parseAuthJson(response, language);

      if (!response.ok) {
        throw new Error(data.message || data.detail || (language === 'zh' ? '密码修改失败' : 'Password change failed'));
      }

      return { success: true };
    } catch (err) {
      throw err;
    }
  };

  const handleNicknameChange = async (newNickname) => {
    try {
      const response = await fetch(`/auth/user/nickname?nickname=${encodeURIComponent(newNickname)}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });

      const data = await parseAuthJson(response, language);

      if (!response.ok) {
        throw new Error(data.message || data.detail || (language === 'zh' ? '昵称修改失败' : 'Nickname change failed'));
      }

      const updatedUser = { ...currentUser, nickname: data.nickname };
      localStorage.setItem(SESSION_KEY, JSON.stringify(updatedUser));
      setCurrentUser(updatedUser);
      return { success: true };
    } catch (err) {
      throw err;
    }
  };

  const handleUserLogout = async () => {
    try {
      const response = await fetch('/auth/logout', {
        method: 'POST',
        headers: getAuthHeaders(),
      });
      const data = await parseAuthJson(response, language);
      if (!response.ok) {
        throw new Error(data.message || data.detail || 'Logout failed');
      }
    } catch (err) {
      console.error('Logout error:', err);
    } finally {
      localStorage.removeItem('token');
      localStorage.removeItem(SESSION_KEY);
      console.log('[Auth] logout');
      setCurrentUser(null);
      setCurrentPage('login');
    }
  };

  const modeResults = result?.mode_results || null;
  const mainResult = normalizeModeResult(modeResults?.tri_fusion || result, result);
  const selectedThresholdMode = THRESHOLD_MODES[thresholdMode] || THRESHOLD_MODES.standard;
  const isAuthenticated = Boolean(currentUser);
  const thresholdOptions = Object.values(THRESHOLD_MODES).map((mode) => ({
    key: mode.key,
    threshold: mode.threshold,
    label: `${mode.labels[language]} (${mode.threshold.toFixed(2)})`,
  }));
  const rawAigcProbability = typeof mainResult?.probabilities?.aigc === 'number'
    ? mainResult.probabilities.aigc
    : (typeof mainResult?.probability === 'number' ? mainResult.probability : 0);
  const displayPrediction = rawAigcProbability >= selectedThresholdMode.threshold ? 'AIGC' : 'REAL';
  const thresholdLabels = Object.fromEntries(
    Object.values(THRESHOLD_MODES).map((mode) => [
      mode.key,
      rawAigcProbability >= mode.threshold ? 'AIGC' : 'REAL',
    ]),
  );
  const thresholdUniqueLabels = [...new Set(Object.values(thresholdLabels))];
  const decisionRuleText = language === 'zh'
    ? `AIGC 概率 ${(rawAigcProbability * 100).toFixed(1)}% ${displayPrediction === 'AIGC' ? '≥' : '<'} 当前阈值 ${(selectedThresholdMode.threshold * 100).toFixed(1)}%，因此判定为 ${displayPrediction === 'AIGC' ? 'AI 生成' : '真实图像'}。`
    : `AIGC probability ${(rawAigcProbability * 100).toFixed(1)}% ${displayPrediction === 'AIGC' ? '>=' : '<'} threshold ${(selectedThresholdMode.threshold * 100).toFixed(1)}%, prediction=${displayPrediction}.`;
  const displayResult = mainResult
    ? {
        ...mainResult,
        prediction: displayPrediction,
        label: displayPrediction,
        label_id: displayPrediction === 'AIGC' ? 1 : 0,
        threshold: selectedThresholdMode.threshold,
        threshold_used: selectedThresholdMode.threshold,
        threshold_percent: selectedThresholdMode.threshold * 100,
        decision_rule_text: decisionRuleText,
        threshold_mode_key: selectedThresholdMode.key,
        threshold_mode_label: selectedThresholdMode.labels[language],
        delta_to_threshold: rawAigcProbability - selectedThresholdMode.threshold,
        threshold_labels: thresholdLabels,
        threshold_sensitive: thresholdUniqueLabels.length > 1,
      }
    : null;
  const branchVisualization = displayResult?.evidence_weights || displayResult?.fusion_weights || displayResult?.branch_scores || displayResult?.branch_triangle || displayResult?.branch_evidence || displayResult?.branch_contribution;

  const handleUpload = useCallback(async (file) => {
    if (!file) {
      setResult(null);
      setError(null);
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const uploadThreshold = THRESHOLD_MODES[thresholdMode]?.threshold ?? THRESHOLD_MODES.standard.threshold;
      const detectUrl = `${DETECT_ENDPOINT}?threshold=${encodeURIComponent(uploadThreshold)}`;
      const response = await fetch(detectUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let detailMessage = 'Analysis failed. Please try again.';
        try {
          const errorData = await response.json();
          if (errorData?.detail) {
            detailMessage = String(errorData.detail);
          }
        } catch {
          detailMessage = 'Analysis failed. Please try again.';
        }
        throw new Error(detailMessage);
      }

      const data = await response.json();
      console.log('[Detect] endpoint:', detectUrl);
      console.log('[Detect] result:', data);
      console.log('[Detect] artifacts:', data?.artifacts);
      console.log('[Detect] fusion_evidence length:', data?.artifacts?.fusion_evidence ? String(data.artifacts.fusion_evidence).length : 0);
      setResult(data);
    } catch (err) {
      setError(err.message);
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  }, [thresholdMode]);

  const copy = {
    zh: {
      eyebrow: '图像取证工作台',
      title: '面向现代图像真实性分析取证系统',
      description: '上传单张图像后，系统将返回最终判定、三分支分析概览以及噪声残差与频域证据等关键取证信息。',
      documentation: '文档与架构',
      workspace: '推理工作台',
      noteTitle: '部署说明',
      noteBody: '系统采用语义结构、频域分布、噪声残差三分支线索形成最终判定',
      resultTitle: '检测结果',
      explanationTitle: '综合解释',
      evidenceTitle: '取证证据',
      evidenceWorkspace: '证据工作区',
      evidenceWorkspaceDesc: '噪声残差与频谱证据用于检视图像中的残差结构与频域异常。',
      fusionTitle: '融合证据',
      fusionDesc: '通过三角图展示当前样本在语义结构、频域分布与噪声残差取证坐标中的融合证据分布。',
      systemStatus: '系统状态',
      uploadState: '上传工作流',
      resultState: '结果面板',
      apiTarget: '接口目标',
      running: '分析中',
      ready: '就绪',
      available: '已生成',
      waiting: '等待上传',
      branchSystem: '三分支取证',
      coreMetrics: '核心指标',
      metricSummaryTitle: '结果摘要',
      confidenceLabel: '置信度',
      probabilityLabel: 'AIGC 概率',
      decisionBasis: '主要依据',
      decisionBasisValue: '语义结构+频域分布+噪声残差',
      quickConclusion: '简要结论',
      quickAigc: '综合当前多分支取证线索，系统当前更倾向 AI 生成图像。',
      quickReal: '综合当前多分支取证线索，系统当前更倾向真实图像。',
      supportNoteTitle: '使用说明',
      supportNoteBody: '上传图像后即可展开完整取证分析工作区。',
      stageReady: '等待输入图像后开始分析',
      uploadStageTitle: '上传并开始取证分析',
      footer: 'AI Image Detector',
      footerMeta: '基于语义结构、频域分布与噪声残差的多分支取证系统',
      thresholdNote: '阈值模式只改变最终分类边界，模型概率与取证证据保持不变。',
      thresholdRule: 'AIGC 概率 ≥ 阈值时判定为 AI 生成，否则判定为真实图像。',
    },
    en: {
      eyebrow: 'AI image forensics',
      title: 'Premium forensic screening for modern image authenticity analysis.',
      description: 'Each upload returns the final decision, a three-branch analysis overview, and key forensic evidence such as noise residual and frequency cues.',
      documentation: 'Documentation',
      workspace: 'Inference workspace',
      noteTitle: 'Deployment note',
      noteBody: 'The system uses semantic, frequency, and noise three-branch cues to form the final decision.',
      resultTitle: 'Detection Results',
      explanationTitle: 'Integrated Explanation',
      evidenceTitle: 'Forensic Evidence',
      evidenceWorkspace: 'Evidence Workspace',
      evidenceWorkspaceDesc: 'Noise residual and spectrum evidence help inspect residual structure and frequency-domain anomalies.',
      fusionTitle: 'Fusion Evidence',
      fusionDesc: 'The triangle view shows the current sample across semantic, noise, and frequency forensic coordinates.',
      systemStatus: 'System Status',
      uploadState: 'Upload Workflow',
      resultState: 'Result Panels',
      apiTarget: 'API Target',
      running: 'Running',
      ready: 'Ready',
      available: 'Available',
      waiting: 'Waiting',
      branchSystem: 'Three-branch forensics',
      coreMetrics: 'Core Metrics',
      metricSummaryTitle: 'Result Summary',
      confidenceLabel: 'Confidence',
      probabilityLabel: 'AIGC Probability',
      decisionBasis: 'Main Basis',
      decisionBasisValue: 'Semantic Structure + Frequency Distribution + Noise Residual',
      quickConclusion: 'Quick Conclusion',
      quickAigc: 'Based on the current multi-branch forensic cues, the system currently leans toward an AI-generated classification.',
      quickReal: 'Based on the current multi-branch forensic cues, the system currently leans toward a real-image classification.',
      supportNoteTitle: 'Usage Note',
      supportNoteBody: 'Upload an image to expand the full forensic analysis workspace.',
      stageReady: 'Waiting for an image to begin analysis',
      uploadStageTitle: 'Upload and start forensic analysis',
      footer: 'AI Image Detector',
      footerMeta: 'Multi-branch forensic system based on semantic structure, frequency distribution, and noise residual',
      thresholdNote: 'Threshold mode only changes the decision boundary. Model probability and evidence remain unchanged.',
      thresholdRule: 'Classify as AIGC when probability is greater than or equal to the threshold; otherwise classify as REAL.',
    },
  }[language];

  const aigcProbability = rawAigcProbability;
  const confidencePercent = Math.round((displayResult?.confidence || 0) * 100);
  const quickConclusion = displayResult?.prediction === 'AIGC' ? copy.quickAigc : copy.quickReal;

  const systemStatusPanel = (
    <section className="premium-panel p-4 lg:p-5">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div className="min-w-0">
          <p className="section-title mb-2">{copy.systemStatus}</p>
          <h3 className="text-xl font-semibold tracking-tight text-ink">{copy.systemStatus}</h3>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4 xl:gap-4">
          <div className="rounded-[18px] border border-line bg-panel px-4 py-3">
            <div className="section-title mb-2">{copy.uploadState}</div>
            <div className="text-sm font-semibold tracking-tight text-ink">{isAnalyzing ? copy.running : copy.ready}</div>
          </div>
          <div className="rounded-[18px] border border-line bg-panel px-4 py-3">
            <div className="section-title mb-2">{copy.resultState}</div>
            <div className="text-sm font-semibold tracking-tight text-ink">{result ? copy.available : copy.waiting}</div>
          </div>
          <div className="rounded-[18px] border border-line bg-panel px-4 py-3">
            <div className="section-title mb-2">{language === 'zh' ? '分析框架' : 'Analysis frame'}</div>
            <div className="text-sm font-semibold tracking-tight text-ink">{copy.branchSystem}</div>
          </div>
          <div className="rounded-[18px] border border-line bg-panel px-4 py-3">
            <div className="section-title mb-2">{copy.apiTarget}</div>
            <div className="truncate text-sm font-semibold tracking-tight text-ink">{API_BASE || '/detect'}</div>
          </div>
        </div>
      </div>
    </section>
  );

  const analysisWorkspace = (
    <div className="mt-6 space-y-6 2xl:space-y-8">
      <RevealSection amount={0.05}>
        <div className="grid gap-6 xl:grid-cols-12 xl:items-stretch">
          <section className="premium-panel h-full p-5 lg:p-6 xl:col-span-3">
            <div className="border-b border-line pb-4">
              <p className="section-title mb-2">{copy.resultTitle}</p>
              <h3 className="text-2xl font-semibold tracking-tight text-ink">{copy.resultTitle}</h3>
            </div>
            <div className="mt-5 h-[calc(100%-5rem)]">
              <Suspense fallback={<div className="h-full animate-pulse rounded-[28px] bg-panel" />}>
                <DetectionResult result={displayResult} language={language} variant="embedded" />
              </Suspense>
            </div>
          </section>

          <section className="premium-panel h-full p-5 lg:p-6 xl:col-span-5">
            <div className="border-b border-line pb-4">
              <p className="section-title mb-2">{language === 'zh' ? '概率视图' : 'Probability view'}</p>
              <h3 className="text-2xl font-semibold tracking-tight text-ink">{copy.metricSummaryTitle}</h3>
            </div>
            <div className="mt-5 h-[calc(100%-5rem)]">
              <Suspense fallback={<div className="h-full animate-pulse rounded-[28px] bg-panel" />}>
                <ProbabilityChart
                  probabilities={displayResult?.probabilities}
                  probability={displayResult?.probability}
                  language={language}
                  variant="embedded"
                  thresholdModeLabel={displayResult?.threshold_mode_label}
                  threshold={displayResult?.threshold_used}
                />
              </Suspense>
            </div>
          </section>

          <section className="premium-panel flex h-full min-w-0 flex-col p-5 lg:p-6 xl:col-span-4">
            <div className="border-b border-line pb-4">
              <p className="section-title mb-2">{copy.coreMetrics}</p>
              <h3 className="text-2xl font-semibold tracking-tight text-ink">{copy.metricSummaryTitle}</h3>
            </div>
            <div className="mt-5 grid h-full gap-3 content-start sm:grid-cols-2 xl:grid-cols-1">
              <div className="rounded-[22px] border border-line bg-panel p-4">
                <div className="section-title mb-2">{copy.confidenceLabel}</div>
                <div className="text-3xl font-semibold tracking-tight text-ink">{confidencePercent}%</div>
              </div>
              <div className="rounded-[22px] border border-line bg-panel p-4">
                <div className="section-title mb-2">{copy.probabilityLabel}</div>
                <div className="text-3xl font-semibold tracking-tight text-ink">{aigcProbability.toFixed(4)}</div>
              </div>
              <div className="rounded-[22px] border border-line bg-panel p-4">
                <div className="section-title mb-2">{copy.decisionBasis}</div>
                <div className="text-lg font-semibold tracking-tight text-ink">{copy.decisionBasisValue}</div>
              </div>
              <div className="rounded-[22px] border border-line bg-panel p-4">
                <div className="section-title mb-2">{copy.quickConclusion}</div>
                <p className="text-sm leading-7 text-muted">{quickConclusion}</p>
              </div>
              <div className="rounded-[22px] border border-line bg-panel p-4 sm:col-span-2 xl:col-span-1">
                <div className="section-title mb-2">{language === 'zh' ? '阈值模式' : 'Threshold Mode'}</div>
                <div className="text-lg font-semibold tracking-tight text-ink">
                  {displayResult?.threshold_mode_label || selectedThresholdMode.labels[language]}
                </div>
                <p className="mt-2 text-sm leading-7 text-muted">
                  {language === 'zh'
                    ? `当前阈值 ${selectedThresholdMode.threshold.toFixed(2)}。${copy.thresholdRule}`
                    : `Current threshold ${selectedThresholdMode.threshold.toFixed(2)}. ${copy.thresholdRule}`}
                </p>
              </div>
            </div>
          </section>
        </div>
      </RevealSection>

      <div className="grid gap-6 xl:grid-cols-12 xl:items-stretch">
        <div className="xl:col-span-8">
          <div className="grid h-full gap-6 xl:grid-rows-[minmax(0,1.08fr)_minmax(0,0.92fr)]">
            <Suspense fallback={<div className="h-[360px] animate-pulse rounded-[32px] bg-panel" />}>
              <BranchContribution result={displayResult} language={language} />
            </Suspense>

            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, delay: 0.06 }}
              className="rounded-[32px] border border-line bg-white p-5 shadow-card lg:p-6"
            >
              <div className="border-b border-line pb-5">
                <p className="section-title mb-2">{copy.fusionTitle}</p>
                <h3 className="text-2xl font-semibold tracking-tight text-ink">{copy.fusionTitle}</h3>
                <p className="mt-3 text-sm leading-7 text-muted">{copy.fusionDesc}</p>
              </div>
              <div className="mt-5">
                <Suspense fallback={<div className="mx-auto h-[320px] w-full max-w-[760px] animate-pulse rounded-[28px] bg-panel" />}>
                  <FusionEvidenceTriangle
                    imageBase64={displayResult?.artifacts?.fusion_evidence || displayResult?.fusion_evidence_image}
                    branchContribution={branchVisualization}
                    analysisMode="evidence_weights"
                    mode={displayResult?.mode}
                    language={language}
                    embedded
                  />
                </Suspense>
              </div>
            </motion.section>
          </div>
        </div>

        <div className="xl:col-span-4">
          <div className="grid h-full grid-rows-2 gap-6">
            <Suspense fallback={<div className="h-full min-h-[220px] animate-pulse rounded-[26px] bg-panel" />}>
              <NoiseResidualViewer
                imageBase64={getEvidenceImage(displayResult, result, 'noise_residual', 'srm_image')}
                language={language}
                embedded
              />
            </Suspense>
            <Suspense fallback={<div className="h-full min-h-[220px] animate-pulse rounded-[26px] bg-panel" />}>
              <FrequencySpectrum
                imageBase64={getEvidenceImage(displayResult, result, 'frequency_spectrum', 'spectrum_image')}
                language={language}
                embedded
              />
            </Suspense>
          </div>
        </div>
      </div>

      <RevealSection amount={0.08}>
        <Suspense fallback={<div className="h-[260px] animate-pulse rounded-[32px] bg-panel" />}>
          <ExplanationReport result={displayResult} language={language} />
        </Suspense>
      </RevealSection>

      {systemStatusPanel}
    </div>
  );

  const progressiveWorkspace = (
    <div className="mx-auto w-full max-w-[1480px]">
      <section className="premium-panel overflow-hidden">
        <div className="grid gap-8 px-6 py-8 lg:px-8 lg:py-9 xl:grid-cols-[minmax(380px,0.82fr)_minmax(0,1.18fr)] xl:items-stretch">
          <div className="flex h-full flex-col justify-between">
            <div>
              <p className="section-title mb-4">{copy.eyebrow}</p>
              <motion.h2
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7 }}
                className="text-3xl font-semibold tracking-tight text-ink xl:whitespace-nowrap xl:text-[2.35rem]"
              >
                {copy.title}
              </motion.h2>
              <motion.p
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, delay: 0.08 }}
                className="mt-4 text-sm leading-7 text-muted"
              >
                {copy.description}
              </motion.p>
            </div>

            <div className="mt-6 surface-muted rounded-[24px] p-5">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white shadow-soft">
                  <ShieldCheck className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <p className="section-title">{copy.noteTitle}</p>
                  <p className="mt-1 text-sm font-medium text-ink">
                    {result ? copy.ready : copy.stageReady}
                  </p>
                </div>
              </div>
              <p className="mt-3 text-sm leading-7 text-muted">{copy.noteBody}</p>
            </div>
          </div>

          <RevealSection amount={0.05} className="h-full">
            <UploadPanel
              onUpload={handleUpload}
              isAnalyzing={isAnalyzing}
              language={language}
              hasResult={Boolean(result)}
              layout="workspace"
              thresholdMode={thresholdMode}
              thresholdOptions={thresholdOptions}
              onThresholdModeChange={setThresholdMode}
              thresholdNote={copy.thresholdNote}
            />
          </RevealSection>
        </div>
      </section>

      {!result && (
        <section className="mx-auto mt-6 w-full max-w-[1080px]">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-[24px] border border-line bg-white p-5 shadow-card">
              <p className="section-title mb-2">{copy.systemStatus}</p>
              <h3 className="text-xl font-semibold tracking-tight text-ink">{copy.stageReady}</h3>
              <p className="mt-3 text-sm leading-7 text-muted">
                {language === 'zh'
                  ? '系统待接收图像输入，当前不会展示分析结果模块。'
                  : 'The system is waiting for an image input and will not show analysis-result modules yet.'}
              </p>
            </div>
            <div className="rounded-[24px] border border-line bg-white p-5 shadow-card">
              <p className="section-title mb-2">{copy.supportNoteTitle}</p>
              <h3 className="text-xl font-semibold tracking-tight text-ink">{copy.uploadStageTitle}</h3>
              <p className="mt-3 text-sm leading-7 text-muted">{copy.supportNoteBody}</p>
            </div>
          </div>
        </section>
      )}

      {result && analysisWorkspace}
    </div>
  );

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-line bg-panel shadow-soft">
          <ShieldAnimation />
        </div>
      </div>
    );
  }

  if (!isAuthenticated || currentPage === 'login') {
    return (
      <>
        <LoginPage
          onLoginSuccess={handleLoginSuccess}
          language={language}
          onLanguageChange={setLanguage}
        />
      </>
    );
  }

  if (currentPage === 'profile') {
    return (
      <UserProfile
        user={currentUser}
        onLogout={handleUserLogout}
        onBack={() => setCurrentPage('main')}
        language={language}
        isAdmin={currentUser.user_type === 'admin'}
        onPasswordChange={handlePasswordChange}
        onNicknameChange={handleNicknameChange}
      />
    );
  }

  return (
    <div className="app-shell min-h-screen relative overflow-hidden">
      <div className="pointer-events-none absolute -left-24 top-24 h-52 w-52 rounded-full bg-slate-200/25 blur-3xl" />
      <div className="pointer-events-none absolute right-0 top-44 h-64 w-64 rounded-full bg-slate-100/70 blur-3xl" />

      <header className="sticky top-0 z-40 border-b border-line/70 bg-white/90 backdrop-blur-xl">
        <div className="mx-auto flex w-full max-w-[1840px] items-center justify-between gap-4 px-4 py-4 md:px-6 lg:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-line bg-panel shadow-soft">
              <ShieldAnimation />
            </div>
            <div className="min-w-0">
              <p className="truncate text-[11px] font-semibold uppercase tracking-[0.18em] text-subtle">
                {copy.workspace}
              </p>
              <h1 className="truncate text-base font-semibold tracking-tight text-ink md:text-lg">
                AI Image Detector
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-2 md:gap-3">
            <div className="hidden items-center rounded-full border border-line bg-panel p-1 sm:flex">
              <button
                onClick={() => setLanguage('zh')}
                className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${language === 'zh' ? 'bg-white text-ink shadow-soft' : 'text-muted hover:text-ink'}`}
              >
                中文
              </button>
              <button
                onClick={() => setLanguage('en')}
                className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${language === 'en' ? 'bg-white text-ink shadow-soft' : 'text-muted hover:text-ink'}`}
              >
                English
              </button>
            </div>

            <button
              onClick={() => setLanguage(language === 'zh' ? 'en' : 'zh')}
              className="flex h-10 w-10 items-center justify-center rounded-full border border-line bg-panel text-muted transition-colors hover:text-ink sm:hidden"
              aria-label="Switch language"
            >
              <Languages className="h-4 w-4" />
            </button>

            <button
              onClick={() => setShowDocumentation(!showDocumentation)}
              className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-all ${showDocumentation ? 'border-primary bg-primary text-white' : 'border-line bg-white text-ink hover:border-lineStrong hover:bg-panel'}`}
            >
              <BookOpenText className="h-4 w-4" />
              <span>{copy.documentation}</span>
            </button>

            <div className="relative">
              <button
                onClick={() => setCurrentPage('profile')}
                className="flex h-10 w-10 items-center justify-center rounded-full border border-line bg-panel text-muted transition-colors hover:text-ink"
                aria-label="User profile"
              >
                <User className="h-4 w-4" />
              </button>
            </div>

            <a
              href="https://github.com/zhangxiuwen040831/AI-Image-Detector"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden items-center gap-2 rounded-full border border-line bg-white px-4 py-2 text-sm font-medium text-ink transition-all hover:border-lineStrong hover:bg-panel md:inline-flex"
            >
              <span>GitHub</span>
              <ArrowUpRight className="h-4 w-4" />
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-[1840px] flex-col gap-6 px-4 pb-14 pt-6 md:px-6 lg:px-8">
        {showDocumentation ? <Documentation language={language} /> : progressiveWorkspace}
      </main>

      <footer className="border-t border-line/70 bg-white/80">
        <div className="mx-auto flex w-full max-w-[1840px] flex-col gap-3 px-4 py-6 text-sm text-muted md:flex-row md:items-center md:justify-between md:px-6 lg:px-8">
          <div className="flex items-center gap-2 text-ink">
            <span className="font-medium">{copy.footer}</span>
            <span className="text-muted">-</span>
            <span className="font-medium">{currentUser?.username || currentUser?.nickname}</span>
          </div>
          <div className="flex flex-col gap-1 md:items-end">
            <span>{copy.footerMeta}</span>
            <span className="text-xs text-subtle">© 2026 AI Forensic Research Lab</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
