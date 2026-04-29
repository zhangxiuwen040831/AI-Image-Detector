import React, { useMemo } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { motion } from 'framer-motion';

const ProbabilityChart = ({
  probabilities,
  probability,
  language = 'zh',
  variant = 'card',
  thresholdModeLabel,
  threshold,
}) => {
  const isEmbedded = variant === 'embedded';
  const aigcProbability = typeof probabilities?.aigc === 'number' ? probabilities.aigc : (typeof probability === 'number' ? probability : 0);
  const realProbability = typeof probabilities?.real === 'number' ? probabilities.real : (1 - aigcProbability);
  const thresholdValue = typeof threshold === 'number' ? threshold : null;
  const thresholdPercent = thresholdValue !== null ? Math.max(0, Math.min(100, thresholdValue * 100)) : 0;
  const probabilityPercent = Math.max(0, Math.min(100, aigcProbability * 100));
  const data = useMemo(() => [
    { name: language === 'zh' ? 'AI生成' : 'AI Generated', value: Math.max(0, aigcProbability * 100) },
    { name: language === 'zh' ? '真实图像' : 'Real Image', value: Math.max(0, realProbability * 100) },
  ], [aigcProbability, realProbability, language]);

  const COLORS = ['#334155', '#cbd5e1'];

  const content = (
    <>
      <div className="mb-4 flex items-end justify-between gap-4 border-b border-line pb-4">
        <div>
          <p className="section-title mb-2">{language === 'zh' ? '概率视图' : 'Probability view'}</p>
          <h3 className="text-xl font-semibold tracking-tight text-ink">
            {language === 'zh' ? '预测概率' : 'Prediction probability'}
          </h3>
        </div>
        <div className="text-right">
          <div className="text-3xl font-semibold tracking-tight text-ink">{(aigcProbability * 100).toFixed(0)}%</div>
          <div className="text-xs uppercase tracking-[0.16em] text-subtle">{language === 'zh' ? 'AIGC 占比' : 'AIGC share'}</div>
        </div>
      </div>

      <div className={`grid items-center gap-5 ${isEmbedded ? '2xl:grid-cols-[minmax(300px,1fr)_minmax(220px,0.82fr)]' : 'xl:grid-cols-[minmax(300px,1fr)_minmax(220px,0.82fr)]'}`}>
        <div className={`min-h-[288px] ${isEmbedded ? 'h-72 2xl:h-80' : 'h-72'}`}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={62}
                outerRadius={isEmbedded ? 92 : 86}
                paddingAngle={3}
                dataKey="value"
                startAngle={90}
                endAngle={-270}
                stroke="rgba(217,225,234,0.9)"
                strokeWidth={2}
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#ffffff', borderColor: '#d9e1ea', borderRadius: '16px', color: '#111827' }}
                itemStyle={{ color: '#111827', fontSize: '12px' }}
                formatter={(value) => `${value.toFixed(1)}%`}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className={`grid gap-3 ${isEmbedded ? 'sm:grid-cols-2 2xl:grid-cols-1' : 'sm:grid-cols-2 xl:grid-cols-1'}`}>
          {data.map((item, index) => (
            <div key={item.name} className="rounded-[20px] border border-line bg-panel p-4">
              <div className="flex items-center gap-3">
                <span className="h-3 w-3 rounded-full" style={{ backgroundColor: COLORS[index] }} />
                <span className="text-sm font-medium text-ink">{item.name}</span>
              </div>
              <div className="mt-3 text-2xl font-semibold tracking-tight text-ink">{item.value.toFixed(1)}%</div>
            </div>
          ))}
        </div>
      </div>

      <p className="mt-5 rounded-[18px] border border-line bg-panel px-4 py-3 text-sm leading-7 text-muted">
        {language === 'zh'
          ? '该视图展示真实与 AI 生成概率分布，数值越偏向单侧说明模型判断越明确。'
          : 'This view shows the probability split between real and AI-generated outcomes. A stronger skew toward one side indicates a clearer model judgment.'}
      </p>

      {(thresholdModeLabel || typeof threshold === 'number') && (
        <div className="mt-3 rounded-[18px] border border-line bg-panel px-4 py-3 text-sm leading-7 text-muted">
          <p>
            {language === 'zh'
              ? `当前采用 ${thresholdModeLabel || 'Threshold Mode'}（判定阈值 ${thresholdValue !== null ? `${(thresholdValue * 100).toFixed(1)}%` : '-'}）。AIGC 概率 ≥ 当前阈值时判定为 AI 生成。`
              : `Current setting: ${thresholdModeLabel || 'Threshold Mode'} (decision threshold ${thresholdValue !== null ? `${(thresholdValue * 100).toFixed(1)}%` : '-'}). The image is classified as AIGC when AIGC probability is greater than or equal to the threshold.`}
          </p>
          {thresholdValue !== null && (
            <div className="mt-3">
              <div className="mb-2 flex items-center justify-between text-xs text-subtle">
                <span>{language === 'zh' ? 'AIGC 概率' : 'AIGC probability'} {(aigcProbability * 100).toFixed(1)}%</span>
                <span>{language === 'zh' ? '阈值' : 'Threshold'} {(thresholdValue * 100).toFixed(1)}%</span>
              </div>
              <div className="relative h-4 overflow-hidden rounded-full border border-line bg-white">
                <div className="absolute inset-y-0 left-0 bg-slate-800" style={{ width: `${probabilityPercent}%` }} />
                <div className="absolute inset-y-[-6px] w-[2px] bg-red-500" style={{ left: `calc(${thresholdPercent}% - 1px)` }} />
              </div>
            </div>
          )}
        </div>
      )}

      {!isEmbedded && (
        <p className="mt-3 rounded-[18px] border border-line bg-panel px-4 py-3 text-sm leading-7 text-muted">
          {language === 'zh'
            ? '当前部署采用三分支特征提取与稳定推理模式：语义结构和频域分布参与最终判定，噪声残差分支作为真实证据返回前端展示。'
            : 'The current deployment uses tri-branch feature extraction with safe inference: semantic and frequency branches drive the final decision, while the noise residual branch is returned as real evidence.'}
        </p>
      )}
    </>
  );

  if (isEmbedded) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
        className="flex h-full flex-col"
      >
        {content}
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
      className="glass-card flex h-full flex-col p-5 lg:p-6"
    >
      {content}
    </motion.div>
  );
};

export default ProbabilityChart;
