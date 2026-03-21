import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { motion } from 'framer-motion';

const labels = {
  zh: {
    titleEvidence: '分支证据分析',
    titleUsage: '分支路径使用',
    semantic: '语义',
    noise: '噪声',
    frequency: '频域',
    evidenceHint: '当前展示的是样本级证据强度，综合了分支自身判断与当前推理路径使用情况。',
    usageHint: '当前展示的是路径使用比例，更接近 gate/fusion 权重，而不是分支证据强弱。',
    baseOnlyNote: '在 base_only 模式下，noise 仅作诊断参考，不参与最终决策。',
  },
  en: {
    titleEvidence: 'Branch Evidence Analysis',
    titleUsage: 'Branch Path Usage',
    semantic: 'Semantic',
    noise: 'Noise',
    frequency: 'Frequency',
    evidenceHint: 'This chart shows sample-level evidence intensity by combining branch support with active path usage.',
    usageHint: 'This chart shows path usage ratios, which are closer to gate/fusion weights than to true branch evidence strength.',
    baseOnlyNote: 'In base_only mode, noise is diagnostic only and does not participate in the final decision.',
  },
};

const COLORS = ['#DA205A', '#00D1FF', '#7C3AED'];

const BranchContribution = ({ scores, analysisMode = 'support_weighted_usage', mode = 'base_only', language = 'zh' }) => {
  const copy = labels[language] || labels.zh;
  const rgb = typeof scores?.rgb === 'number' ? scores.rgb : 0;
  const noise = typeof scores?.noise === 'number' ? scores.noise : 0;
  const frequency = typeof scores?.frequency === 'number' ? scores.frequency : 0;
  const isEvidenceMode = analysisMode === 'support_weighted_usage';

  const data = useMemo(
    () => [
      { name: copy.semantic, score: rgb * 100 },
      { name: copy.noise, score: noise * 100 },
      { name: copy.frequency, score: frequency * 100 },
    ],
    [copy.frequency, copy.noise, copy.semantic, frequency, noise, rgb]
  );

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.6, delay: 0.4 }}
      className="glass-card rounded-2xl p-4 border border-white/10 flex flex-col justify-center relative overflow-hidden h-full min-w-0"
    >
      <h3 className="h-8 leading-8 text-base font-medium text-white mb-3 text-center tracking-wider border-b border-white/10 pb-2 w-full">
        {isEvidenceMode ? copy.titleEvidence : copy.titleUsage}
      </h3>

      <div className="w-full h-48 relative">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal vertical={false} stroke="rgba(255,255,255,0.05)" />
            <XAxis type="number" hide />
            <YAxis
              dataKey="name"
              type="category"
              tick={{ fill: '#aaa', fontSize: 12, fontWeight: 600 }}
              axisLine={false}
              tickLine={false}
              width={50}
            />
            <Tooltip
              cursor={{ fill: 'rgba(255,255,255,0.05)' }}
              contentStyle={{ backgroundColor: 'rgba(0,0,0,0.85)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
              itemStyle={{ color: '#fff', fontSize: '12px' }}
              formatter={(value) => `${Number(value).toFixed(1)}%`}
            />
            <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={20}>
              {data.map((entry, index) => (
                <Cell key={`cell-${entry.name}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-xs text-gray-400 text-center mt-3 leading-relaxed">
        {isEvidenceMode ? copy.evidenceHint : copy.usageHint}
      </p>
      {mode === 'base_only' && (
        <p className="text-[11px] text-cyan-300/80 text-center mt-2 leading-relaxed">
          {copy.baseOnlyNote}
        </p>
      )}
    </motion.div>
  );
};

export default BranchContribution;
