
import React, { useMemo } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { motion } from 'framer-motion';

const ProbabilityChart = ({ probabilities, probability, language = 'zh' }) => {
  const aigcProbability = typeof probabilities?.aigc === 'number' ? probabilities.aigc : (typeof probability === 'number' ? probability : 0);
  const realProbability = typeof probabilities?.real === 'number' ? probabilities.real : (1 - aigcProbability);
  const data = useMemo(() => [
    { name: language === 'zh' ? 'AI生成' : 'AI Generated', value: Math.max(0, aigcProbability * 100) },
    { name: language === 'zh' ? '真实图像' : 'Real Image', value: Math.max(0, realProbability * 100) },
  ], [aigcProbability, realProbability, language]);

  const COLORS = ['#DA205A', '#00D1FF'];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.3 }}
      className="glass-card rounded-2xl p-4 border border-white/10 flex flex-col items-center justify-center relative overflow-hidden h-full min-w-0"
    >
      <h3 className="h-8 leading-8 text-base font-medium text-white mb-3 text-center tracking-wider border-b border-white/10 pb-2 w-full">
        {language === 'zh' ? '预测概率' : 'Prediction Probability'}
      </h3>
      
      <div className="w-full h-56 relative">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              fill="#8884d8"
              paddingAngle={5}
              dataKey="value"
              startAngle={90}
              endAngle={-270}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="rgba(255,255,255,0.1)" strokeWidth={2} />
              ))}
            </Pie>
            <Tooltip 
              contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
              itemStyle={{ color: '#fff', fontSize: '12px' }}
              formatter={(value) => `${value.toFixed(1)}%`}
            />
            <Legend verticalAlign="bottom" height={36} iconType="circle" />
          </PieChart>
        </ResponsiveContainer>
        
        {/* Center Text */}
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none">
          <span className="text-3xl font-bold text-white block drop-shadow-lg">
            {(aigcProbability * 100).toFixed(0)}%
          </span>
          <span className="text-xs text-gray-400 tracking-widest">{language === 'zh' ? 'AI生成' : 'AI Generated'}</span>
        </div>
      </div>
      <p className="text-xs text-gray-400 text-center mt-3 leading-relaxed">
        {language === 'zh' ? '该图展示真实与AI生成概率分布，数值越偏向单侧说明模型判断越明确。' : 'This chart shows the probability distribution between real and AI-generated images. The more one-sided the value, the more definite the model\'s judgment.'}
      </p>
    </motion.div>
  );
};

export default ProbabilityChart;
