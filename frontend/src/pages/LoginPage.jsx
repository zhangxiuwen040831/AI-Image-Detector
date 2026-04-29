import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Languages } from 'lucide-react';
import ShieldAnimation from '../components/ShieldAnimation';

const SESSION_KEY = 'ai_image_detector_user';
const API_URL = (import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '')).trim();
const API_BASE = API_URL ? API_URL.replace(/\/$/, '') : '';
const LOGIN_ENDPOINT = API_BASE ? `${API_BASE}/login` : '/login';
const REGISTER_ENDPOINT = API_BASE ? `${API_BASE}/register` : '/register';

const normalizeUser = (user) => {
  if (!user) return null;
  const userId = user.user_id ?? user.id;
  if (!userId || !user.username || !user.user_type) {
    return null;
  }
  return {
    user_id: userId,
    id: userId,
    username: user.username,
    nickname: user.nickname || '用户',
    user_type: user.user_type,
    last_login_time: user.last_login_time ?? null,
    last_logout_time: user.last_logout_time ?? null,
  };
};

const parseJsonSafely = (text, language) => {
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(language === 'zh' ? '服务器返回异常，请检查后端接口。' : 'Unexpected server response. Please check the backend API.');
  }
};

const isSuccessResponse = (response, data) => {
  if (typeof data?.success === 'boolean') {
    return response.ok && data.success;
  }
  return response.ok;
};

const LoginPage = ({ onLoginSuccess, language, onLanguageChange }) => {
  const [mode, setMode] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [nickname, setNickname] = useState('');
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [loading, setLoading] = useState(false);

  const text = {
    zh: {
      loginTitle: '用户登录',
      registerTitle: '注册账号',
      loginSubtitle: '请输入账号信息以进入图像取证工作台',
      registerSubtitle: '创建普通用户账号后即可登录系统',
      username: '用户名',
      password: '密码',
      confirmPassword: '确认密码',
      nickname: '昵称',
      nicknamePlaceholder: '用户',
      login: '登录',
      register: '注册账号',
      backToLogin: '已有账号？返回登录',
      switchToRegister: '注册账号',
      error: '错误',
      success: '成功',
      loggingIn: '登录中...',
      registering: '注册中...',
      mismatch: '两次输入的密码不一致',
      shortPassword: '密码长度至少6个字符',
      shortUsername: '用户名长度至少3个字符',
    },
    en: {
      loginTitle: 'User Login',
      registerTitle: 'Register Account',
      loginSubtitle: 'Sign in to access the image forensics workspace',
      registerSubtitle: 'Create a normal user account to sign in to the system',
      username: 'Username',
      password: 'Password',
      confirmPassword: 'Confirm Password',
      nickname: 'Nickname',
      nicknamePlaceholder: 'User',
      login: 'Login',
      register: 'Register Account',
      backToLogin: 'Already have an account? Back to login',
      switchToRegister: 'Register Account',
      error: 'Error',
      success: 'Success',
      loggingIn: 'Logging in...',
      registering: 'Registering...',
      mismatch: 'Passwords do not match',
      shortPassword: 'Password must be at least 6 characters',
      shortUsername: 'Username must be at least 3 characters',
    },
  }[language] || {
    loginTitle: '用户登录',
    registerTitle: '注册账号',
    loginSubtitle: '请输入账号信息以进入图像取证工作台',
    registerSubtitle: '创建普通用户账号后即可登录系统',
    username: '用户名',
    password: '密码',
    confirmPassword: '确认密码',
    nickname: '昵称',
    nicknamePlaceholder: '用户',
    login: '登录',
    register: '注册账号',
    backToLogin: '已有账号？返回登录',
    switchToRegister: '注册账号',
    error: '错误',
    success: '成功',
    loggingIn: '登录中...',
    registering: '注册中...',
    mismatch: '两次输入的密码不一致',
    shortPassword: '密码长度至少6个字符',
    shortUsername: '用户名长度至少3个字符',
  };

  const resetMessages = () => {
    setError('');
    setSuccessMessage('');
  };

  const switchMode = (nextMode) => {
    resetMessages();
    setMode(nextMode);
    setPassword('');
    setConfirmPassword('');
    if (nextMode === 'register' && !nickname) {
      setNickname('用户');
    }
  };

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    resetMessages();
    setLoading(true);

    try {
      const response = await fetch(LOGIN_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      const rawText = await response.text();
      const data = parseJsonSafely(rawText, language);
      console.log('login response:', response.status, data);

      if (!isSuccessResponse(response, data)) {
        throw new Error(data.message || data.detail || (language === 'zh' ? '登录失败' : 'Login failed'));
      }

      if (data.access_token) {
        localStorage.setItem('token', data.access_token);
      }
      const normalizedUser = normalizeUser(data.user);
      if (!normalizedUser) {
        throw new Error(language === 'zh' ? '登录返回的用户信息无效' : 'Invalid user payload returned from login');
      }
      localStorage.setItem(SESSION_KEY, JSON.stringify(normalizedUser));
      console.log('[Auth] login success', normalizedUser.username, normalizedUser.user_type);
      onLoginSuccess(normalizedUser);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRegisterSubmit = async (e) => {
    e.preventDefault();
    resetMessages();

    if (username.trim().length < 3) {
      setError(text.shortUsername);
      return;
    }

    if (password.length < 6) {
      setError(text.shortPassword);
      return;
    }

    if (password !== confirmPassword) {
      setError(text.mismatch);
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(REGISTER_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: username.trim(),
          password,
          nickname: nickname.trim() || '用户',
        }),
      });

      const rawText = await response.text();
      const data = parseJsonSafely(rawText, language);

      if (!isSuccessResponse(response, data)) {
        throw new Error(data.message || data.detail || (language === 'zh' ? '注册失败' : 'Registration failed'));
      }

      setSuccessMessage(data.message || (language === 'zh' ? '注册成功，请登录' : 'Registration successful, please sign in.'));
      setMode('login');
      setPassword('');
      setConfirmPassword('');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="absolute top-4 right-4">
        <button
          onClick={() => onLanguageChange(language === 'zh' ? 'en' : 'zh')}
          className="flex items-center gap-2 rounded-full border border-line bg-white px-4 py-2 text-sm font-medium text-ink shadow-soft transition-colors hover:bg-panel"
        >
          <Languages className="h-4 w-4" />
          <span>{language === 'zh' ? 'English' : '中文'}</span>
        </button>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mx-4 w-full max-w-md"
      >
        <div className="rounded-[32px] border border-line bg-white p-8 shadow-lg">
          <div className="mb-8 flex flex-col items-center">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-line bg-panel shadow-soft">
              <ShieldAnimation />
            </div>
            <h1 className="text-2xl font-semibold text-ink">
              {mode === 'login' ? text.loginTitle : text.registerTitle}
            </h1>
            <p className="mt-3 text-center text-sm leading-7 text-muted">
              {mode === 'login' ? text.loginSubtitle : text.registerSubtitle}
            </p>
          </div>

          {error && (
            <div className="mb-4 rounded-[16px] border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
              {text.error}: {error}
            </div>
          )}

          {successMessage && (
            <div className="mb-4 rounded-[16px] border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
              {text.success}: {successMessage}
            </div>
          )}

          {mode === 'login' ? (
            <form onSubmit={handleLoginSubmit} className="space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium text-ink">{text.username}</label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full rounded-[16px] border border-line bg-panel px-4 py-3 text-ink focus:outline-none focus:ring-2 focus:ring-primary"
                  required
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-ink">{text.password}</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full rounded-[16px] border border-line bg-panel px-4 py-3 text-ink focus:outline-none focus:ring-2 focus:ring-primary"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-full border border-primary bg-primary px-4 py-3 font-medium text-white transition-all hover:bg-primary/90 disabled:opacity-50"
              >
                {loading ? text.loggingIn : text.login}
              </button>

              <button
                type="button"
                onClick={() => switchMode('register')}
                className="w-full rounded-full border border-line bg-white px-4 py-3 font-medium text-ink transition-all hover:bg-panel"
              >
                {text.switchToRegister}
              </button>
            </form>
          ) : (
            <form onSubmit={handleRegisterSubmit} className="space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium text-ink">{text.username}</label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full rounded-[16px] border border-line bg-panel px-4 py-3 text-ink focus:outline-none focus:ring-2 focus:ring-primary"
                  required
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-ink">{text.nickname}</label>
                <input
                  type="text"
                  value={nickname}
                  onChange={(e) => setNickname(e.target.value)}
                  placeholder={text.nicknamePlaceholder}
                  className="w-full rounded-[16px] border border-line bg-panel px-4 py-3 text-ink focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-ink">{text.password}</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full rounded-[16px] border border-line bg-panel px-4 py-3 text-ink focus:outline-none focus:ring-2 focus:ring-primary"
                  required
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-ink">{text.confirmPassword}</label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full rounded-[16px] border border-line bg-panel px-4 py-3 text-ink focus:outline-none focus:ring-2 focus:ring-primary"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-full border border-primary bg-primary px-4 py-3 font-medium text-white transition-all hover:bg-primary/90 disabled:opacity-50"
              >
                {loading ? text.registering : text.register}
              </button>

              <button
                type="button"
                onClick={() => switchMode('login')}
                className="w-full rounded-full border border-line bg-white px-4 py-3 font-medium text-ink transition-all hover:bg-panel"
              >
                {text.backToLogin}
              </button>
            </form>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default LoginPage;
