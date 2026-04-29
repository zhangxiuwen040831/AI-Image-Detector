import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { LogOut, Settings, User, Clock, Users } from 'lucide-react';

const parseAuthJson = async (response, language) => {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(language === 'zh' ? '服务器返回异常，请检查后端接口。' : 'Unexpected server response. Please check the backend API.');
  }
};

const UserProfile = ({ user, onLogout, onBack, language, isAdmin = false, onPasswordChange, onNicknameChange }) => {
  const [showPasswordForm, setShowPasswordForm] = useState(false);
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [userList, setUserList] = useState([]);
  const [showUserList, setShowUserList] = useState(false);
  const [loadingUserList, setLoadingUserList] = useState(false);
  const [userLogs, setUserLogs] = useState([]);
  const [showUserLogs, setShowUserLogs] = useState(false);
  const [loadingUserLogs, setLoadingUserLogs] = useState(false);
  const [selectedUserId, setSelectedUserId] = useState(null);
  const [selectedUserName, setSelectedUserName] = useState('');
  const [resetPasswordUserId, setResetPasswordUserId] = useState(null);
  const [resetPasswordValue, setResetPasswordValue] = useState('');
  const [editingNickname, setEditingNickname] = useState(false);
  const [nicknameValue, setNicknameValue] = useState(user.nickname || '');

  const t = {
    zh: {
      title: '个人主页',
      accountInfo: '账户信息',
      username: '用户名',
      nickname: '昵称',
      userType: '用户类型',
      normal: '普通用户',
      admin: '管理员',
      changePassword: '修改密码',
      oldPassword: '原密码',
      newPassword: '新密码',
      confirmPassword: '确认密码',
      submit: '提交',
      cancel: '取消',
      logout: '退出登录',
      back: '返回',
      lastLogin: '最近登录',
      lastLogout: '最近登出',
      userList: '用户列表',
      noData: '暂无数据',
      success: '操作成功',
      error: '操作失败',
      manageUsers: '管理用户',
      operationLogs: '操作日志',
      selectedUserLogs: '用户操作日志',
      viewLogs: '查看日志',
      resetPassword: '重置密码',
      enterNewPassword: '请输入新密码',
      passwordMismatch: '两次输入的密码不一致',
      passwordTooShort: '密码长度至少6个字符',
      editNickname: '修改昵称',
      save: '保存',
      operateType: '操作类型',
      operateContent: '操作内容',
      operateIp: '操作 IP',
      operateTime: '操作时间',
    },
    en: {
      title: 'User Profile',
      accountInfo: 'Account Information',
      username: 'Username',
      nickname: 'Nickname',
      userType: 'User Type',
      normal: 'Normal User',
      admin: 'Administrator',
      changePassword: 'Change Password',
      oldPassword: 'Old Password',
      newPassword: 'New Password',
      confirmPassword: 'Confirm Password',
      submit: 'Submit',
      cancel: 'Cancel',
      logout: 'Logout',
      back: 'Back',
      lastLogin: 'Last Login',
      lastLogout: 'Last Logout',
      userList: 'User List',
      noData: 'No Data',
      success: 'Success',
      error: 'Error',
      manageUsers: 'Manage Users',
      operationLogs: 'Operation Logs',
      selectedUserLogs: 'User Operation Logs',
      viewLogs: 'View Logs',
      resetPassword: 'Reset Password',
      enterNewPassword: 'Please enter new password',
      passwordMismatch: 'Passwords do not match',
      passwordTooShort: 'Password must be at least 6 characters',
      editNickname: 'Edit Nickname',
      save: 'Save',
      operateType: 'Type',
      operateContent: 'Content',
      operateIp: 'IP',
      operateTime: 'Time',
    },
  };

  const text = t[language] || t.zh;

  const getAuthHeaders = () => {
    const token = localStorage.getItem('token');
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    };
  };

  const handlePasswordChange = async (e) => {
    e.preventDefault();
    setError('');
    setMessage('');

    if (newPassword !== confirmPassword) {
      setError(text.passwordMismatch);
      return;
    }

    if (newPassword.length < 6) {
      setError(text.passwordTooShort);
      return;
    }

    try {
      await onPasswordChange(oldPassword, newPassword);
      setMessage(text.success);
      setShowPasswordForm(false);
      setOldPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err) {
      setError(err.message);
    }
  };

  const handleNicknameSave = async () => {
    if (!nicknameValue.trim()) return;
    try {
      await onNicknameChange(nicknameValue);
      setEditingNickname(false);
      window.location.reload();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleLogout = async () => {
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
      localStorage.removeItem('user');
      onLogout();
    }
  };

  const handleAdminResetPassword = async (userId) => {
    setResetPasswordUserId(userId);
    setResetPasswordValue('');
    setError('');
  };

  const handleResetPasswordSubmit = async (e) => {
    e.preventDefault();

    if (resetPasswordValue.length < 6) {
      setError(text.passwordTooShort);
      return;
    }

    try {
      const response = await fetch(`/auth/admin/user/${resetPasswordUserId}/password`, {
        method: 'PUT',
        headers: getAuthHeaders(),
        body: JSON.stringify({ new_password: resetPasswordValue }),
      });

      const data = await parseAuthJson(response, language);

      if (!response.ok) {
        throw new Error(data.message || data.detail || (language === 'zh' ? '密码重置失败' : 'Password reset failed'));
      }

      setResetPasswordUserId(null);
      setResetPasswordValue('');
      alert(language === 'zh' ? '密码重置成功' : 'Password reset successfully');
    } catch (err) {
      setError(err.message);
    }
  };

  const loadUserList = async () => {
    setLoadingUserList(true);
    try {
      const response = await fetch('/auth/admin/users', {
        method: 'GET',
        headers: getAuthHeaders(),
      });

      const data = await parseAuthJson(response, language);

      if (!response.ok) {
        throw new Error(data.message || data.detail || (language === 'zh' ? '获取用户列表失败' : 'Failed to get user list'));
      }

      setUserList(data.users || []);
      setShowUserList(true);
      setShowUserLogs(false);
      setUserLogs([]);
      setSelectedUserId(null);
      setSelectedUserName('');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingUserList(false);
    }
  };

  const loadUserLogs = async (targetUser) => {
    if (!targetUser?.id) return;
    setLoadingUserLogs(true);
    setSelectedUserId(targetUser.id);
    setSelectedUserName(targetUser.username);
    try {
      const response = await fetch(`/auth/admin/logs?user_id=${encodeURIComponent(targetUser.id)}&limit=100`, {
        method: 'GET',
        headers: getAuthHeaders(),
      });

      const data = await parseAuthJson(response, language);

      if (!response.ok) {
        throw new Error(data.message || data.detail || (language === 'zh' ? '获取日志失败' : 'Failed to get logs'));
      }

      setUserLogs(data.logs || []);
      setShowUserLogs(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingUserLogs(false);
    }
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleString(language === 'zh' ? 'zh-CN' : 'en-US');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 py-8">
      <div className="mx-auto max-w-5xl px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="rounded-[32px] border border-line bg-white p-8 shadow-lg"
        >
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-2xl font-semibold text-ink">{text.title}</h1>
            <button
              onClick={onBack}
              className="rounded-full border border-line bg-panel px-4 py-2 text-sm font-medium text-ink hover:bg-slate-100 transition-colors"
            >
              {text.back}
            </button>
          </div>

          {error && (
            <div className="mb-4 rounded-[16px] border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
              {text.error}: {error}
            </div>
          )}

          {message && (
            <div className="mb-4 rounded-[16px] border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-600">
              {text.success}: {message}
            </div>
          )}

          <div className="mb-8">
            <h2 className="section-title mb-4">{text.accountInfo}</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between rounded-[16px] border border-line bg-panel p-4">
                <div className="flex items-center gap-3">
                  <User className="h-5 w-5 text-muted" />
                  <span className="text-sm text-muted">{text.username}</span>
                </div>
                <span className="font-medium text-ink">{user.username}</span>
              </div>

              <div className="flex items-center justify-between rounded-[16px] border border-line bg-panel p-4">
                <div className="flex items-center gap-3">
                  <Settings className="h-5 w-5 text-muted" />
                  <span className="text-sm text-muted">{text.nickname}</span>
                </div>
                <div className="flex items-center gap-2">
                  {editingNickname ? (
                    <form onSubmit={(e) => { e.preventDefault(); handleNicknameSave(); }} className="flex items-center gap-2">
                      <input
                        type="text"
                        value={nicknameValue}
                        onChange={(e) => setNicknameValue(e.target.value)}
                        className="rounded-[12px] border border-line bg-white px-3 py-1.5 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-primary"
                        autoFocus
                      />
                      <button
                        type="submit"
                        className="rounded-full border border-primary bg-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-primary/90"
                      >
                        {text.save}
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setEditingNickname(false);
                          setNicknameValue(user.nickname || '');
                        }}
                        className="rounded-full border border-line bg-white px-3 py-1.5 text-xs font-medium text-ink hover:bg-slate-100"
                      >
                        {text.cancel}
                      </button>
                    </form>
                  ) : (
                    <>
                      <span className="font-medium text-ink">{user.nickname}</span>
                      <button
                        onClick={() => setEditingNickname(true)}
                        className="text-xs text-primary hover:underline"
                      >
                        {text.editNickname}
                      </button>
                    </>
                  )}
                </div>
              </div>

              <div className="flex items-center justify-between rounded-[16px] border border-line bg-panel p-4">
                <div className="flex items-center gap-3">
                  <Users className="h-5 w-5 text-muted" />
                  <span className="text-sm text-muted">{text.userType}</span>
                </div>
                <span className={`font-medium ${user.user_type === 'admin' ? 'text-amber-600' : 'text-ink'}`}>
                  {user.user_type === 'admin' ? text.admin : text.normal}
                </span>
              </div>

              {user.last_login_time && (
                <div className="flex items-center justify-between rounded-[16px] border border-line bg-panel p-4">
                  <div className="flex items-center gap-3">
                    <Clock className="h-5 w-5 text-muted" />
                    <span className="text-sm text-muted">{text.lastLogin}</span>
                  </div>
                  <span className="font-medium text-ink">{formatDate(user.last_login_time)}</span>
                </div>
              )}

              {user.last_logout_time && (
                <div className="flex items-center justify-between rounded-[16px] border border-line bg-panel p-4">
                  <div className="flex items-center gap-3">
                    <Clock className="h-5 w-5 text-muted" />
                    <span className="text-sm text-muted">{text.lastLogout}</span>
                  </div>
                  <span className="font-medium text-ink">{formatDate(user.last_logout_time)}</span>
                </div>
              )}
            </div>
          </div>

          {!showPasswordForm ? (
            <div className="space-y-3">
              <button
                onClick={() => setShowPasswordForm(true)}
                className="w-full flex items-center justify-center gap-2 rounded-full border border-primary bg-primary px-4 py-3 text-white font-medium transition-all hover:bg-primary/90"
              >
                <Settings className="h-4 w-4" />
                <span>{text.changePassword}</span>
              </button>

              {isAdmin && (
                <div className="grid gap-3">
                  <button
                    onClick={loadUserList}
                    disabled={loadingUserList}
                    className="w-full flex items-center justify-center gap-2 rounded-full border border-line bg-panel px-4 py-3 font-medium text-ink transition-all hover:bg-slate-100 disabled:opacity-50"
                  >
                    <Users className="h-4 w-4" />
                    <span>{loadingUserList ? (language === 'zh' ? '加载中...' : 'Loading...') : text.manageUsers}</span>
                  </button>
                </div>
              )}

              <button
                onClick={handleLogout}
                className="w-full flex items-center justify-center gap-2 rounded-full border border-red-200 bg-red-50 px-4 py-3 font-medium text-red-600 transition-all hover:bg-red-100"
              >
                <LogOut className="h-4 w-4" />
                <span>{text.logout}</span>
              </button>
            </div>
          ) : (
            <form onSubmit={handlePasswordChange} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-ink mb-2">
                  {text.oldPassword}
                </label>
                <input
                  type="password"
                  value={oldPassword}
                  onChange={(e) => setOldPassword(e.target.value)}
                  className="w-full rounded-[16px] border border-line bg-panel px-4 py-3 text-ink focus:outline-none focus:ring-2 focus:ring-primary"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-ink mb-2">
                  {text.newPassword}
                </label>
                <input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full rounded-[16px] border border-line bg-panel px-4 py-3 text-ink focus:outline-none focus:ring-2 focus:ring-primary"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-ink mb-2">
                  {text.confirmPassword}
                </label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full rounded-[16px] border border-line bg-panel px-4 py-3 text-ink focus:outline-none focus:ring-2 focus:ring-primary"
                  required
                />
              </div>

              <div className="flex gap-3">
                <button
                  type="submit"
                  className="flex-1 rounded-full border border-primary bg-primary px-4 py-3 text-white font-medium transition-all hover:bg-primary/90"
                >
                  {text.submit}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowPasswordForm(false);
                    setOldPassword('');
                    setNewPassword('');
                    setConfirmPassword('');
                    setError('');
                  }}
                  className="flex-1 rounded-full border border-line bg-panel px-4 py-3 font-medium text-ink transition-all hover:bg-slate-100"
                >
                  {text.cancel}
                </button>
              </div>
            </form>
          )}

          {showUserList && (
            <div className="mt-8">
              <h2 className="section-title mb-4">{text.userList}</h2>
              {userList.length === 0 ? (
                <p className="text-sm text-muted text-center py-4">{text.noData}</p>
              ) : (
                <div className="space-y-3">
                  {userList.map((u) => (
                    <div
                      key={u.id}
                      className={`flex flex-col gap-4 rounded-[16px] border p-4 transition-colors md:flex-row md:items-center md:justify-between ${
                        selectedUserId === u.id ? 'border-primary bg-white shadow-sm' : 'border-line bg-panel'
                      }`}
                    >
                      <div>
                        <p className="font-medium text-ink">{u.nickname}</p>
                        <p className="text-xs text-muted">{u.username}</p>
                        <p className="text-xs text-muted mt-1">
                          {text.lastLogin}: {formatDate(u.last_login_time)}
                        </p>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 md:justify-end">
                        <button
                          onClick={() => loadUserLogs(u)}
                          disabled={loadingUserLogs && selectedUserId === u.id}
                          className="rounded-full border border-primary bg-primary px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-primary/90 disabled:opacity-50"
                        >
                          {loadingUserLogs && selectedUserId === u.id ? (language === 'zh' ? '加载中...' : 'Loading...') : text.viewLogs}
                        </button>
                        {resetPasswordUserId === u.id ? (
                          <form onSubmit={handleResetPasswordSubmit} className="flex items-center gap-2">
                            <input
                              type="password"
                              value={resetPasswordValue}
                              onChange={(e) => setResetPasswordValue(e.target.value)}
                              placeholder={text.enterNewPassword}
                              className="rounded-[12px] border border-line bg-white px-3 py-1.5 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-primary"
                              autoFocus
                            />
                            <button
                              type="submit"
                              className="rounded-full border border-primary bg-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-primary/90"
                            >
                              {text.submit}
                            </button>
                            <button
                              type="button"
                              onClick={() => setResetPasswordUserId(null)}
                              className="rounded-full border border-line bg-white px-3 py-1.5 text-xs font-medium text-ink hover:bg-slate-100"
                            >
                              {text.cancel}
                            </button>
                          </form>
                        ) : (
                          <button
                            onClick={() => handleAdminResetPassword(u.id)}
                            className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-600 hover:bg-amber-100 transition-colors"
                          >
                            {text.resetPassword}
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {showUserLogs && (
            <div className="mt-8">
              <h2 className="section-title mb-4">
                {text.selectedUserLogs}{selectedUserName ? ` - ${selectedUserName}` : ''}
              </h2>
              {userLogs.length === 0 ? (
                <p className="text-sm text-muted text-center py-4">{text.noData}</p>
              ) : (
                <div className="space-y-3">
                  {userLogs.map((log) => (
                    <div key={log.id} className="rounded-[16px] border border-line bg-panel p-4">
                      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
                        <div>
                          <p className="font-medium text-ink">{log.username}</p>
                          <p className="text-xs text-muted mt-1">{text.operateType}: {log.operate_type}</p>
                          <p className="text-xs text-muted mt-1">{text.operateContent}: {log.operate_content}</p>
                        </div>
                        <div className="text-xs text-muted md:text-right">
                          <p>{text.operateIp}: {log.operate_ip || '-'}</p>
                          <p className="mt-1">{text.operateTime}: {formatDate(log.created_at)}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default UserProfile;
