module.exports = {
  root: true,
  env: {
    node: true
  },
  extends: [
    'plugin:vue/essential',
    'eslint:recommended'
  ],
  parserOptions: {
    parser: 'babel-eslint'
  },
  rules: {
    'no-console': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
    'no-debugger': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
    // 添加更宽松的规则，避免过多警告
    'vue/no-unused-components': 'warn',
    'no-unused-vars': 'warn',
    // 禁用多词组件名规则
    'vue/multi-word-component-names': 'off'
  }
} 