import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AI 黑话翻译器',
  description: '把 AI 圈的黑话，从听不懂的咒语翻译成能笑出声的常识。硬核考据，冷面吐槽，一个词读懂五层。',
  lang: 'zh-CN',
  lastUpdated: true,
  cleanUrls: true,

  head: [
    ['meta', { name: 'author', content: 'ai-terms-fun' }],
    ['meta', { property: 'og:title', content: 'AI 黑话翻译器' }],
    ['meta', { property: 'og:description', content: '一个词读懂五层。硬核考据 + 冷幽默。' }],
    ['meta', { property: 'og:type', content: 'website' }],
  ],

  themeConfig: {
    siteTitle: '🧠 AI 黑话翻译器',

    nav: [
      { text: '首页', link: '/' },
      { text: '热点', link: '/rag' },
      {
        text: 'GitHub',
        link: 'https://github.com/your-name/ai-terms-fun',
      },
    ],

    sidebar: [
      {
        text: '🔥 热点',
        items: [
          { text: 'RAG - 检索增强生成', link: '/rag' },
        ],
      },
      {
        text: '📚 经典',
        items: [
          { text: '（征集投稿中）', link: '/' },
        ],
      },
      {
        text: '🧪 工程',
        items: [
          { text: '（征集投稿中）', link: '/' },
        ],
      },
      {
        text: '关于',
        items: [
          { text: '贡献指南', link: '/contributing' },
          { text: '词条模板', link: '/_template' },
        ],
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/your-name/ai-terms-fun' },
    ],

    outline: {
      level: [2, 3],
      label: '本页目录',
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇',
    },

    lastUpdatedText: '最后更新',

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            displayDetails: '显示详情',
            resetButtonTitle: '清除',
            backButtonTitle: '返回',
            noResultsText: '没有结果',
            footer: {
              selectText: '选择',
              navigateText: '切换',
              closeText: '关闭',
            },
          },
        },
      },
    },

    footer: {
      message: '内容采用 CC BY-SA 4.0，代码采用 MIT。',
      copyright: '© 2026 ai-terms-fun',
    },
  },

  markdown: {
    lineNumbers: false,
    config(md) {
      // 支持 ::: 风格的容器
    },
  },
})
