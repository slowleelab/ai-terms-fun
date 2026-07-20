import { defineConfig } from 'vitepress'

// GitHub Pages 项目站点需要 base = '/<repo>/'，本地预览也兼容
const isCI = !!process.env.CI
const base = isCI ? '/ai-terms-fun/' : '/'

// 未写词条统一指向路线图页，引导读者看到完整规划
const TODO = { text: '⬜ 待编写（见路线图）', link: '/roadmap' }

export default defineConfig({
  title: 'AI 黑话翻译器',
  description: '把 AI 圈的概念拆到骨头，再用人话讲清楚。硬核考据为主，通俗类比辅助，一个词读懂五层。',
  lang: 'zh-CN',
  lastUpdated: true,
  cleanUrls: true,
  base,

  // 词条间的交叉引用：允许指向路线图中已规划但尚未编写的词条。
  // 已写词条自然不会死链；拼错的不在白名单内，仍会被报出。
  // 前缀 \.? 同时匹配 /xxx 和 ./xxx 两种写法。
  ignoreDeadLinks: [
    /^<.*>$/,            // <slug> 这类占位
    /\/_template$/,      // 模板页内的示例链接
    /\.\/链接/,           // 模板里的中文占位「链接」
    /index$/,            // demos/<slug>/index 这类未生成的目录链接
    /^\.?\/(two-tower|cross-encoder|colbert|recall-rerank|top-k|hybrid-search|rrf|weighted-fusion|ltr)$/,
    /^\.?\/(hit-rate|recall-precision-at-k|mrr|ndcg|knowledge-base)$/,
  ],

  head: [
    ['meta', { name: 'author', content: 'ai-terms-fun' }],
    ['meta', { property: 'og:title', content: 'AI 黑话翻译器' }],
    ['meta', { property: 'og:description', content: '把 AI 概念拆到骨头，再用人话讲清楚。一个词读懂五层。' }],
    ['meta', { property: 'og:type', content: 'website' }],
  ],

  themeConfig: {
    siteTitle: '🧠 AI 黑话翻译器',

    nav: [
      { text: '首页', link: '/' },
      { text: '内容路线图', link: '/roadmap' },
      {
        text: 'GitHub',
        link: 'https://github.com/slowleelab/ai-terms-fun',
      },
    ],

    sidebar: [
      {
        text: '🗺️ 内容路线图',
        link: '/roadmap',
      },
      {
        text: '🏗️ 模型架构与训练',
        collapsed: false,
        items: [
          {
            text: '基础架构',
            collapsed: true,
            items: [
              { text: '✅ Transformer', link: '/transformer' },
              { text: '✅ 自注意力', link: '/self-attention' },
              { text: '✅ 多头注意力', link: '/multi-head-attention' },
              { text: '✅ 编码器-解码器', link: '/encoder-decoder' },
              { text: '✅ BERT（仅编码器）', link: '/bert' },
              { text: '✅ GPT / LLaMA（仅解码器）', link: '/gpt' },
              { text: '✅ 传统模型：CNN / RNN / LSTM', link: '/cnn-rnn-lstm' },
              { text: '✅ 模型组件：参数 / 层 / 激活函数', link: '/model-components' },
            ],
          },
          {
            text: '训练范式',
            collapsed: true,
            items: [
              { text: '✅ 预训练', link: '/pre-training' },
              { text: '✅ 微调', link: '/fine-tuning' },
              { text: '✅ 指令微调', link: '/instruction-tuning' },
              { text: '✅ RLHF', link: '/rlhf' },
              { text: '✅ 迁移学习', link: '/transfer-learning' },
              { text: '✅ 损失函数', link: '/loss-function' },
              { text: '✅ 优化器（Adam / AdamW）', link: '/optimizer' },
              { text: '✅ 过拟合 & 正则化', link: '/overfitting' },
            ],
          },
        ],
      },
      {
        text: '⚡ 推理与生成',
        collapsed: false,
        items: [
          { text: '✅ 自回归生成', link: '/autoregressive' },
          {
            text: '解码策略',
            collapsed: true,
            items: [
              { text: '✅ 贪婪解码', link: '/greedy-decoding' },
              { text: '✅ 束搜索 Beam Search', link: '/beam-search' },
              { text: '✅ Top-k 采样', link: '/top-k-sampling' },
              { text: '✅ Top-p 采样', link: '/top-p-sampling' },
            ],
          },
          { text: '✅ 温度 Temperature', link: '/temperature' },
          { text: '✅ 幻觉 Hallucination', link: '/hallucination' },
        ],
      },
      {
        text: '🗜️ 模型压缩与加速',
        collapsed: false,
        items: [
          { text: '✅ 量化（INT8 / INT4）', link: '/quantization' },
          { text: '✅ 知识蒸馏', link: '/knowledge-distillation' },
          { text: '✅ 剪枝', link: '/pruning' },
          { text: '✅ 推理引擎（vLLM / TensorRT-LLM）', link: '/inference-engine' },
        ],
      },
      {
        text: '🔤 数据表示与编码',
        collapsed: false,
        items: [
          {
            text: '文本预处理',
            collapsed: true,
            items: [
              { text: '✅ 分词器 Tokenizer', link: '/tokenizer' },
              { text: '✅ Token 词元', link: '/token' },
              { text: '✅ 分块 Chunking', link: '/chunking' },
            ],
          },
          {
            text: '向量表示',
            collapsed: false,
            items: [
              { text: '✅ Embedding - 嵌入', link: '/embedding' },
              { text: '✅ 高维向量', link: '/high-dim-vector' },
              { text: '✅ 稠密向量 vs 稀疏向量', link: '/dense-sparse-vector' },
              { text: '✅ 位置编码', link: '/positional-encoding' },
              { text: '✅ 多模态 Embedding（CLIP）', link: '/clip' },
            ],
          },
          {
            text: '上下文管理',
            collapsed: true,
            items: [
              { text: '✅ 上下文窗口 Context Window', link: '/context-window' },
            ],
          },
        ],
      },
      {
        text: '🔍 检索与索引',
        collapsed: false,
        items: [
          {
            text: '关键词检索',
            collapsed: true,
            items: [
              { text: '✅ 倒排索引', link: '/inverted-index' },
              { text: '✅ TF-IDF', link: '/tf-idf' },
              { text: '✅ BM25', link: '/bm25' },
            ],
          },
          {
            text: '向量检索',
            collapsed: true,
            items: [
              { text: '✅ KNN / ANN', link: '/knn-ann' },
              { text: '✅ 索引算法：HNSW / IVF / PQ / LSH', link: '/ann-algorithms' },
              { text: '✅ 算法库：Faiss / ScaNN / Annoy', link: '/ann-libraries' },
              { text: '✅ 向量数据库', link: '/vector-database' },
            ],
          },
          {
            text: '神经检索模型',
            collapsed: true,
            items: [
              { text: '双塔模型 Two-Tower', link: '/roadmap' },
              { text: '交叉编码器 Cross-encoder', link: '/roadmap' },
              { text: 'ColBERT 迟交互', link: '/roadmap' },
            ],
          },
          {
            text: '排序与融合',
            collapsed: true,
            items: [
              { text: '召回 vs 重排序', link: '/roadmap' },
              { text: 'Top-K', link: '/roadmap' },
              { text: '混合搜索 Hybrid Search', link: '/roadmap' },
              { text: 'RRF 倒数排名融合', link: '/roadmap' },
              { text: '加权重排', link: '/roadmap' },
              { text: '学习排序 LTR', link: '/roadmap' },
            ],
          },
        ],
      },
      {
        text: '📊 评估与应用',
        collapsed: false,
        items: [
          {
            text: '评估指标',
            collapsed: true,
            items: [
              { text: 'Hit Rate', link: '/roadmap' },
              { text: 'Recall@K / Precision@K', link: '/roadmap' },
              { text: 'MRR', link: '/roadmap' },
              { text: 'NDCG', link: '/roadmap' },
            ],
          },
          {
            text: '应用框架',
            collapsed: false,
            items: [
              { text: '✅ RAG - 检索增强生成', link: '/rag' },
              { text: '知识库 Knowledge Base', link: '/roadmap' },
            ],
          },
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
      { icon: 'github', link: 'https://github.com/slowleelab/ai-terms-fun' },
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
