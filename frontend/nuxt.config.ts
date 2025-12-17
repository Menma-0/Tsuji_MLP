// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },

  app: {
    head: {
      title: 'Onoma2DSP - オノマトペで音声編集',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: 'オノマトペを使って音声を編集するシステム' }
      ]
    }
  },

  // CORS設定（バックエンドAPIと通信するため）
  nitro: {
    devProxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  },

  // CSS設定
  css: ['~/assets/css/main.css']
})
