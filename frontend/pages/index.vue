<template>
  <div class="container">
    <header class="header">
      <h1 class="title">🎵 OnomatoDSP</h1>
      <p class="subtitle">オノマトペで音声を編集するシステム（最大3回まで編集可能）</p>
    </header>

    <main class="main-content">
      <!-- 説明セクション -->
      <section class="info-section">
        <h2>使い方</h2>
        <ol>
          <li>音声ファイルをアップロード</li>
          <li>現在の音を表すオノマトペを入力（例: チリン）</li>
          <li>変換後の音を表すオノマトペを入力（例: ゴロゴロ）</li>
          <li>「音声を変換」ボタンをクリック</li>
          <li>最大3回まで連続して編集できます</li>
        </ol>
      </section>

      <!-- 初回編集前（0回目） -->
      <section v-if="currentEditCount === 0" class="form-section">
        <div class="form-card">
          <!-- 音声ファイルアップロード -->
          <div class="form-group">
            <label class="label">
              <span class="label-icon">🎧</span>
              音声ファイル
            </label>
            <div class="file-upload-area" @click="triggerFileInput">
              <input
                ref="fileInput"
                type="file"
                accept="audio/*"
                @change="handleFileChange"
                style="display: none;"
              />
              <div v-if="!audioFile" class="upload-placeholder">
                <p>📁 クリックしてファイルを選択</p>
                <p class="upload-hint">対応形式: WAV, MP3, FLAC, OGG</p>
              </div>
              <div v-else class="upload-success">
                <p>✓ {{ audioFile.name }}</p>
                <button @click.stop="clearFile" class="clear-btn">×</button>
              </div>
            </div>
          </div>

          <!-- アップロードした音声プレーヤー -->
          <div v-if="editHistory.length > 0" class="uploaded-audio-player">
            <label class="label">
              <span class="label-icon">🔊</span>
              アップロードされた音声
            </label>
            <audio controls :src="editHistory[0].audioUrl" class="audio-player"></audio>
          </div>

          <!-- オノマトペ入力 -->
          <div class="onomatopoeia-inputs">
            <div class="form-group">
              <label class="label">
                <span class="label-icon">🔊</span>
                元の音（Source）
              </label>
              <input
                v-model="sourceOnoma"
                type="text"
                class="input"
                placeholder="例: チリン"
                @input="validateKatakana('source')"
              />
              <p v-if="errors.source" class="error-message">{{ errors.source }}</p>
              <p class="hint">カタカナで入力してください</p>
            </div>

            <div class="arrow">→</div>

            <div class="form-group">
              <label class="label">
                <span class="label-icon">🎵</span>
                変換後の音（Target）
              </label>
              <input
                v-model="targetOnoma"
                type="text"
                class="input"
                placeholder="例: ゴロゴロ"
                @input="validateKatakana('target')"
              />
              <p v-if="errors.target" class="error-message">{{ errors.target }}</p>
              <p class="hint">カタカナで入力してください</p>
            </div>
          </div>

          <!-- パラメータ調整（オプション） -->
          <details class="advanced-settings">
            <summary>詳細設定（オプション）</summary>
            <div class="settings-grid">
              <div class="form-group">
                <label class="label">Amplification Factor</label>
                <input
                  v-model.number="amplificationFactor"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  class="slider"
                />
                <span class="value-display">{{ amplificationFactor.toFixed(1) }}</span>
              </div>

              <div class="form-group">
                <label class="label">Lambda Attention</label>
                <input
                  v-model.number="lambdaAtt"
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  class="slider"
                />
                <span class="value-display">{{ lambdaAtt.toFixed(1) }}</span>
              </div>
            </div>
          </details>

          <!-- 実行ボタン -->
          <button
            @click="processAudio"
            :disabled="!canProcess || isProcessing"
            class="process-btn"
          >
            <span v-if="!isProcessing">🎨 音声を変換</span>
            <span v-else>⏳ 処理中...</span>
          </button>
        </div>
      </section>

      <!-- 編集後（1-2回目） -->
      <section v-if="currentEditCount > 0 && currentEditCount < maxEdits" class="history-section">
        <h2>編集履歴 ({{ currentEditCount }}/{{ maxEdits }}回)</h2>

        <!-- 全ての音声を表示 -->
        <div class="history-list">
          <div
            v-for="(edit, index) in editHistory"
            :key="edit.step"
            :class="['history-item', edit.step === 0 ? 'original' : 'edited']"
          >
            <div class="history-header">
              <h3>{{ edit.step === 0 ? '元の音声' : `編集 ${edit.step}回目` }}</h3>
              <span v-if="edit.step > 0" class="edit-badge">
                {{ edit.sourceOnoma }} → {{ edit.targetOnoma }}
              </span>
            </div>
            <audio controls :src="edit.audioUrl" class="audio-player"></audio>
          </div>
        </div>

        <!-- 次の編集フォーム -->
        <div class="next-edit-form">
          <h3>さらに編集 (残り {{ maxEdits - currentEditCount }}回)</h3>
          <p class="hint-text">前回の編集結果をさらに変換します</p>

          <div class="onomatopoeia-inputs">
            <div class="form-group">
              <label class="label">現在の音（前回結果）</label>
              <input
                v-model="sourceOnoma"
                type="text"
                class="input"
                placeholder="例: チリン"
                @input="validateKatakana('source')"
              />
              <p v-if="errors.source" class="error-message">{{ errors.source }}</p>
            </div>

            <div class="arrow">→</div>

            <div class="form-group">
              <label class="label">変換後の音</label>
              <input
                v-model="targetOnoma"
                type="text"
                class="input"
                placeholder="例: ゴロゴロ"
                @input="validateKatakana('target')"
              />
              <p v-if="errors.target" class="error-message">{{ errors.target }}</p>
            </div>
          </div>

          <details class="advanced-settings">
            <summary>詳細設定（オプション）</summary>
            <div class="settings-grid">
              <div class="form-group">
                <label class="label">Amplification Factor</label>
                <input
                  v-model.number="amplificationFactor"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  class="slider"
                />
                <span class="value-display">{{ amplificationFactor.toFixed(1) }}</span>
              </div>

              <div class="form-group">
                <label class="label">Lambda Attention</label>
                <input
                  v-model.number="lambdaAtt"
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  class="slider"
                />
                <span class="value-display">{{ lambdaAtt.toFixed(1) }}</span>
              </div>
            </div>
          </details>

          <button
            @click="processAudio"
            :disabled="!canContinueEdit || isProcessing"
            class="process-btn edit-btn"
          >
            <span v-if="!isProcessing">✏️ さらに音声を編集</span>
            <span v-else>⏳ 処理中...</span>
          </button>
        </div>
      </section>

      <!-- 3回目編集後 -->
      <section v-if="currentEditCount >= maxEdits" class="history-section completed">
        <h2>編集完了 ({{ maxEdits }}/{{ maxEdits }}回)</h2>
        <div class="completion-message">
          <p>最大編集回数に到達しました。新しい音声をアップロードして再度編集できます。</p>
        </div>

        <!-- 全ての音声を表示（編集フォームなし） -->
        <div class="history-list">
          <div
            v-for="edit in editHistory"
            :key="edit.step"
            :class="['history-item', edit.step === 0 ? 'original' : 'edited']"
          >
            <div class="history-header">
              <h3>{{ edit.step === 0 ? '元の音声' : `編集 ${edit.step}回目` }}</h3>
              <span v-if="edit.step > 0" class="edit-badge">
                {{ edit.sourceOnoma }} → {{ edit.targetOnoma }}
              </span>
            </div>
            <audio controls :src="edit.audioUrl" class="audio-player"></audio>
          </div>
        </div>

        <!-- 新しい音声アップロードボタン -->
        <button @click="triggerFileInput" class="reset-btn">
          🔄 新しい音声をアップロード
        </button>
      </section>

      <!-- エラー表示 -->
      <section v-if="errorMessage" class="error-section">
        <div class="error-card">
          <h3>⚠️ エラー</h3>
          <p>{{ errorMessage }}</p>
        </div>
      </section>
    </main>

    <footer class="footer">
      <p>Onoma2DSP System - Differential Onomatopoeia to Audio Processing</p>
    </footer>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'

// セッション管理
const sessionId = ref(null)
const editHistory = ref([])
const currentEditCount = ref(0)
const maxEdits = 3

// 既存の状態
const audioFile = ref(null)
const sourceOnoma = ref('')
const targetOnoma = ref('')
const amplificationFactor = ref(1.0)
const lambdaAtt = ref(0.7)
const isProcessing = ref(false)
const errorMessage = ref('')
const errors = ref({
  source: '',
  target: ''
})

const fileInput = ref(null)

// カタカナ検証
const katakanaRegex = /^[ァ-ヶー]+$/

const validateKatakana = (field) => {
  const value = field === 'source' ? sourceOnoma.value : targetOnoma.value
  if (value && !katakanaRegex.test(value)) {
    errors.value[field] = 'カタカナのみ入力してください'
  } else {
    errors.value[field] = ''
  }
}

// 処理可能かチェック
const canProcess = computed(() => {
  // 初回編集
  return (
    currentEditCount.value === 0 &&
    sessionId.value !== null &&
    sourceOnoma.value &&
    targetOnoma.value &&
    !errors.value.source &&
    !errors.value.target
  )
})

const canContinueEdit = computed(() => {
  // 2回目以降の編集
  return (
    currentEditCount.value > 0 &&
    currentEditCount.value < maxEdits &&
    sessionId.value !== null &&
    sourceOnoma.value &&
    targetOnoma.value &&
    !errors.value.source &&
    !errors.value.target
  )
})

// ファイル選択
const triggerFileInput = () => {
  fileInput.value?.click()
}

const handleFileChange = async (event) => {
  const file = event.target.files?.[0]
  if (!file) return

  // 既存セッションがあれば確認
  if (sessionId.value && currentEditCount.value > 0) {
    const confirmed = confirm(
      `現在の編集履歴 (${currentEditCount.value}回) が削除されます。\n` +
      '新しい音声をアップロードしますか?'
    )
    if (!confirmed) {
      if (fileInput.value) fileInput.value.value = ''
      return
    }
  }

  try {
    isProcessing.value = true
    errorMessage.value = ''

    // 古いセッションを削除
    if (sessionId.value) {
      await resetSession()
    }

    // セッション作成API呼び出し
    const formData = new FormData()
    formData.append('audio_file', file)

    console.log('[Frontend] Creating new session...')

    const response = await fetch('http://localhost:8000/api/session/create', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to create session')
    }

    const data = await response.json()

    // セッション情報を保存
    sessionId.value = data.session_id
    audioFile.value = file

    // 履歴を初期化（元音声のみ）
    editHistory.value = [{
      step: 0,
      audioUrl: `http://localhost:8000${data.audio_url}`,
      sourceOnoma: null,
      targetOnoma: null,
      timestamp: new Date().toISOString()
    }]

    currentEditCount.value = 0

    console.log('[Frontend] Session created:', data.session_id)

  } catch (error) {
    console.error('[Frontend] Error creating session:', error)

    if (error.message === 'Failed to fetch' || error.name === 'TypeError') {
      errorMessage.value = 'バックエンドAPIサーバーに接続できません。\n\n' +
                          '【確認事項】\n' +
                          '1. APIサーバーが起動していますか？\n' +
                          '   → python api_server.py を実行してください\n' +
                          '2. http://localhost:8000 にアクセスできますか？'
    } else {
      errorMessage.value = 'セッション作成中にエラーが発生しました:\n' + error.message
    }
  } finally {
    isProcessing.value = false
  }
}

const clearFile = async () => {
  if (sessionId.value) {
    const confirmed = confirm(
      `現在の編集履歴 (${currentEditCount.value}回) が削除されます。\n` +
      'ファイルをクリアしますか?'
    )
    if (!confirmed) return
  }

  await resetSession()
  audioFile.value = null
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

// 音声処理
const processAudio = async () => {
  if (!canProcess.value && !canContinueEdit.value) return
  if (!sessionId.value) {
    errorMessage.value = 'セッションが初期化されていません'
    return
  }

  isProcessing.value = true
  errorMessage.value = ''

  try {
    const formData = new FormData()
    formData.append('source_onomatopoeia', sourceOnoma.value)
    formData.append('target_onomatopoeia', targetOnoma.value)
    formData.append('amplification_factor', amplificationFactor.value.toString())
    formData.append('lambda_att', lambdaAtt.value.toString())

    console.log('[Frontend] Editing session:', sessionId.value)
    console.log('  Source:', sourceOnoma.value)
    console.log('  Target:', targetOnoma.value)

    const response = await fetch(
      `http://localhost:8000/api/session/${sessionId.value}/edit`,
      {
        method: 'POST',
        body: formData
      }
    )

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'API request failed')
    }

    const data = await response.json()

    // 履歴を更新
    editHistory.value = data.history.map(edit => ({
      step: edit.step,
      audioUrl: `http://localhost:8000${edit.audio_url}`,
      sourceOnoma: edit.source_onoma,
      targetOnoma: edit.target_onoma,
      timestamp: edit.timestamp,
      processingTime: edit.processing_time
    }))

    currentEditCount.value = data.current_step

    console.log('[Frontend] Edit completed. Current step:', currentEditCount.value)

    // 入力フィールドをクリア
    sourceOnoma.value = ''
    targetOnoma.value = ''

  } catch (error) {
    console.error('[Frontend] Error processing audio:', error)

    if (error.message === 'Failed to fetch' || error.name === 'TypeError') {
      errorMessage.value = 'バックエンドAPIサーバーに接続できません。\n\n' +
                          '【確認事項】\n' +
                          '1. APIサーバーが起動していますか？\n' +
                          '2. http://localhost:8000 にアクセスできますか？'
    } else {
      errorMessage.value = '音声処理中にエラーが発生しました:\n' + error.message
    }
  } finally {
    isProcessing.value = false
  }
}

// リセット処理
const resetSession = async () => {
  if (sessionId.value) {
    try {
      await fetch(`http://localhost:8000/api/session/${sessionId.value}`, {
        method: 'DELETE'
      })
      console.log('[Frontend] Session deleted:', sessionId.value)
    } catch (error) {
      console.error('[Frontend] Error deleting session:', error)
    }
  }

  sessionId.value = null
  editHistory.value = []
  currentEditCount.value = 0
  audioFile.value = null
  sourceOnoma.value = ''
  targetOnoma.value = ''
  errorMessage.value = ''
}

// ページリロード時のリセット
onMounted(() => {
  console.log('[Frontend] Page mounted, resetting session')
  resetSession()
})

// ページ離脱時のクリーンアップ
onBeforeUnmount(() => {
  console.log('[Frontend] Page unmounting, cleaning up session')
  resetSession()
})
</script>

<style scoped>
.container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #333;
}

.header {
  text-align: center;
  padding: 2rem 1rem;
  color: white;
}

.title {
  font-size: 3rem;
  margin: 0;
  font-weight: bold;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.subtitle {
  font-size: 1.2rem;
  margin: 0.5rem 0 0;
  opacity: 0.9;
}

.main-content {
  flex: 1;
  max-width: 900px;
  width: 100%;
  margin: 0 auto;
  padding: 2rem 1rem;
}

.info-section {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.info-section h2 {
  margin-top: 0;
  color: #667eea;
}

.info-section ol {
  margin: 1rem 0;
  padding-left: 1.5rem;
}

.info-section li {
  margin: 0.5rem 0;
}

.form-section {
  margin-bottom: 2rem;
}

.form-card {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.form-group {
  margin-bottom: 1.5rem;
}

.label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: bold;
  margin-bottom: 0.5rem;
  color: #555;
}

.label-icon {
  font-size: 1.2rem;
}

.input {
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 1rem;
  transition: border-color 0.3s;
}

.input:focus {
  outline: none;
  border-color: #667eea;
}

.hint {
  font-size: 0.85rem;
  color: #888;
  margin: 0.25rem 0 0;
}

.hint-text {
  font-size: 0.9rem;
  color: #666;
  margin-bottom: 1rem;
}

.error-message {
  color: #e74c3c;
  font-size: 0.85rem;
  margin: 0.25rem 0 0;
}

.file-upload-area {
  border: 2px dashed #667eea;
  border-radius: 8px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s;
  background: #f8f9ff;
}

.file-upload-area:hover {
  background: #e8eaff;
  border-color: #5568d3;
}

.upload-placeholder p {
  margin: 0.5rem 0;
}

.upload-hint {
  font-size: 0.85rem;
  color: #888;
}

.upload-success {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
}

.upload-success p {
  margin: 0;
  color: #27ae60;
  font-weight: bold;
}

.clear-btn {
  background: #e74c3c;
  color: white;
  border: none;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  cursor: pointer;
  font-size: 1.2rem;
  line-height: 1;
}

.clear-btn:hover {
  background: #c0392b;
}

.uploaded-audio-player {
  margin: 1.5rem 0;
  padding: 1rem;
  background: #f8f9ff;
  border-radius: 8px;
}

.onomatopoeia-inputs {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 1rem;
  align-items: start;
}

.arrow {
  font-size: 2rem;
  color: #667eea;
  padding-top: 2rem;
}

.advanced-settings {
  margin: 1.5rem 0;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.advanced-settings summary {
  cursor: pointer;
  font-weight: bold;
  color: #667eea;
}

.settings-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-top: 1rem;
}

.slider {
  width: 100%;
}

.value-display {
  display: inline-block;
  margin-left: 0.5rem;
  font-weight: bold;
  color: #667eea;
}

.process-btn {
  width: 100%;
  padding: 1rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: bold;
  cursor: pointer;
  transition: transform 0.2s;
}

.process-btn:hover:not(:disabled) {
  transform: translateY(-2px);
}

.process-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 履歴セクション */
.history-section {
  margin-bottom: 2rem;
}

.history-section h2 {
  color: white;
  margin-bottom: 1rem;
  text-align: center;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.completion-message {
  background: #e8f5e9;
  border-left: 4px solid #4caf50;
  padding: 1rem;
  margin-bottom: 1.5rem;
  border-radius: 8px;
}

/* 履歴リスト */
.history-list {
  display: grid;
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.history-item {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.history-item.original {
  border: 2px solid #667eea;
  background: linear-gradient(135deg, #f8f9ff 0%, #e8eaff 100%);
}

.history-item.edited {
  border-left: 4px solid #27ae60;
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.history-header h3 {
  margin: 0;
  color: #667eea;
}

.edit-badge {
  background: #667eea;
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.85rem;
}

.audio-player {
  width: 100%;
}

/* 次の編集フォーム */
.next-edit-form {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border: 2px dashed #667eea;
}

.next-edit-form h3 {
  margin-top: 0;
  color: #667eea;
}

.edit-btn {
  background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
}

.reset-btn {
  width: 100%;
  padding: 1rem;
  background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: bold;
  cursor: pointer;
  transition: transform 0.2s;
}

.reset-btn:hover {
  transform: translateY(-2px);
}

.error-section {
  margin-bottom: 2rem;
}

.error-card {
  background: #fee;
  border-left: 4px solid #e74c3c;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.error-card h3 {
  margin-top: 0;
  color: #e74c3c;
}

.error-card p {
  white-space: pre-line;
  line-height: 1.6;
}

.footer {
  text-align: center;
  padding: 1.5rem;
  color: white;
  opacity: 0.9;
}

@media (max-width: 768px) {
  .onomatopoeia-inputs {
    grid-template-columns: 1fr;
  }

  .arrow {
    transform: rotate(90deg);
    padding: 0;
  }

  .settings-grid {
    grid-template-columns: 1fr;
  }
}
</style>
