import React, { useState, useEffect, useCallback, useRef } from 'react'
import { Graph } from '@antv/g6'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from './api/client'
import './styles.css'

// Types
interface GraphNode {
  id: string
  label?: string
  type?: string
  [key: string]: unknown
}

interface GraphEdge {
  id: string
  source: string
  target: string
  label?: string
  type?: string
  [key: string]: unknown
}

interface DocDetail {
  title: string
  text: string
}

interface SelectedElement {
  type: 'node' | 'edge'
  id: string
  label?: string
  nodeType?: string
  source?: string
  target?: string
  docDetail?: DocDetail
  data: unknown
}

interface GraphData {
  elements: {
    nodes: Array<{ data: GraphNode }>
    edges: Array<{ data: GraphEdge }>
  }
}

interface IngestResponse {
  job_id: string
  status: string
  total_items: number
  processed_items: number
  extracted_entities: number
  extracted_relations: number
  errors: string[]
}

interface PendingJob {
  fileName: string
  uploadProgress: number
  jobId?: string
  backendStatus?: 'processing' | 'completed' | 'failed'
  backendProgress?: { total: number; processed: number; errors: string[] }
  timestamp: number
}

interface QueryLogInfo {
  query: string
  parameters: Record<string, unknown>
  duration_ms: number
  result_count: number
}

interface AskResponse {
  answer: string
  cypher_query: string | null
  sources: Array<{
    type: string
    name: string
    node_type?: string
    relation_type?: string
    evidence?: string
  }>
  total_nodes: number
  total_edges: number
  query_logs: QueryLogInfo[]
}

// Node color mapping - Dark theme vibrant colors
const NODE_COLORS: Record<string, string> = {
  NewsItem: '#FF9500',
  Organization: '#30D158',
  Person: '#64D2FF',
  Policy: '#BF5AF2',
  Project: '#FF2D55',
  ThemeTag: '#FFD60A',
  ProvinceTag: '#FF375F',
  CityTag: '#0A84FF',
  Location: '#A2845E',
  Event: '#FF6B35',
  Technology: '#32ADE6',
  Category: '#5856D6',
  IndustryTag: '#AF52DE',
  Time: '#8E8E93',
  URL: '#636366',
  Entity: '#48484A',
}

const PENDING_JOB_KEY = 'wiki_pending_job'

function loadPendingJob(): PendingJob | null {
  try {
    const raw = localStorage.getItem(PENDING_JOB_KEY)
    if (!raw) return null
    const job = JSON.parse(raw) as PendingJob
    // 超过 24 小时的旧记录视为过期
    if (Date.now() - job.timestamp > 24 * 60 * 60 * 1000) {
      localStorage.removeItem(PENDING_JOB_KEY)
      return null
    }
    return job
  } catch {
    return null
  }
}

function savePendingJob(job: PendingJob | null) {
  try {
    if (job) {
      localStorage.setItem(PENDING_JOB_KEY, JSON.stringify(job))
    } else {
      localStorage.removeItem(PENDING_JOB_KEY)
    }
  } catch {
    // ignore
  }
}

function getNodeColor(type: string | undefined): string {
  if (!type) return '#8E8E93'
  if (NODE_COLORS[type]) return NODE_COLORS[type]
  // 为未知类型生成一个稳定颜色（基于字符串哈希）
  const palette = [
    '#FF9500', '#30D158', '#64D2FF', '#BF5AF2', '#FF2D55',
    '#FFD60A', '#FF375F', '#0A84FF', '#A2845E', '#FF6B35',
    '#32ADE6', '#5856D6', '#AF52DE', '#8E8E93', '#636366',
    '#FF9F0A', '#34C759', '#5AC8FA', '#AF52DE', '#FF3B30',
  ]
  let hash = 0
  for (let i = 0; i < type.length; i++) {
    hash = type.charCodeAt(i) + ((hash << 5) - hash)
  }
  return palette[Math.abs(hash) % palette.length]
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'upload' | 'graph' | 'chat'>('upload')
  const [isLoading, setIsLoading] = useState(false)

  // Upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadResult, setUploadResult] = useState<IngestResponse | null>(null)
  const [uploadMode, setUploadMode] = useState<'incremental' | 'overwrite'>('incremental')
  const [uploadProgress, setUploadProgress] = useState<number>(0)
  const [pendingJob, setPendingJob] = useState<PendingJob | null>(null)
  const pollTimerRef = useRef<number | null>(null)

  // Graph state
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [selectedElement, setSelectedElement] = useState<SelectedElement | null>(null)
  const [themeTags, setThemeTags] = useState<Array<{ theme: string; news_count: number }>>([])
  const [selectedTheme, setSelectedTheme] = useState<string>('')
  const [selectedRelTypes, setSelectedRelTypes] = useState<string[]>([])
  const [availableRelTypes, setAvailableRelTypes] = useState<string[]>([])
  const [nodeLimit, setNodeLimit] = useState<number>(25)
  const graphRef = useRef<Graph | null>(null)
  const graphContainerRef = useRef<HTMLDivElement>(null)
  const sliderTimerRef = useRef<number | null>(null)

  // Chat state
  const [question, setQuestion] = useState('')
  const [chatHistory, setChatHistory] = useState<Array<{
    role: 'user' | 'assistant'
    content: string
    sources?: AskResponse['sources']
    query_logs?: QueryLogInfo[]
    cypher_query?: string | null
  }>>([])

  // Load graph on mount
  useEffect(() => {
    loadGraph()
  }, [])

  // Restore pending job from localStorage on mount
  useEffect(() => {
    const job = loadPendingJob()
    if (job) {
      setPendingJob(job)
    }
  }, [])

  // Poll backend job status when there is a pending job with jobId
  useEffect(() => {
    if (!pendingJob?.jobId) return

    const poll = async () => {
      try {
        const { data } = await api.get<{
          job_id: string
          status: 'processing' | 'completed' | 'failed'
          progress: { total: number; processed: number; errors: string[] }
        }>(`/api/v1/json/jobs/${pendingJob.jobId}`)

        const updated: PendingJob = {
          ...pendingJob,
          backendStatus: data.status,
          backendProgress: data.progress,
          timestamp: Date.now(),
        }
        setPendingJob(updated)
        savePendingJob(updated)

        if (data.status === 'completed' || data.status === 'failed') {
          // Job finished: load graph and clear after a short delay
          await loadGraph()
          if (pollTimerRef.current) {
            window.clearInterval(pollTimerRef.current)
            pollTimerRef.current = null
          }
          setTimeout(() => {
            setPendingJob(null)
            savePendingJob(null)
          }, 3000)
        }
      } catch (e) {
        console.error('Poll job status failed:', e)
      }
    }

    poll()
    pollTimerRef.current = window.setInterval(poll, 2000)

    return () => {
      if (pollTimerRef.current) {
        window.clearInterval(pollTimerRef.current)
        pollTimerRef.current = null
      }
    }
  }, [pendingJob?.jobId])

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (sliderTimerRef.current) {
        window.clearTimeout(sliderTimerRef.current)
      }
      if (pollTimerRef.current) {
        window.clearInterval(pollTimerRef.current)
      }
    }
  }, [])

  // Initialize G6 graph when switching to graph tab
  useEffect(() => {
    if (activeTab === 'graph' && graphContainerRef.current && graphData && !graphRef.current) {
      const nodes = graphData.elements.nodes.map(n => ({
        id: n.data.id,
        label: `${n.data.label || n.data.id} (${n.data.type || 'Entity'})`,
        data: {
          ...n.data,
          nodeType: n.data.type,
        },
      }))

      const edges = graphData.elements.edges.map(e => {
        const isCorrelation = e.data.type === 'CORRELATED_WITH'
        const isRel = e.data.type === 'REL'
        const labelText = isCorrelation ? `相似度: ${(Number(e.data.score) || 0).toFixed(2)}` : (e.data.label || e.data.type || '')

        return {
          id: e.data.id,
          source: e.data.source,
          target: e.data.target,
          label: labelText,
          data: {
            ...e.data,
          },
          // 根据边类型设置不同样式
          style: isCorrelation ? {
            stroke: '#FF375F',      // 红色 - 相似度
            lineWidth: 3,           // 更粗
            lineDash: [8, 4],       // 虚线更明显
            endArrow: false,        // 双向无边箭头
            labelText,
            labelFill: '#FF375F',
            labelFontSize: 11,
            labelFontWeight: 'bold' as const,
            labelBackgroundFill: 'rgba(28,28,30,0.9)',
            labelBackgroundRadius: 4,
            halo: true,             // 发光效果
            haloStroke: '#FF375F',
            haloLineWidth: 4,
            haloOpacity: 0.3,
          } : isRel ? {
            stroke: 'rgba(255,255,255,0.25)',  // 白色半透明 - OneKE关系
            lineWidth: 1,
            endArrow: true,
            endArrowSize: 6,
            labelText,
            labelFill: 'rgba(255,255,255,0.5)',
            labelFontSize: 9,
            labelBackgroundFill: 'rgba(28,28,30,0.8)',
            labelBackgroundRadius: 4,
          } : {
            stroke: 'rgba(255,255,255,0.3)',
            lineWidth: 1,
            endArrow: true,
            labelText,
            labelFill: 'rgba(255,255,255,0.6)',
            labelFontSize: 9,
            labelBackgroundFill: 'rgba(28,28,30,0.8)',
            labelBackgroundRadius: 4,
          }
        }
      })

      const graph = new Graph({
        container: graphContainerRef.current,
        data: { nodes, edges },
        node: {
          style: {
            size: 36,
            fill: function(this: Graph, d: { data?: { nodeType?: string } }) {
              return getNodeColor(d.data?.nodeType)
            },
            stroke: '#1C1C1E',
            lineWidth: 2,
            labelText: function(this: Graph, d: { label?: string; id?: string }) {
              return d.label || d.id || ''
            } as unknown as string,
            labelFill: '#FFFFFF',
            labelFontSize: 9,
            labelFontWeight: 500,
            labelOffsetY: 10,
            labelWordWrap: true,
            labelMaxWidth: 120,
            cursor: 'pointer',
          },
          state: {
            hover: {
              stroke: '#0A84FF',
              lineWidth: 3,
              shadowColor: 'rgba(10,132,255,0.5)',
              shadowBlur: 10,
            },
            selected: {
              stroke: '#FFD60A',
              lineWidth: 3,
              shadowColor: 'rgba(255,214,10,0.5)',
              shadowBlur: 15,
            },
          },
        },
        edge: {
          style: {
            stroke: 'rgba(255,255,255,0.3)',
            lineWidth: 1,
            endArrow: true,
            endArrowSize: 6,
            labelText: function(this: Graph, d: { label?: string }) { return d.label || '' } as unknown as string,
            labelFill: 'rgba(255,255,255,0.6)',
            labelFontSize: 9,
            labelBackgroundFill: 'rgba(28,28,30,0.8)',
            labelBackgroundRadius: 4,
          },
        },
        layout: {
          type: 'force',
          preventOverlap: true,
          nodeSpacing: 25,
          linkDistance: 80,
          maxIteration: 300,
          animated: false,
        },
        behaviors: [
          { type: 'drag-canvas' },
          { type: 'zoom-canvas', sensitivity: 1.2 },
          { type: 'drag-element' },
        ],
      })

      graph.render().then(() => {
        console.log('Graph rendered successfully')
      }).catch((err: Error) => {
        console.error('Graph render failed:', err)
      })

      // Handle previous selection clearing when new node is clicked
      const handleNodeSelect = (nodeId: string) => {
        if (selectedElement && selectedElement.id !== nodeId) {
          try {
            graph.setElementState(selectedElement.id, [])
          } catch (e) {
            // Ignore errors if element doesn't exist
          }
        }
      }

      // Handle node click - G6 v5 event structure
      graph.on('node:click', async (e: unknown) => {
        const event = e as { itemId?: string; target?: { id?: string } }
        const nodeId = event.itemId || event.target?.id
        if (!nodeId) return

        console.log('Node clicked:', nodeId)

        const nodeData = graph.getNodeData(nodeId) as { label?: string; data?: { nodeType?: string }; [key: string]: unknown } | undefined
        if (!nodeData) return

        handleNodeSelect(nodeId)
        graph.setElementState(nodeId, 'selected')

        const element: SelectedElement = {
          type: 'node',
          id: nodeId,
          label: nodeData.label || nodeId,
          nodeType: nodeData.data?.nodeType || 'Unknown',
          data: nodeData
        }
        setSelectedElement(element)

        // Load full text for NewsItem nodes
        if (nodeData.data?.nodeType === 'NewsItem') {
          try {
            const { data: docData } = await api.get<DocDetail>(`/api/v1/docs/${nodeId}`)
            setSelectedElement(prev => prev && prev.id === nodeId && prev.type === 'node' ? { ...prev, docDetail: docData } : prev)
          } catch (err) {
            console.error('Failed to load doc detail:', err)
          }
        }
      })

      // Handle edge click
      graph.on('edge:click', (e: unknown) => {
        const event = e as { itemId?: string; target?: { id?: string } }
        const edgeId = event.itemId || event.target?.id
        if (!edgeId) return

        console.log('Edge clicked:', edgeId)

        const edgeData = graph.getEdgeData(edgeId) as { label?: string; type?: string; source?: string; target?: string; [key: string]: unknown } | undefined
        if (!edgeData) return

        setSelectedElement({
          type: 'edge',
          id: edgeId,
          label: edgeData.label || edgeData.type || '关联',
          source: edgeData.source,
          target: edgeData.target,
          data: edgeData
        })
      })

      // Click on empty canvas to deselect
      graph.on('canvas:click', (e: unknown) => {
        const event = e as { stopPropagation?: () => void }
        event.stopPropagation?.()
        if (selectedElement) {
          try {
            graph.setElementState(selectedElement.id, [])
          } catch (e) {
            // Ignore errors
          }
        }
        setSelectedElement(null)
      })

      graphRef.current = graph
    }

    return () => {
      if (graphRef.current) {
        graphRef.current.destroy()
        graphRef.current = null
      }
    }
  }, [activeTab, graphData])

  const loadGraph = useCallback(async () => {
    try {
      console.log('Loading graph...')
      const { data } = await api.get<GraphData>('/api/v1/graph', {
        params: { node_limit: nodeLimit }
      })
      console.log('Graph data loaded:', data.elements.nodes.length, 'nodes,', data.elements.edges.length, 'edges')
      // 重置关系筛选和选中状态，强制 graph useEffect 重新初始化
      setSelectedRelTypes([])
      setSelectedElement(null)
      if (graphRef.current) {
        graphRef.current.destroy()
        graphRef.current = null
      }
      setGraphData(data)
    } catch (e) {
      console.error('Failed to load graph:', e)
    }
  }, [nodeLimit])

  // Load theme tags
  const loadThemeTags = useCallback(async () => {
    try {
      const { data } = await api.get<{ themes: Array<{ theme: string; news_count: number }>; total: number }>('/api/v1/graph/themes')
      setThemeTags(data.themes)
    } catch (e) {
      console.error('Failed to load theme tags:', e)
    }
  }, [])

  // Load graph by theme
  const loadGraphByTheme = useCallback(async (theme: string) => {
    try {
      setIsLoading(true)
      const { data } = await api.get<GraphData>(`/api/v1/graph/theme/${encodeURIComponent(theme)}`, {
        params: { news_limit: nodeLimit }
      })
      console.log('Theme graph loaded:', data.elements.nodes.length, 'nodes,', data.elements.edges.length, 'edges')
      // 重置关系筛选和选中状态，强制 graph useEffect 重新初始化
      setSelectedRelTypes([])
      setSelectedElement(null)
      if (graphRef.current) {
        graphRef.current.destroy()
        graphRef.current = null
      }
      setGraphData(data)
    } catch (e) {
      console.error('Failed to load graph by theme:', e)
    } finally {
      setIsLoading(false)
    }
  }, [nodeLimit])

  const buildCorrelationEdges = useCallback(async () => {
    try {
      setIsLoading(true)
      const { data } = await api.post<{ created_edges: number; message: string }>(
        '/api/v1/correlations/build-edges?min_score=0.05&use_vector=true'
      )
      alert(`相似度关联构建完成：${data.message}`)
      if (selectedTheme) {
        await loadGraphByTheme(selectedTheme)
      } else {
        await loadGraph()
      }
    } catch (e: any) {
      console.error('Build correlation edges failed:', e)
      alert('构建相似度关联失败：' + (e.response?.data?.detail || e.message))
    } finally {
      setIsLoading(false)
    }
  }, [loadGraph, loadGraphByTheme, selectedTheme])

  // Load theme tags when graph tab is active
  useEffect(() => {
    if (activeTab === 'graph') {
      loadThemeTags()
    }
  }, [activeTab, loadThemeTags])

  // 当图谱数据变化时，更新可用关系类型并默认全选
  useEffect(() => {
    if (!graphData) return
    const relTypes = Array.from(new Set(graphData.elements.edges.map(e => e.data.type || 'REL'))).sort()
    setAvailableRelTypes(relTypes)
    setSelectedRelTypes(prev => {
      if (prev.length === 0) return relTypes
      const filtered = prev.filter(t => relTypes.includes(t))
      return filtered.length > 0 ? filtered : relTypes
    })
  }, [graphData])

  // 当关系筛选变化时更新图谱显示（仅更新 visibility，不触发重新布局）
  useEffect(() => {
    const graph = graphRef.current
    if (!graph || !graphData) return

    try {
      const updates = graphData.elements.edges.map((e) => {
        const visible = selectedRelTypes.includes(e.data.type || 'REL')
        return {
          id: e.data.id,
          style: {
            visibility: visible ? ('visible' as const) : ('hidden' as const),
          },
        }
      })
      graph.updateEdgeData(updates)
      graph.draw()
    } catch (e) {
      console.error('Failed to filter edges:', e)
    }
  }, [selectedRelTypes, graphData])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.name.endsWith('.json')) {
      setSelectedFile(file)
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    setIsLoading(true)
    setUploadProgress(0)
    const initialJob: PendingJob = {
      fileName: selectedFile.name,
      uploadProgress: 0,
      timestamp: Date.now(),
    }
    setPendingJob(initialJob)
    savePendingJob(initialJob)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('schema_name', 'MOE_News')
      formData.append('mode', uploadMode)

      const { data } = await api.post<IngestResponse>('/api/v1/json/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total)
            setUploadProgress(percent)
            const updated: PendingJob = {
              fileName: selectedFile.name,
              uploadProgress: percent,
              timestamp: Date.now(),
            }
            setPendingJob(updated)
            savePendingJob(updated)
          }
        }
      })

      setUploadResult(data)
      const withJobId: PendingJob = {
        fileName: selectedFile.name,
        uploadProgress: 100,
        jobId: data.job_id,
        backendStatus: data.status as any,
        backendProgress: {
          total: data.total_items,
          processed: data.processed_items,
          errors: data.errors,
        },
        timestamp: Date.now(),
      }
      setPendingJob(withJobId)
      savePendingJob(withJobId)

      if (data.status === 'completed' && data.processed_items > 0) {
        await loadGraph()
        setPendingJob(null)
        savePendingJob(null)
      }
    } catch (e) {
      console.error('Upload failed:', e)
      alert('上传失败，请检查服务是否正常运行')
      setPendingJob(null)
      savePendingJob(null)
    } finally {
      setIsLoading(false)
      setUploadProgress(0)
    }
  }

  const handleAsk = async () => {
    if (!question.trim()) return

    const q = question.trim()
    setQuestion('')
    setChatHistory(prev => [...prev, { role: 'user', content: q }])
    setIsLoading(true)

    try {
      const { data } = await api.post<AskResponse>('/api/v1/qa/ask-graph', {
        question: q,
        top_k: 10
      })

      setChatHistory(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        query_logs: data.query_logs,
        cypher_query: data.cypher_query
      }])
    } catch (e) {
      console.error('Ask failed:', e)
      setChatHistory(prev => [...prev, {
        role: 'assistant',
        content: '抱歉，查询失败。请确保图谱数据已加载。'
      }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">K</div>
          <div className="brand-text">
            <h1>Knowledge Graph</h1>
            <p>OneKE + Neo4j + LLM</p>
          </div>
        </div>

        <nav className="nav-tabs">
          {[
            { id: 'upload', label: '数据上传', icon: '↑' },
            { id: 'graph', label: '知识图谱', icon: '◉' },
            { id: 'chat', label: '智能问答', icon: '◆' }
          ].map(tab => (
            <button
              key={tab.id}
              className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id as typeof activeTab)}
            >
              <span>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      {/* Main Content */}
      <main className="app-main">
        <div className="content-wrapper">
          {/* Upload Panel */}
          {activeTab === 'upload' && (
            <div className="card" style={{ maxWidth: 600, margin: '0 auto' }}>
              <div className="card-header">
                <h2 className="card-title">上传 JSON 数据</h2>
                <p className="card-subtitle">上传新闻 JSON 文件，自动构建知识图谱</p>
              </div>
              <div className="card-body">
                {pendingJob && (
                  <div style={{
                    marginBottom: 20,
                    padding: 14,
                    background: pendingJob.backendStatus === 'failed'
                      ? 'rgba(255,55,95,0.12)'
                      : pendingJob.backendStatus === 'completed'
                        ? 'rgba(48,209,88,0.12)'
                        : 'rgba(10,132,255,0.12)',
                    borderRadius: 10,
                    border: `1px solid ${pendingJob.backendStatus === 'failed'
                      ? 'rgba(255,55,95,0.35)'
                      : pendingJob.backendStatus === 'completed'
                        ? 'rgba(48,209,88,0.35)'
                        : 'rgba(10,132,255,0.35)'}`,
                  }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
                      {pendingJob.backendStatus === 'failed' && '❌ 处理失败'}
                      {pendingJob.backendStatus === 'completed' && '✅ 处理完成'}
                      {!pendingJob.backendStatus && '⏳ 准备上传'}
                      {pendingJob.backendStatus === 'processing' && '⚙️ 后端处理中'}
                      {!pendingJob.jobId && pendingJob.uploadProgress < 100 && '⏫ 正在上传'}
                      <span style={{ marginLeft: 8, fontWeight: 500, color: 'var(--text-secondary)' }}>
                        {pendingJob.fileName}
                      </span>
                    </div>

                    {(pendingJob.uploadProgress < 100 || !pendingJob.jobId) ? (
                      <>
                        <div style={{
                          height: 6,
                          background: 'rgba(255,255,255,0.1)',
                          borderRadius: 3,
                          overflow: 'hidden',
                        }}>
                          <div style={{
                            width: `${pendingJob.uploadProgress}%`,
                            height: '100%',
                            background: 'linear-gradient(90deg, #0A84FF, #64D2FF)',
                            borderRadius: 3,
                            transition: 'width 0.2s ease',
                          }} />
                        </div>
                        <div style={{ fontSize: 12, color: 'var(--text-tertiary)', marginTop: 6 }}>
                          上传进度: {pendingJob.uploadProgress}%
                        </div>
                      </>
                    ) : pendingJob.backendStatus === 'processing' && pendingJob.backendProgress ? (
                      <>
                        <div style={{
                          height: 6,
                          background: 'rgba(255,255,255,0.1)',
                          borderRadius: 3,
                          overflow: 'hidden',
                        }}>
                          <div style={{
                            width: `${pendingJob.backendProgress.total
                              ? Math.min(100, Math.round((pendingJob.backendProgress.processed * 100) / pendingJob.backendProgress.total))
                              : 0}%`,
                            height: '100%',
                            background: 'linear-gradient(90deg, #30D158, #64D2FF)',
                            borderRadius: 3,
                            transition: 'width 0.3s ease',
                          }} />
                        </div>
                        <div style={{ fontSize: 12, color: 'var(--text-tertiary)', marginTop: 6 }}>
                          后端处理进度: {pendingJob.backendProgress.processed} / {pendingJob.backendProgress.total} 条
                          {pendingJob.backendProgress.errors.length > 0 && (
                            <span style={{ color: '#FF375F', marginLeft: 8 }}>
                              ({pendingJob.backendProgress.errors.length} 个错误)
                            </span>
                          )}
                        </div>
                      </>
                    ) : null}

                    {pendingJob.backendStatus === 'failed' && (
                      <button
                        className="btn btn-secondary"
                        style={{ marginTop: 10, fontSize: 12, padding: '6px 12px' }}
                        onClick={() => {
                          setPendingJob(null)
                          savePendingJob(null)
                        }}
                      >
                        关闭提示
                      </button>
                    )}
                  </div>
                )}

                <input
                  type="file"
                  accept=".json"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                  id="file-input"
                />
                <label htmlFor="file-input">
                  <div className={`upload-zone ${selectedFile ? 'active' : ''}`}>
                    <div className="upload-icon">📄</div>
                    <p style={{ fontSize: 16, fontWeight: 500, marginBottom: 8 }}>
                      {selectedFile ? selectedFile.name : '点击选择 JSON 文件'}
                    </p>
                    <p style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
                      支持科策云等导出的新闻数据格式
                    </p>
                  </div>
                </label>

                {selectedFile && (
                  <>
                    <div style={{ marginTop: 24 }}>
                      <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 12 }}>
                        上传模式
                      </p>
                      <div className="mode-selector">
                        <button
                          type="button"
                          onClick={() => setUploadMode('incremental')}
                          className={`mode-option ${uploadMode === 'incremental' ? 'selected' : ''}`}
                        >
                          <div className="mode-label">
                            <span>➕</span>
                            增量上传
                          </div>
                          <div className="mode-desc">在现有数据基础上添加</div>
                        </button>
                        <button
                          type="button"
                          onClick={() => setUploadMode('overwrite')}
                          className={`mode-option danger ${uploadMode === 'overwrite' ? 'selected' : ''}`}
                        >
                          <div className="mode-label">
                            <span>🔄</span>
                            覆盖上传
                          </div>
                          <div className="mode-desc">清空旧数据后上传</div>
                        </button>
                      </div>
                      {uploadMode === 'overwrite' && (
                        <p style={{
                          marginTop: 12,
                          padding: 12,
                          background: 'rgba(255,55,95,0.1)',
                          borderRadius: 8,
                          fontSize: 13,
                          color: '#FF375F'
                        }}>
                          ⚠️ 警告：覆盖上传将删除所有现有的图谱数据
                        </p>
                      )}
                    </div>

                    <button
                      className="btn btn-block"
                      onClick={handleUpload}
                      disabled={isLoading}
                      style={{ marginTop: 24 }}
                    >
                      {isLoading ? (
                        <>
                          <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                          处理中...
                        </>
                      ) : (
                        <>
                          {uploadMode === 'overwrite' ? '🔄 覆盖上传' : '➕ 增量上传'}
                          <span style={{ marginLeft: 8, fontSize: 13, opacity: 0.8 }}>
                            ({selectedFile.name})
                          </span>
                        </>
                      )}
                    </button>

                    {isLoading && uploadProgress > 0 && (
                      <div style={{ marginTop: 16 }}>
                        <div style={{
                          height: 6,
                          background: 'rgba(255,255,255,0.1)',
                          borderRadius: 3,
                          overflow: 'hidden',
                        }}>
                          <div style={{
                            width: `${uploadProgress}%`,
                            height: '100%',
                            background: 'linear-gradient(90deg, #0A84FF, #64D2FF)',
                            borderRadius: 3,
                            transition: 'width 0.2s ease',
                          }} />
                        </div>
                        <p style={{
                          marginTop: 8,
                          fontSize: 12,
                          color: 'var(--text-secondary)',
                          textAlign: 'center',
                        }}>
                          上传进度: {uploadProgress}%
                        </p>
                      </div>
                    )}
                  </>
                )}

                {uploadResult && (
                  <div style={{
                    marginTop: 24,
                    padding: 20,
                    background: uploadResult.errors.length === 0 ? 'rgba(48,209,88,0.1)' : 'rgba(255,55,95,0.1)',
                    borderRadius: 12
                  }}>
                    <div className={`badge ${uploadResult.errors.length === 0 ? 'badge-success' : 'badge-error'}`}
                      style={{ marginBottom: 16 }}>
                      {uploadResult.errors.length === 0 ? '✓ 处理完成' : '✗ 处理失败'}
                    </div>
                    <div className="stats-grid">
                      <div className="stat-item">
                        <div className="stat-value">{uploadResult.processed_items}/{uploadResult.total_items}</div>
                        <div className="stat-label">处理条目</div>
                      </div>
                      <div className="stat-item">
                        <div className="stat-value">{uploadResult.extracted_entities}</div>
                        <div className="stat-label">实体数量</div>
                      </div>
                      <div className="stat-item">
                        <div className="stat-value">{uploadResult.extracted_relations}</div>
                        <div className="stat-label">关系数量</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Graph Panel */}
          {activeTab === 'graph' && (
            <div className="graph-layout">
              <div className="card graph-main" style={{ position: 'relative' }}>
                {graphData && graphData.elements.nodes.length > 0 ? (
                  <>
                    <div ref={graphContainerRef} style={{ width: '100%', height: '100%' }} />
                    <div style={{
                      position: 'absolute',
                      top: 12,
                      left: 12,
                      display: 'flex',
                      gap: 8,
                      pointerEvents: 'none',
                      flexWrap: 'wrap',
                    }}>
                      <span style={{
                        padding: '4px 10px',
                        background: 'rgba(5,10,20,0.8)',
                        border: '1px solid var(--border-color)',
                        borderRadius: 6,
                        fontSize: 11,
                        color: 'var(--text-tertiary)',
                      }}>
                        拖拽画布移动 · 滚轮缩放
                      </span>
                      <span style={{
                        padding: '4px 10px',
                        background: 'rgba(5,10,20,0.8)',
                        border: '1px solid var(--border-color)',
                        borderRadius: 6,
                        fontSize: 11,
                        color: 'var(--text-secondary)',
                      }}>
                        {graphData.elements.nodes.length} 节点 · {graphData.elements.edges.length} 关系
                      </span>
                    </div>
                  </>
                ) : (
                  <div className="empty-state">
                    <div className="empty-state-icon">◉</div>
                    <div className="empty-state-title">暂无图谱数据</div>
                    <div className="empty-state-desc">请先上传 JSON 文件构建图谱</div>
                    <button className="btn" onClick={() => setActiveTab('upload')} style={{ marginTop: 24 }}>
                      去上传数据
                    </button>
                  </div>
                )}
              </div>

              <div className="card graph-sidebar">
                <div className="card-header">
                  <h3 className="card-title">显示数量</h3>
                </div>
                <div className="card-body" style={{ paddingBottom: 12, borderBottom: '1px solid var(--border-color)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <input
                      type="range"
                      min={10}
                      max={200}
                      step={5}
                      value={nodeLimit}
                      onChange={(e) => {
                        const val = parseInt(e.target.value, 10)
                        setNodeLimit(val)
                        if (sliderTimerRef.current) {
                          window.clearTimeout(sliderTimerRef.current)
                        }
                        sliderTimerRef.current = window.setTimeout(() => {
                          if (selectedTheme) {
                            loadGraphByTheme(selectedTheme)
                          } else {
                            loadGraph()
                          }
                        }, 400)
                      }}
                      style={{ flex: 1 }}
                    />
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)', minWidth: 40, textAlign: 'right' }}>
                      {nodeLimit}
                    </span>
                  </div>
                  <p style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 8, marginBottom: 0 }}>
                    拖动后自动重新加载
                  </p>
                  <button
                    className="btn btn-secondary"
                    onClick={buildCorrelationEdges}
                    disabled={isLoading}
                    style={{ marginTop: 12, width: '100%', fontSize: 12 }}
                  >
                    {isLoading ? '构建中...' : '重建相似度关联'}
                  </button>
                </div>

                <div className="card-header" style={{ marginTop: 8 }}>
                  <h3 className="card-title">主题筛选</h3>
                </div>
                <div className="card-body" style={{ paddingBottom: 12, borderBottom: '1px solid var(--border-color)' }}>
                  <select
                    className="form-control"
                    value={selectedTheme}
                    onChange={(e) => {
                      const theme = e.target.value
                      setSelectedTheme(theme)
                      if (theme) {
                        loadGraphByTheme(theme)
                      } else {
                        loadGraph()
                      }
                    }}
                    style={{ width: '100%' }}
                  >
                    <option value="">全部主题</option>
                    {themeTags.map((tag) => (
                      <option key={tag.theme} value={tag.theme}>
                        {tag.theme} ({tag.news_count})
                      </option>
                    ))}
                  </select>
                  {selectedTheme && (
                    <button
                      className="btn btn-secondary"
                      onClick={() => {
                        setSelectedTheme('')
                        loadGraph()
                      }}
                      style={{ marginTop: 8, width: '100%' }}
                    >
                      清除筛选
                    </button>
                  )}
                </div>

                <div className="card-header" style={{ marginTop: 8 }}>
                  <h3 className="card-title">关系筛选</h3>
                </div>
                <div className="card-body" style={{ paddingBottom: 12, borderBottom: '1px solid var(--border-color)' }}>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, maxHeight: 120, overflowY: 'auto' }}>
                    {availableRelTypes.map((relType) => {
                      const isSelected = selectedRelTypes.includes(relType)
                      return (
                        <button
                          key={relType}
                          onClick={() => {
                            setSelectedRelTypes((prev) =>
                              isSelected ? prev.filter((t) => t !== relType) : [...prev, relType]
                            )
                          }}
                          style={{
                            padding: '4px 10px',
                            borderRadius: 9999,
                            border: '1px solid var(--border-color)',
                            background: isSelected ? 'var(--active-bg)' : 'transparent',
                            color: isSelected ? 'var(--neon-cyan)' : 'var(--text-secondary)',
                            fontSize: 12,
                            cursor: 'pointer',
                            transition: 'all 0.15s',
                          }}
                        >
                          {relType}
                        </button>
                      )
                    })}
                  </div>
                  <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                    <button
                      className="btn btn-secondary"
                      onClick={() => setSelectedRelTypes(availableRelTypes)}
                      style={{ flex: 1, fontSize: 12, padding: '6px 0' }}
                    >
                      全选
                    </button>
                    <button
                      className="btn btn-secondary"
                      onClick={() => setSelectedRelTypes([])}
                      style={{ flex: 1, fontSize: 12, padding: '6px 0' }}
                    >
                      清空
                    </button>
                  </div>
                </div>

                <div className="card-header" style={{ marginTop: 8 }}>
                  <h3 className="card-title">元素详情</h3>
                </div>
                <div className="card-body">
                  {selectedElement ? (
                    <div className="detail-content">
                      <div className={`detail-type ${selectedElement.type}`}>
                        <span style={{
                          width: 6,
                          height: 6,
                          borderRadius: '50%',
                          background: selectedElement.type === 'node' ? '#0A84FF' : '#30D158'
                        }} />
                        {selectedElement.type === 'node' ? '节点' : '关系'}
                      </div>

                      <div>
                        <div className="detail-section-title">
                          {selectedElement.type === 'node' ? (selectedElement.nodeType || 'Entity') : '关系类型'}
                        </div>
                        <div className="detail-title">{selectedElement.label || selectedElement.id}</div>
                      </div>

                      {selectedElement.type === 'edge' && selectedElement.source && selectedElement.target && (
                        <div className="detail-section">
                          <div className="detail-section-title">连接信息</div>
                          <div style={{ fontSize: 13, display: 'flex', flexDirection: 'column', gap: 8 }}>
                            <div style={{ display: 'flex', gap: 8 }}>
                              <span style={{ color: 'var(--text-tertiary)' }}>从:</span>
                              <span>{selectedElement.source}</span>
                            </div>
                            <div style={{ display: 'flex', gap: 8 }}>
                              <span style={{ color: 'var(--text-tertiary)' }}>到:</span>
                              <span>{selectedElement.target}</span>
                            </div>
                          </div>
                        </div>
                      )}

                      {selectedElement.type === 'node' && selectedElement.nodeType === 'NewsItem' && (
                        <div className="detail-section">
                          <div className="detail-section-title">新闻全文</div>
                          {selectedElement.docDetail ? (
                            <div style={{
                              fontSize: 13,
                              lineHeight: 1.7,
                              color: 'var(--text-secondary)',
                              maxHeight: 280,
                              overflowY: 'auto',
                              padding: 12,
                              background: 'rgba(255,255,255,0.03)',
                              borderRadius: 8,
                            }}>
                              {selectedElement.docDetail.text}
                            </div>
                          ) : (
                            <div style={{ fontSize: 13, color: 'var(--text-tertiary)' }}>加载中...</div>
                          )}
                        </div>
                      )}

                      <details>
                        <summary style={{ fontSize: 13, color: 'var(--accent-blue)', cursor: 'pointer' }}>
                          查看原始数据
                        </summary>
                        <pre style={{
                          marginTop: 12,
                          padding: 12,
                          background: 'var(--bg-primary)',
                          borderRadius: 8,
                          fontSize: 12,
                          overflow: 'auto',
                          fontFamily: 'var(--font-mono)'
                        }}>
                          {JSON.stringify(selectedElement.data, null, 2)}
                        </pre>
                      </details>
                    </div>
                  ) : (
                    <div className="detail-empty">
                      <div className="detail-empty-icon">◉</div>
                      <p>点击图谱中的节点或边<br />查看详细信息</p>
                    </div>
                  )}
                </div>

                {/* 图例说明 */}
                <div className="card-footer" style={{ padding: '16px', borderTop: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: 'var(--text-secondary)' }}>
                    图例说明
                  </div>

                  {/* OneKE 关系 */}
                  <div style={{ marginBottom: 12, padding: 10, background: 'rgba(255,255,255,0.03)', borderRadius: 8 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                      <div style={{
                        width: 30,
                        height: 2,
                        background: 'rgba(255,255,255,0.25)',
                        borderRadius: 1,
                      }} />
                      <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-secondary)' }}>
                        OneKE 实体关系
                      </span>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-tertiary)', paddingLeft: 40 }}>
                      从文档中提取的实体关系，如"发布"、"涉及"
                    </div>
                  </div>

                  {/* 相似度关联 */}
                  <div style={{ padding: 10, background: 'rgba(255,55,95,0.05)', borderRadius: 8 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                      <div style={{
                        width: 30,
                        height: 3,
                        background: 'repeating-linear-gradient(90deg, #FF375F 0px, #FF375F 6px, transparent 6px, transparent 10px)',
                        borderRadius: 1,
                      }} />
                      <span style={{ fontSize: 12, fontWeight: 500, color: '#FF375F' }}>
                        相似度关联
                      </span>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-tertiary)', paddingLeft: 40 }}>
                      基于实体+向量相似度计算，连接相关新闻
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Chat Panel */}
          {activeTab === 'chat' && (
            <div className="chat-layout">
              <div className="card chat-main">
                <div className="chat-messages">
                  {chatHistory.length === 0 && (
                    <div className="empty-state">
                      <div className="empty-state-icon">◆</div>
                      <div className="empty-state-title">基于知识图谱的智能问答</div>
                      <div className="empty-state-desc">询问关于政策、机构、项目等问题</div>
                    </div>
                  )}

                  {chatHistory.map((msg, idx) => (
                    <div
                      key={idx}
                      className={`message message-${msg.role}`}
                    >
                      {msg.role === 'assistant' ? (
                        <div className="markdown-body">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                        </div>
                      ) : (
                        msg.content
                      )}

                      {(msg.query_logs || msg.cypher_query) && (
                        <div className="query-log">
                          <details>
                            <summary>
                              🗄️ 查看 Neo4j 查询 ({msg.query_logs?.length || 0} 次查询)
                            </summary>
                            <div className="query-log-content">
                              {msg.cypher_query && (
                                <div style={{ marginBottom: 12 }}>
                                  <div className="detail-section-title">示例 Cypher 查询</div>
                                  <div className="query-item">
                                    <div className="query-code">{msg.cypher_query}</div>
                                  </div>
                                </div>
                              )}

                              {msg.query_logs && msg.query_logs.length > 0 && (
                                <div>
                                  <div className="detail-section-title">执行的查询</div>
                                  {msg.query_logs.map((log, i) => (
                                    <div key={i} className="query-item">
                                      <div className="query-code">{log.query}</div>
                                      <div className="query-meta">
                                        <span>⏱️ {log.duration_ms.toFixed(2)}ms</span>
                                        <span>📊 {log.result_count} 结果</span>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          </details>
                        </div>
                      )}

                      {msg.sources && msg.sources.length > 0 && (
                        <div style={{
                          marginTop: 12,
                          paddingTop: 12,
                          borderTop: '1px solid rgba(255,255,255,0.1)',
                          fontSize: 12,
                          opacity: 0.8
                        }}>
                          <p style={{ marginBottom: 4 }}>来源: {msg.sources.length} 个</p>
                          {msg.sources.slice(0, 3).map((s, i) => (
                            <p key={i} style={{ margin: 0 }}>• {s.name}</p>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}

                  {isLoading && (
                    <div className="message message-assistant">
                      <div className="loading-dots">
                        <span /><span /><span />
                      </div>
                    </div>
                  )}
                </div>

                <div className="chat-input-bar">
                  <input
                    type="text"
                    className="input"
                    value={question}
                    onChange={e => setQuestion(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleAsk()}
                    placeholder="输入问题，例如：教育部发布了哪些政策？"
                  />
                  <button
                    className="btn"
                    onClick={handleAsk}
                    disabled={isLoading || !question.trim()}
                  >
                    发送
                  </button>
                </div>
              </div>

              <div className="card chat-sidebar">
                <div className="card-header">
                  <h3 className="card-title">推荐问题</h3>
                </div>
                <div className="card-body">
                  <div className="suggestion-list">
                    {[
                      '教育部发布了哪些政策？',
                      '人工智能发展规划涉及哪些方面？',
                      '上海市的教育政策有哪些？',
                      '哪些政策与教师队伍建设相关？',
                      '教育信息化的目标是什么？'
                    ].map((q, i) => (
                      <button
                        key={i}
                        onClick={() => setQuestion(q)}
                        className="suggestion-chip"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
