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

interface SelectedElement {
  type: 'node' | 'edge'
  id: string
  label?: string
  nodeType?: string
  source?: string
  target?: string
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

export default function App() {
  const [activeTab, setActiveTab] = useState<'upload' | 'graph' | 'chat'>('upload')
  const [isLoading, setIsLoading] = useState(false)

  // Upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadResult, setUploadResult] = useState<IngestResponse | null>(null)
  const [uploadMode, setUploadMode] = useState<'incremental' | 'overwrite'>('incremental')

  // Graph state
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [selectedElement, setSelectedElement] = useState<SelectedElement | null>(null)
  const [themeTags, setThemeTags] = useState<Array<{ theme: string; news_count: number }>>([])
  const [selectedTheme, setSelectedTheme] = useState<string>('')
  const [selectedRelTypes, setSelectedRelTypes] = useState<string[]>([])
  const [availableRelTypes, setAvailableRelTypes] = useState<string[]>([])
  const graphRef = useRef<Graph | null>(null)
  const graphContainerRef = useRef<HTMLDivElement>(null)

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

  // Initialize G6 graph when switching to graph tab
  useEffect(() => {
    if (activeTab === 'graph' && graphContainerRef.current && graphData && !graphRef.current) {
      const nodes = graphData.elements.nodes.map(n => ({
        id: n.data.id,
        label: n.data.label || n.data.id,
        data: {
          ...n.data,
          type: n.data.type,
        },
      }))

      // 提取所有关系类型
      const relTypes = Array.from(new Set(graphData.elements.edges.map(e => e.data.type || 'REL'))).sort()
      setAvailableRelTypes(relTypes)
      if (selectedRelTypes.length === 0) {
        setSelectedRelTypes(relTypes)
      }

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
            size: 32,
            fill: function(this: Graph, d: { type?: string }) {
              return NODE_COLORS[d.type as keyof typeof NODE_COLORS] || '#8E8E93'
            },
            stroke: '#1C1C1E',
            lineWidth: 2,
            labelFill: '#FFFFFF',
            labelFontSize: 10,
            labelFontWeight: 500,
            labelOffsetY: 8,
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
        },
        behaviors: [
          'drag-canvas',
          'zoom-canvas',
          'drag-node',
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
      graph.on('node:click', (e: Event) => {
        const event = e as unknown as { target: { id: string } }
        const nodeId = event.target?.id
        if (!nodeId) return

        console.log('Node clicked:', nodeId)

        const nodeData = graph.getNodeData(nodeId) as { label?: string; type?: string; [key: string]: unknown } | undefined
        if (!nodeData) return

        handleNodeSelect(nodeId)
        graph.setElementState(nodeId, 'selected')

        setSelectedElement({
          type: 'node',
          id: nodeId,
          label: nodeData.label || nodeId,
          nodeType: nodeData.type || 'Unknown',
          data: nodeData
        })
      })

      // Handle edge click
      graph.on('edge:click', (e: Event) => {
        const event = e as unknown as { target: { id: string } }
        const edgeId = event.target?.id
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
      graph.on('canvas:click', (e: Event) => {
        // Stop propagation to avoid triggering other handlers
        e.stopPropagation()
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
      const { data } = await api.get<GraphData>('/api/v1/graph')
      console.log('Graph data loaded:', data.elements.nodes.length, 'nodes,', data.elements.edges.length, 'edges')
      setGraphData(data)
    } catch (e) {
      console.error('Failed to load graph:', e)
    }
  }, [])

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
      const { data } = await api.get<GraphData>(`/api/v1/graph/theme/${encodeURIComponent(theme)}`)
      console.log('Theme graph loaded:', data.elements.nodes.length, 'nodes,', data.elements.edges.length, 'edges')
      setGraphData(data)
      // Destroy old graph to force re-render
      if (graphRef.current) {
        graphRef.current.destroy()
        graphRef.current = null
      }
    } catch (e) {
      console.error('Failed to load graph by theme:', e)
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Load theme tags when graph tab is active
  useEffect(() => {
    if (activeTab === 'graph') {
      loadThemeTags()
    }
  }, [activeTab, loadThemeTags])

  // 当关系筛选变化时更新图谱显示
  useEffect(() => {
    const graph = graphRef.current
    if (!graph || !graphData) return

    try {
      const allEdges = graphData.elements.edges.map((e) => {
        const isCorrelation = e.data.type === 'CORRELATED_WITH'
        const isRel = e.data.type === 'REL'
        const labelText = isCorrelation
          ? `相似度: ${(Number(e.data.score) || 0).toFixed(2)}`
          : e.data.label || e.data.type || ''
        return {
          id: e.data.id,
          source: e.data.source,
          target: e.data.target,
          label: labelText,
          data: { ...e.data },
          style: isCorrelation
            ? {
                stroke: '#FF375F',
                lineWidth: 3,
                lineDash: [8, 4],
                endArrow: false,
                labelText,
                labelFill: '#FF375F',
                labelFontSize: 11,
                labelFontWeight: 'bold' as const,
                labelBackgroundFill: 'rgba(28,28,30,0.9)',
                labelBackgroundRadius: 4,
                halo: true,
                haloStroke: '#FF375F',
                haloLineWidth: 4,
                haloOpacity: 0.3,
                visibility: selectedRelTypes.includes(e.data.type || 'REL') ? ('visible' as const) : ('hidden' as const),
              }
            : isRel
              ? {
                  stroke: 'rgba(255,255,255,0.25)',
                  lineWidth: 1,
                  endArrow: true,
                  endArrowSize: 6,
                  labelText,
                  labelFill: 'rgba(255,255,255,0.5)',
                  labelFontSize: 9,
                  labelBackgroundFill: 'rgba(28,28,30,0.8)',
                  labelBackgroundRadius: 4,
                  visibility: selectedRelTypes.includes(e.data.type || 'REL') ? ('visible' as const) : ('hidden' as const),
                }
              : {
                  stroke: 'rgba(255,255,255,0.3)',
                  lineWidth: 1,
                  endArrow: true,
                  labelText,
                  labelFill: 'rgba(255,255,255,0.6)',
                  labelFontSize: 9,
                  labelBackgroundFill: 'rgba(28,28,30,0.8)',
                  labelBackgroundRadius: 4,
                  visibility: selectedRelTypes.includes(e.data.type || 'REL') ? ('visible' as const) : ('hidden' as const),
                },
        }
      })
      graph.setData({ nodes: graph.getData().nodes, edges: allEdges })
      graph.render()
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
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('schema_name', 'MOE_News')
      formData.append('mode', uploadMode)

      const { data } = await api.post<IngestResponse>('/api/v1/json/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      setUploadResult(data)
      if (data.status === 'completed' && data.processed_items > 0) {
        await loadGraph()
      }
    } catch (e) {
      console.error('Upload failed:', e)
      alert('上传失败，请检查服务是否正常运行')
    } finally {
      setIsLoading(false)
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
